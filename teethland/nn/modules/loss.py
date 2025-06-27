from itertools import filterfalse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtyping import TensorType

from teethland import PointTensor


class BCELoss(nn.Module):

    def __init__(self):
        super().__init__()

        self.criterion = nn.BCEWithLogitsLoss()

    def forward(
        self,
        x: PointTensor,
        y: PointTensor,
    ) -> TensorType[torch.float32]:
        loss = self.criterion(x.F[:, 0], (y.F >= 0).float())

        return loss
    

class BinarySegmentationLoss(nn.Module):
    "Implements binary segmentation loss function."

    def __init__(
        self,
        bce_weight: float=1.0,
        dice_weight: float=0.0,
        focal_weight: float=0.0,
        focal_alpha: float=0.25,
        focal_gamma: float=2.0,
    ) -> None:
        super().__init__()

        self.bce = nn.BCEWithLogitsLoss()

        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.alpha = focal_alpha
        self.gamma = focal_gamma

    def forward(
        self,
        x: PointTensor,
        y: PointTensor
    ) -> TensorType[torch.float32]:
        pred = x.F[:, 0]
        target = (y.F >= 0).float()
        
        loss = self.bce_weight * self.bce(pred, target)

        if self.dice_weight:        
            probs = torch.sigmoid(pred)
            dim = tuple(range(1, len(probs.shape)))
            numerator = 2 * torch.sum(probs * target, dim=dim)
            denominator = torch.sum(probs ** 2, dim=dim) + torch.sum(target ** 2, dim=dim)
            dice_loss = 1 - torch.mean((numerator + 1) / (denominator + 1))

            loss += self.dice_weight * dice_loss

        if self.focal_weight:
            bce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
            pt = torch.exp(-bce_loss)
            focal_loss = self.alpha * (1 - pt)**self.gamma * bce_loss

            loss += self.focal_weight * focal_loss.mean()

        return loss
    

class MultiSegmentationLoss(nn.Module):
    "Implements multiclass segmentation loss function."

    def __init__(
        self,
        ce_weight: float=1.0,
        dice_weight: float=1.0,
        focal_weight: float=0.0,
        focal_alpha: float=0.25,
        focal_gamma: float=2.0,
    ):
        super().__init__()

        self.ce = nn.CrossEntropyLoss()

        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.alpha = focal_alpha
        self.gamma = focal_gamma

    def __call__(
        self,
        x: PointTensor,
        y: PointTensor
    ) -> TensorType[torch.float32]:
        loss = self.ce_weight * self.ce(x.F, y.F)

        if self.dice_weight:
            probs = torch.softmax(x.F, dim=1).T
            probs = probs.reshape(probs.shape[0], -1, x.batch_counts[0])

            target = F.one_hot(y.F, num_classes=x.F.shape[1]).T
            target = target.reshape(target.shape[0], -1, x.batch_counts[0])

            dim = tuple(range(2, len(probs.shape)))
            numerator = 2 * torch.sum(probs * target, dim=dim)
            denominator = torch.sum(probs ** 2, dim=dim) + torch.sum(target ** 2, dim=dim)
            dice_loss = 1 - torch.mean((numerator + 1e-5) / (denominator + 1e-5))

            loss += self.dice_weight * dice_loss

        if self.focal_weight:
            log_p = F.log_softmax(x.F, dim=-1)
            ce = F.nll_loss(log_p, y.F, reduction='none')

            # get true class column from each row
            all_rows = torch.arange(x.F.shape[0])
            log_pt = log_p[all_rows, y.F]

            # compute focal loss
            pt = log_pt.exp()
            focal_loss = (1 - pt)**self.gamma * ce

            loss += self.focal_weight * focal_loss.mean()

        return loss


def mean(l, ignore_nan=False, empty=0):
    """
    nanmean compatible with generators.
    """
    l = iter(l)
    if ignore_nan:
        l = filterfalse(np.isnan, l)
    try:
        n = 1
        acc = next(l)
    except StopIteration:
        if empty == 'raise':
            raise ValueError('Empty mean')
        return empty
    for n, v in enumerate(l, 2):
        acc += v
    if n == 1:
        return acc

    return acc / n


def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts.float() - gt_sorted.float().cumsum(0)
    union = gts.float() + (~gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1:  # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def lovasz_hinge(logits, labels):
    """
    Binary Lovasz hinge loss
      logits: [B, H, W] Variable, logits at each pixel (between -\infty and +\infty)
      labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
      ignore: void class id
    """
    loss = mean(
        lovasz_hinge_flat(log.reshape(-1), lab.reshape(-1))
        for log, lab in zip(logits, labels)
    )

    return loss


def lovasz_hinge_flat(logits, labels):
    """
    Binary Lovasz hinge loss
      logits: [P] Variable, logits at each prediction (between -\infty and +\infty)
      labels: [P] Tensor, binary ground truth labels (0 or 1)
      ignore: label to ignore
    """
    if len(labels) == 0:
        # only void pixels, the gradients should be 0
        return logits.sum() * 0.
    signs = 2. * labels.float() - 1.
    errors = (1. - logits * signs)
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    perm = perm.data
    gt_sorted = labels[perm]
    grad = lovasz_grad(gt_sorted)
    loss = torch.dot(F.relu(errors_sorted), grad)
    return loss
