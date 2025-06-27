from typing import Any, Dict, List, Tuple

import pytorch_lightning as pl
import torch
from torch.optim.lr_scheduler import _LRScheduler
from torchmetrics.classification import (
    BinaryF1Score,    
    MulticlassF1Score,
)
from torchtyping import TensorType

from teethland import PointTensor
import teethland.nn as nn
from teethland.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearWarmupLR,
)
from teethland.visualization import draw_point_clouds


class BinSegAttributesNet(pl.LightningModule):
    """
    Implements network for tooth instance binary segmentation with attribute classification.
    """

    def __init__(
        self,
        num_classes: int,
        lr: float,
        weight_decay: float,
        epochs: int,
        warmup_epochs: int,
        **model_args: Dict[str, Any],
    ):
        super().__init__()

        model_args.pop('checkpoint_path', None)
        self.binseg_model = nn.StratifiedTransformer(
            out_channels=[1, 1, None], **model_args,
        )
        self.identify_model = nn.MaskedAveragePooling(
            num_features=self.binseg_model.out_channels[-1],
            out_channels=4,
        )

        self.seg_criterion = nn.BinarySegmentationLoss(
            bce_weight=0.0, dice_weight=1.0, focal_weight=1.0,
        )
        self.identify_criterion = nn.IdentificationLoss()

        self.dice_train = BinaryF1Score()
        self.dice_val = BinaryF1Score()
        self.attr_f1 = MulticlassF1Score(num_classes=4)

        self.num_classes = num_classes
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.warmup_epochs = warmup_epochs

    def forward(
        self,
        x: PointTensor,
        labels: PointTensor,
    ) -> Tuple[PointTensor, PointTensor, PointTensor, torch.Tensor]:
        _, out = self.binseg_model(x)

        seg1, seg2, features = out

        labels = labels.new_tensor(features=torch.where(labels.F >= 2, labels.F - 2, -1))
        prototypes, classes = self.identify_model(features, labels)

        return seg1, seg2, prototypes, classes

    def training_step(
        self,
        batch: Tuple[PointTensor, Tuple[PointTensor, PointTensor]],
        batch_idx: int,
    ) -> TensorType[torch.float32]:
        x, (points, instances) = batch

        seg1, seg2, prototypes, classes = self(x, points)

        labels1 = points.new_tensor(features=(points.F >= 1).to(int) - 1)
        seg_loss = self.seg_criterion(seg1, labels1)

        labels2 = points.new_tensor(features=(points.F >= 2).to(int) - 1)
        seg_loss = seg_loss + self.seg_criterion(seg2, labels2)

        identify_loss = self.identify_criterion(prototypes, classes, instances)

        loss = seg_loss + 0.1 * identify_loss
        log_dict = {
            'loss/train_seg': seg_loss,
            'loss/train_identify': identify_loss,
            'loss/train': loss,
        }
        self.log_dict(log_dict, batch_size=x.batch_size, sync_dist=True)

        return loss

    def validation_step(
        self,
        batch: Tuple[PointTensor, Tuple[PointTensor, PointTensor]],
        batch_idx: int,
    ):
        x, (points, instances) = batch

        seg1, seg2, prototypes, classes = self(x, points)

        labels1 = points.new_tensor(features=(points.F >= 1).to(int) - 1)
        seg_loss = self.seg_criterion(seg1, labels1)

        labels2 = points.new_tensor(features=(points.F >= 2).to(int) - 1)
        seg_loss = seg_loss + self.seg_criterion(seg2, labels2)

        identify_loss = self.identify_criterion(prototypes, classes, instances)

        self.dice_val(
            (seg2.F[:, 0] >= 0).long(),
            (points.F >= 2).long(),
        )
        log_dict = {
            'loss/val_seg': seg_loss,
            'loss/val_identify': identify_loss,
            'loss/val': seg_loss + 0.1 * identify_loss,
            'dice/val': self.dice_val,
        }

        if classes.F.shape[0] > 0:
            self.attr_f1(classes.F, instances.F)
            log_dict['f1/val'] = self.attr_f1

        self.log_dict(log_dict, batch_size=x.batch_size, sync_dist=True)

    def configure_optimizers(self) -> Tuple[
        List[torch.optim.Optimizer],
        List[_LRScheduler],
    ]:
        opt = torch.optim.AdamW(
            params=[
                *self.binseg_model.param_groups(self.lr),
                *self.identify_model.param_groups(self.lr),
            ],
            weight_decay=self.weight_decay,
        )

        non_warmup_epochs = self.epochs - self.warmup_epochs
        sch = CosineAnnealingLR(opt, T_max=non_warmup_epochs, min_lr_ratio=0.0)
        sch = LinearWarmupLR(sch, self.warmup_epochs, init_lr_ratio=0.01)

        return [opt], [sch]
