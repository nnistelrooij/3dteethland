from typing import Any, Dict, List, Tuple

import pytorch_lightning as pl
import torch
from torch.optim.lr_scheduler import _LRScheduler
from torchmetrics.classification import (
    BinaryF1Score,    
    BinaryJaccardIndex,
)
from torchtyping import TensorType

from teethland import PointTensor
import teethland.nn as nn
from teethland.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearWarmupLR,
)
from teethland.visualization import draw_point_clouds


class BinSegNet(pl.LightningModule):
    """
    Implements network for tooth instance binary segmentation.
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
        self.model = nn.StratifiedTransformer(
            out_channels=[1 for _ in range(num_classes - 1)], **model_args,
        )

        self.seg_criterion = nn.BinarySegmentationLoss(
            bce_weight=0.0, dice_weight=1.0, focal_weight=1.0,
        )

        self.dice_train = BinaryF1Score()
        self.dice_val = BinaryF1Score()
        self.iou = BinaryJaccardIndex()

        self.num_classes = num_classes
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.warmup_epochs = warmup_epochs

    def forward(
        self,
        x: Tuple[PointTensor, ...],
    ) -> PointTensor:
        _, segs = self.model(x)

        return segs

    def training_step(
        self,
        batch: Tuple[PointTensor, PointTensor],
        batch_idx: int,
    ) -> TensorType[torch.float32]:
        x, labels = batch

        segs = self(x)

        loss = 0
        for i, seg in enumerate(segs, 1):
            labels_i = labels.new_tensor(features=(labels.F >= i).to(int) - 1)
            loss = loss + self.seg_criterion(seg, labels_i)
        
        self.dice_train(
            (segs[-1].F[:, 0] >= 0).long(),
            (labels.F == (self.num_classes - 1)).long(),
        )            

        log_dict = {
            'loss/train': loss,
            'dice/train': self.dice_train,
        }
        self.log_dict(log_dict, batch_size=x.batch_size, sync_dist=True)

        return loss

    def validation_step(
        self,
        batch: Tuple[PointTensor, PointTensor],
        batch_idx: int,
    ):
        x, labels = batch

        segs = self(x)

        loss = 0
        for i, seg in enumerate(segs, 1):
            labels_i = labels.new_tensor(features=(labels.F >= i).to(int) - 1)
            loss = loss + self.seg_criterion(seg, labels_i)
        
        self.dice_val(
            (segs[-1].F[:, 0] >= 0).long(),
            (labels.F == (self.num_classes - 1)).long(),
        )

        log_dict = {
            'loss/val': loss,
            'dice/val': self.dice_val,
        }
        self.log_dict(log_dict, batch_size=x.batch_size, sync_dist=True)

    def configure_optimizers(self) -> Tuple[
        List[torch.optim.Optimizer],
        List[_LRScheduler],
    ]:
        opt = torch.optim.AdamW(
            params=self.model.param_groups(self.lr),
            weight_decay=self.weight_decay,
        )

        non_warmup_epochs = self.epochs - self.warmup_epochs
        sch = CosineAnnealingLR(opt, T_max=non_warmup_epochs, min_lr_ratio=0.0)
        sch = LinearWarmupLR(sch, self.warmup_epochs, init_lr_ratio=0.01)

        return [opt], [sch]
