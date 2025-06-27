from typing import Any, Dict, Tuple

import torch
from torch.nn import ModuleList
from torchmetrics.classification import (
    MulticlassF1Score,
    BinaryF1Score,
)
from torchtyping import TensorType

from teethland import PointTensor
from teethland.models.binsegnet import BinSegNet
import teethland.nn as nn


class MultiSegNet(BinSegNet):
    """
    Implements network for tooth instance binary segmentation.
    """

    def __init__(
        self,
        num_classes,
        **model_args: Dict[str, Any],
    ):
        super().__init__(num_classes=num_classes, **model_args)

        model_args.pop('checkpoint_path', None)
        self.model = nn.StratifiedTransformer(
            out_channels=num_classes, **model_args,
        )

        self.seg_criterion = nn.MultiSegmentationLoss(
            ce_weight=0.0, dice_weight=1.0, focal_weight=1.0,
        )

        self.dice_val = MulticlassF1Score(num_classes=11, average='macro', ignore_index=0)
        self.dices = ModuleList([BinaryF1Score() for _ in range(num_classes)])

    def training_step(
        self,
        batch: Tuple[PointTensor, PointTensor],
        batch_idx: int,
    ) -> TensorType[torch.float32]:
        x, labels = batch

        seg = self(x)

        loss = self.seg_criterion(seg, labels)

        log_dict = {
            'loss/train': loss,
        }
        self.log_dict(log_dict, batch_size=x.batch_size, sync_dist=True)

        return loss

    def validation_step(
        self,
        batch: Tuple[PointTensor, PointTensor],
        batch_idx: int,
    ):
        x, labels = batch

        seg = self(x)

        loss = self.seg_criterion(seg, labels)

        pred = seg.F.argmax(dim=-1)
        for i, dice in enumerate(self.dices):
            dice((pred == i).long(), (labels.F == i).long())

        log_dict = {
            'loss/val': loss,
            **{f'dice/val_{i}': dice for i, dice in enumerate(self.dices)},
            'dice/val': self.dice_val(pred, labels.F),
        }
        self.log_dict(log_dict, batch_size=x.batch_size, sync_dist=True)
    
    def on_load_checkpoint(self, checkpoint: dict) -> None:
        state_dict = checkpoint["state_dict"]
        model_state_dict = self.state_dict()
        is_changed = False
        for k in state_dict:
            if k in model_state_dict:
                if state_dict[k].shape != model_state_dict[k].shape:
                    print(f"Skip loading parameter: {k}, "
                                f"required shape: {model_state_dict[k].shape}, "
                                f"loaded shape: {state_dict[k].shape}")
                    state_dict[k] = model_state_dict[k]
                    is_changed = True
            else:
                print(f"Dropping parameter {k}")
                is_changed = True

        if is_changed:
            checkpoint.pop("optimizer_states", None)