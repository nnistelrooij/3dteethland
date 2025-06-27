from typing import Any, Dict, List, Optional, Tuple

import pytorch_lightning as pl
import torch
from torch.optim.lr_scheduler import _LRScheduler
from torchtyping import TensorType
from torch_scatter import scatter_mean

from teethland import PointTensor
import teethland.nn as nn
from teethland.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearWarmupLR,
)
from teethland.visualization import draw_point_clouds


class AlignNet(pl.LightningModule):
    """
    Implements network aligning (partial) intra-oral scans
    """

    def __init__(
        self,
        lr: float,
        weight_decay: float,
        epochs: int,
        warmup_epochs: int,
        do_seg: bool,
        **model_args: Dict[str, Any],
    ):
        super().__init__()

        model_args.pop('checkpoint_path', None)
        self.backbone = nn.StratifiedTransformer(
            **model_args, out_channels=1 if do_seg else None,
        )
        self.mlp = nn.MLP(self.backbone.enc_channels, 256, 9)

        self.trans_criterion = torch.nn.SmoothL1Loss()
        self.seg_criterion = nn.BinarySegmentationLoss(bce_weight=1.0, dice_weight=1.0)

        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.warmup_epochs = warmup_epochs
        self.do_seg = do_seg

    def forward(
        self,
        x: PointTensor,
    ) -> Tuple[
        TensorType[3, torch.float32],
        TensorType[3, torch.float32],
        TensorType[3, torch.float32],
        Optional[PointTensor],
    ]:
        out = self.backbone(x)
        encoding = out[0] if self.do_seg else out
        embeddings = scatter_mean(encoding.F, encoding.batch_indices, dim=0)
        embeddings = PointTensor(
            coordinates=torch.zeros(x.batch_size, 3).to(x.C),
            features=embeddings,
        )
        preds = self.mlp(embeddings)

        dir_up = preds.F[:, :3] / torch.linalg.norm(preds.F[:, :3], dim=-1, keepdim=True)
        dir_fwd = preds.F[:, 3:6] / torch.linalg.norm(preds.F[:, 3:6], dim=-1, keepdim=True)
        trans = preds.F[:, 6:]

        return dir_up, dir_fwd, trans, out[1] if self.do_seg else None

    def training_step(
        self,
        batch: Tuple[PointTensor, PointTensor],
        batch_idx: int,
    ):
        x, (dir_up, dir_fwd, trans, points) = batch

        pred_up, pred_fwd, pred_trans, seg = self(x)

        loss_align = (
            2
            - torch.einsum('bi,bi->b', pred_up, dir_up)
            - torch.einsum('bi,bi->b', pred_fwd, dir_fwd)
            + torch.abs(torch.einsum('bi,bi->b', pred_up, pred_fwd))
        ).mean()
        loss_trans = self.trans_criterion(pred_trans, trans)
        loss_seg = self.seg_criterion(seg, points) if self.do_seg else 0

        log_dict = {
            'loss/train_align': loss_align,
            'loss/train_trans': loss_trans,
            'loss/train_seg': loss_seg,
            'loss/train': loss_align + loss_trans + loss_seg,
        }
        self.log_dict(log_dict, batch_size=x.batch_size, sync_dist=True)

        return loss_align + loss_trans + loss_seg

    def validation_step(
        self,
        batch: Tuple[PointTensor, PointTensor],
        batch_idx: int,
    ):
        x, (dir_up, dir_fwd, trans, points) = batch

        pred_up, pred_fwd, pred_trans, seg = self(x)

        loss_align = (
            2
            - torch.einsum('bi,bi->b', pred_up, dir_up)
            - torch.einsum('bi,bi->b', pred_fwd, dir_fwd)
            + torch.abs(torch.einsum('bi,bi->b', pred_up, pred_fwd))
        ).mean()
        loss_trans = self.trans_criterion(pred_trans, trans)
        loss_seg = self.seg_criterion(seg, points) if self.do_seg else 0

        log_dict = {
            'loss/val_align': loss_align,
            'loss/val_trans': loss_trans,
            'loss/val_seg': loss_seg,
            'loss/val': loss_align + loss_trans + loss_seg,
        }
        self.log_dict(log_dict, batch_size=x.batch_size, sync_dist=True)

    def predict_step(
        self,
        x: PointTensor,
        batch_idx: int,
    ):
        pred_up, pred_fwd, pred_trans, _ = self(x)

        # make two vectors orthogonal
        dots = torch.einsum('bi,bi->b', pred_up, pred_fwd)
        if torch.abs(dots[0]) > 0.01:
            print(dots, self.trainer.datamodule.scan_file)
        pred_fwd -= dots[:, None] * pred_up

        # determine non-reflecting rotation matrix to standard basis
        pred_right = torch.cross(pred_fwd, pred_up, dim=-1)
        R = torch.stack((pred_right, pred_fwd, pred_up))[:, 0]
        if torch.linalg.det(R) < 0:
            print('Determinant < 0')
            R = torch.tensor([
                [-1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
            ]).to(R) @ R

        # determine rotation matrix in 3DTeethSeg basis
        R = torch.tensor([
            [-1, 0, 0],
            [0, -1, 0],
            [0, 0, 1],
        ]).to(R) @ R

        # apply translation and determine affine matrix
        T = torch.eye(4).to(R)
        T[:3, :3] = R
        T[:3, 3] = -pred_trans @ R.T
        
        self.trainer.datamodule.save_aligned_mesh(T)

    def configure_optimizers(self) -> Tuple[
        List[torch.optim.Optimizer],
        List[_LRScheduler],
    ]:
        opt = torch.optim.AdamW(
            params=[
                *self.backbone.param_groups(self.lr),
                {
                    'params': self.mlp.parameters(),
                    'lr': self.lr,
                    'name': 'MLP',
                },
            ],
            weight_decay=self.weight_decay,
        )

        non_warmup_epochs = self.epochs - self.warmup_epochs
        sch = CosineAnnealingLR(opt, T_max=non_warmup_epochs, min_lr_ratio=0.0)
        sch = LinearWarmupLR(sch, self.warmup_epochs, init_lr_ratio=0.1)

        return [opt], [sch]
