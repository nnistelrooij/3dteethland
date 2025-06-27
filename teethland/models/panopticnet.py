from typing import Any, Dict, List, Tuple

import pytorch_lightning as pl
import torch
from torch.optim.lr_scheduler import _LRScheduler
from torchmetrics.classification import (
    BinaryF1Score,    
    BinaryJaccardIndex,
    MulticlassF1Score,
)
from torchtyping import TensorType

from teethland import PointTensor
from teethland.cluster import learned_region_cluster
from teethland.metrics import ToothF1Score
import teethland.nn as nn
from teethland.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearWarmupLR,
)
from teethland.visualization import draw_point_clouds


class PanopticNet(pl.LightningModule):
    """
    Implements DentalNet for point cloud instance segmentation and tooth type classification.
    """

    def __init__(
        self,
        lr: float,
        weight_decay: float,
        epochs: int,
        warmup_epochs: int,
        **model_args: Dict[str, Any],
    ):
        super().__init__()

        model_args.pop('checkpoint_path', None)
        self.panoptic_model = nn.StratifiedTransformer(
            out_channels=[6, 1, None, None],
            **model_args,
        )
        with torch.no_grad():
            output_layer = self.panoptic_model.heads[0].linear
            # offsets
            output_layer.weight[:, :3].fill_(0)
            output_layer.bias[:3].fill_(0)
            # sigmas
            output_layer.weight[:, 3:].fill_(0)
            output_layer.bias[3:].fill_(1)

        self.fdi_model = nn.MaskedAveragePooling(
            num_features=self.panoptic_model.out_channels[-2],
            out_channels=model_args['num_classes'][0],
        )
        self.type_model = nn.MaskedAveragePooling(
            num_features=self.panoptic_model.out_channels[-1],
            out_channels=model_args['num_classes'][1],
        )

        self.instance_criterion = nn.SpatialEmbeddingLoss()
        self.identify_criterion = nn.IdentificationLoss()

        self.dice = BinaryF1Score()
        self.iou = BinaryJaccardIndex()
        self.fdi_f1 = MulticlassF1Score(num_classes=model_args['num_classes'][0])
        self.type_f1 = MulticlassF1Score(num_classes=model_args['num_classes'][1])
        self.tooth_f1 = ToothF1Score()

        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.warmup_epochs = warmup_epochs

    def forward(
        self,
        x: PointTensor,
        labels: PointTensor,
    ) -> Tuple[PointTensor, PointTensor, PointTensor, PointTensor, PointTensor, torch.Tensor, torch.Tensor]:
        x = x[x.cache['cp_downsample_idxs']]
        
        _, (spatial_embeds, seeds, features1, features2) = self.panoptic_model(x)
        prototypes1, classes1 = self.fdi_model(features1, labels)
        prototypes2, classes2 = self.type_model(features2, labels)

        offsets = spatial_embeds.new_tensor(features=spatial_embeds.F[:, :3])
        sigmas = spatial_embeds.new_tensor(features=spatial_embeds.F[:, 3:])

        return offsets, sigmas, seeds, prototypes1, prototypes2, classes1, classes2

    def training_step(
        self,
        batch: Tuple[PointTensor, Tuple[PointTensor, PointTensor]],
        batch_idx: int,
    ) -> TensorType[torch.float32]:
        x, (instances, labels) = batch

        offsets, sigmas, seeds, prototypes1, prototypes2, classes1, classes2 = self(x, labels)

        instance_loss = self.instance_criterion(offsets, sigmas, seeds, labels)
        fdi_loss = self.identify_criterion(
            prototypes1, classes1, instances.new_tensor(features=instances.F[:, 0]))
        type_loss = self.identify_criterion(
            prototypes2, classes2, instances.new_tensor(features=instances.F[:, 1]))

        loss = instance_loss + fdi_loss + type_loss
        self.log_dict({
            'loss/train_instance': instance_loss,
            'loss/train_fdi': fdi_loss,
            'loss/train_type': type_loss,
            'loss/train': loss,
        }, batch_size=x.batch_size)

        return loss

    def validation_step(
        self,
        batch: Tuple[PointTensor, Tuple[PointTensor, PointTensor]],
        batch_idx: int,
    ) -> Tuple[PointTensor, PointTensor]:
        x, (instances, labels) = batch

        offsets, sigmas, seeds, prototypes1, prototypes2, classes1, classes2 = self(x, labels)

        instance_loss = self.instance_criterion(offsets, sigmas, seeds, labels)
        fdi_loss = self.identify_criterion(
            prototypes1, classes1, instances.new_tensor(features=instances.F[:, 0]))
        type_loss = self.identify_criterion(
            prototypes2, classes2, instances.new_tensor(features=instances.F[:, 1]))

        loss = instance_loss + fdi_loss + type_loss
        log_dict = {
            'loss/val_instance': instance_loss,
            'loss/val_fdi': fdi_loss,
            'loss/val_type': type_loss,
            'loss/val': loss,
        }

        if self.trainer.state.fn == 'validate' or self.current_epoch >= 1:
            pred_labels = learned_region_cluster(
                offsets, sigmas, seeds,
            )
            self.dice((pred_labels.F >= 0).long(), (labels.F >= 0).long())
            self.iou((pred_labels.F >= 0).long(), (labels.F >= 0).long())
            
            metric_dict = self.tooth_f1(pred_labels, classes1, labels, instances)
            metric_dict = {f'{k}/val': v for k, v in metric_dict.items()}

            log_dict.update({
                'dice/val': self.dice,
                'iou/val': self.iou,
                **metric_dict,
            })
        
        self.fdi_f1(classes1.F, instances.F[:, 0])
        self.type_f1(classes2.F, instances.F[:, 1])
        log_dict.update({'fdi_f1/val_epoch': self.fdi_f1})
        log_dict.update({'type_f1/val': self.type_f1})

        self.log_dict(log_dict, batch_size=x.batch_size, sync_dist=True)

    def configure_optimizers(self) -> Tuple[
        List[torch.optim.Optimizer],
        List[_LRScheduler],
    ]:
        fdi_params = [
            {'params': g['params'], 'lr': g['lr'], 'name': 'fdi_' + g['name']}
            for g in self.fdi_model.param_groups(self.lr)
        ]
        type_params = [
            {'params': g['params'], 'lr': g['lr'], 'name': 'type_' + g['name']}
            for g in self.type_model.param_groups(self.lr)
        ]
        opt = torch.optim.AdamW(
            params=[
                *self.panoptic_model.param_groups(self.lr),
                *fdi_params,
                *type_params,
            ],
            weight_decay=self.weight_decay,
        )

        non_warmup_epochs = self.epochs - self.warmup_epochs
        sch = CosineAnnealingLR(opt, T_max=non_warmup_epochs, min_lr_ratio=0.0)
        sch = LinearWarmupLR(sch, self.warmup_epochs, init_lr_ratio=0.01)

        return [opt], [sch]
