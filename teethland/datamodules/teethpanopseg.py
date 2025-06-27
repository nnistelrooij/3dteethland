from typing import Any, Dict, List

from pytorch_lightning.trainer.states import RunningStage
import torch
from torchtyping import TensorType

from teethland import PointTensor
from teethland.datamodules.teethinstseg import TeethInstSegDataModule
from teethland.visualization import draw_point_clouds


class TeethPanopticSegDataModule(TeethInstSegDataModule):
    """Data module to load intraoral scans with teeth pantopic instances."""

    @property
    def num_classes(self) -> List[int]:
        num_fdi_classes = super().num_classes
        num_type_classes = 4

        return [num_fdi_classes, num_type_classes]
    
    def collate_fn(
        self,
        batch: List[Dict[str, TensorType[..., Any]]],
    ):
        batch_dict = {key: [d[key] for d in batch] for key in batch[0]}

        scan_file = batch_dict['scan_file'][0]
        is_lower = torch.stack(batch_dict['is_lower'])

        # collate input points and features
        point_counts = torch.stack(batch_dict['point_count'])
        x = PointTensor(
            coordinates=torch.cat(batch_dict['points']),
            features=torch.cat(batch_dict['features']),
            batch_counts=point_counts,
        )

        x.cache['cp_downsample_idxs'] = self.collate_downsample(
            batch_dict['point_count'],
            batch_dict['ud_downsample_idxs'],
            batch_dict['ud_downsample_count'],
        )
        x.cache['ts_downsample_idxs'] = x.cache['cp_downsample_idxs']

        # collate output points
        points = PointTensor(
            coordinates=torch.cat(batch_dict['points']),
            batch_counts=point_counts,
        )
        if self.trainer.state.stage == RunningStage.PREDICTING:
            return scan_file, is_lower, x, points

        # collate tooth instance centroids and classes
        instance_centroids = [ic[1:] for ic in batch_dict['instance_centroids']]
        instance_labels = [il[1:] for il in batch_dict['instance_labels']]
        instance_types = [il[1:] for il in batch_dict['instance_types']]
        instance_counts = torch.stack(batch_dict['instance_count']) - 1
        instances = PointTensor(
            coordinates=torch.cat(instance_centroids),
            features=torch.column_stack((
                self.teeth_labels_to_classes(torch.cat(instance_labels)),
                torch.cat(instance_types),
            )),
            batch_counts=instance_counts,
        )
        
        # determine gingiva (-1) or tooth instance index (>= 0) for each point
        points.F = torch.cat(batch_dict['instances']) - 1
        instance_offsets = instance_counts.cumsum(dim=0) - instance_counts
        instance_offsets = instance_offsets.repeat_interleave(point_counts)
        points.F[points.F >= 0] += instance_offsets[points.F >= 0]

        # take subsample and remove instances not present in subsample
        points = points[x.cache['ts_downsample_idxs']]
        unique, inverse_idxs = torch.unique(points.F, return_inverse=True)

        instances = instances[unique[unique >= 0]]
        points.F = inverse_idxs - 1
        
        return scan_file, is_lower, x, (instances, points)
