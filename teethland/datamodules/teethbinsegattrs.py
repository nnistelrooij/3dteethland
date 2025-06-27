from pathlib import Path
from typing import Any, List, Dict, Optional, Tuple, Union

import numpy as np
import torch
from torchtyping import TensorType
from torch_scatter import scatter_max

from teethland import PointTensor
from teethland.datamodules.teethseg import TeethSegDataModule
from teethland.data.datasets import TeethSegDataset
import teethland.data.transforms as T


class TeethBinSegWithAttributesDataModule(TeethSegDataModule):
    """Implements data module that loads tooth crops and segmentations of the 3DTeethLand challenge."""

    def __init__(
        self,
        uniform_density_voxel_size: int,
        random_partial: bool,
        proposal_points: int,
        max_proposals: int,
        label_as_instance: bool,
        with_color: bool,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.default_transforms = T.Compose(
            T.XYZAsFeatures(),
            T.NormalAsFeatures(),
            (T.ColorAsFeatures() if with_color else dict),
            T.CentroidOffsetsAsFeatures(),
            T.ToTensor(),
        )

        self.uniform_density_voxel_size = uniform_density_voxel_size[1]
        self.rand_partial = random_partial
        self.proposal_points = proposal_points
        self.max_proposals = max_proposals
        self.label_as_instance = label_as_instance
        self.with_color = with_color

    def _files(
        self,
        stage: str,
        exclude: List[str]=[],
    ) -> Union[List[Path], List[Tuple[Path, Path]]]:
        return super()._files(stage, exclude)

    def setup(self, stage: Optional[str]=None):
        rng = np.random.default_rng(self.seed)
        default_transforms = T.Compose(
            T.UniformDensityDownsample(self.uniform_density_voxel_size, inplace=True),
            T.GenerateProposals(
                self.proposal_points, self.max_proposals,
                rng=rng, label_as_instance=self.label_as_instance,
            ),
            self.default_transforms,
        )

        if stage is None or stage == 'fit':
            files = self._files('fit')
            print('Total number of files:', len(files))
            train_files, val_files = self._split(files)
                                      
            train_transforms = T.Compose(
                T.UniformDensityDownsample(self.uniform_density_voxel_size, inplace=True),
                T.RandomPartial(
                    rng=rng, min_points=self.proposal_points,
                    do_translate=False, do_single_component=False,
                ) if self.rand_partial else dict,
                T.RandomXAxisFlip(rng=rng),
                T.RandomScale(rng=rng),
                T.RandomZAxisRotate(rng=rng),
                T.RandomShiftCentroids(rng=rng),
                T.GenerateProposals(
                    self.proposal_points, self.max_proposals,
                    rng=rng, label_as_instance=self.label_as_instance,
                ),
                self.default_transforms,
            )

            self.train_dataset = TeethSegDataset(
                stage='fit',
                root=self.root,
                files=train_files,
                norm=self.norm,
                clean=self.clean,
                transform=train_transforms,
            )
            self.val_dataset = TeethSegDataset(
                stage='fit',
                root=self.root,
                files=val_files,
                norm=self.norm,
                clean=self.clean,
                transform=default_transforms,
            )
    
    @property
    def num_channels(self) -> int:
        return 9 + 3 * self.with_color
    
    @property
    def num_classes(self) -> int:
        return 3 if self.label_as_instance else 2

    def collate_fn(
        self,
        batch: List[Dict[str, TensorType[..., Any]]],
    ) -> Tuple[
        Path,
        TensorType['B', torch.bool],
        PointTensor,
        Tuple[PointTensor, PointTensor],
    ]:
        batch_dict = {key: [d[key] for d in batch] for key in batch[0]}

        scan_file = batch_dict['scan_file'][0]
        is_lower = torch.stack(batch_dict['is_lower'])

        # collate input points and features
        point_counts = torch.cat(batch_dict['point_count'])
        x = PointTensor(
            coordinates=torch.cat(batch_dict['points']).reshape(-1, 3),
            features=torch.cat(batch_dict['features']).reshape(-1, self.num_channels),
            batch_counts=point_counts,
        )

        instance_counts = torch.stack(batch_dict['instance_count'])
        if self.label_as_instance:
            points = x.new_tensor(features=torch.cat(batch_dict['labels']).flatten())
        else:
            points = x.new_tensor(features=torch.cat(batch_dict['labels']).flatten() - 1)
            instance_offsets = torch.arange(instance_counts.sum())
            instance_offsets = instance_offsets.repeat_interleave(point_counts)
            points.F[points.F >= 0] += instance_offsets[points.F >= 0]

        # put individual caries instances as indices 2 or more in points tensor
        caries_inst_attributes = []
        for i, attrs in enumerate(batch_dict['attributes']):
            caries_insts = torch.unique(attrs // 100, return_inverse=True)[1] - 1
            caries_inst_attributes.append(caries_insts)

        caries_instances = torch.cat(caries_inst_attributes)
        instance_counts = [torch.unique(insts).shape[0] - 1 for insts in caries_instances]
        instance_counts = torch.tensor(instance_counts).to(point_counts)
        instance_offsets = instance_counts.cumsum(dim=0) - instance_counts
        instance_offsets = instance_offsets.repeat_interleave(point_counts)
        instance_offsets = instance_offsets.reshape(caries_instances.shape[0], -1)
        caries_instances[caries_instances >= 0] += instance_offsets[caries_instances >= 0]
        caries_instances = caries_instances.flatten()

        points.F[caries_instances >= 0] = 2 + caries_instances[caries_instances >= 0]

        # make a tensor of individual caries instances with caries types as features
        instance_labels = scatter_max(
            src=torch.cat(batch_dict['attributes']).flatten() % 100,
            index=caries_instances.flatten() + 1,
            dim=0,
        )[0][1:]
        instances = PointTensor(
            coordinates=torch.zeros(instance_counts.sum(), 3).to(points.C),
            features=instance_labels,
            batch_counts=instance_counts,
        )

        return scan_file, is_lower, x, (points, instances)
    
    def transfer_batch_to_device(
        self,
        batch,
        device: torch.device,
        dataloader_idx: int,
    ) -> Tuple[
        PointTensor, 
        Tuple[PointTensor, PointTensor],
    ]:
        self.scan_file = batch[0]
        self.is_lower = batch[1].to(device)

        x, (points, instances) = batch[2:]
        x = x.to(device)
        points = points.to(device)
        instances = instances.to(device)

        return x, (points, instances)
