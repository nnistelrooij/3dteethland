from pathlib import Path
from typing import Any, List, Dict, Optional, Tuple

import numpy as np
import torch
from torchtyping import TensorType

from teethland import PointTensor
from teethland.datamodules.teethbinseg import TeethBinSegDataModule
from teethland.data.datasets import TeethLandDataset
import teethland.data.transforms as T


class TeethLandDataModule(TeethBinSegDataModule):
    """Implements data module that loads meshes and landmarks of the 3DTeethLand challenge."""

    def __init__(
        self,
        landmarks_root: Path,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.landmarks_root = Path(landmarks_root)

    def _files(
        self,
        stage: str,
        exclude: List[str]=[
            # missing tooth segmentation
            'K4BAII5F_upper',
            '87N5YSES_upper',
            '67PV9M7X_lower',
            # missing more than one landmark
            'S0AON6PZ_lower',
        ],
    ):
        seg_files = super()._files(stage, exclude=exclude)
        seg_stems = set([fs[0].stem for fs in seg_files])
        
        landmark_files = sorted(self.landmarks_root.glob('**/*.json'))
        landmark_files = [f.relative_to(self.landmarks_root) for f in landmark_files]
        landmark_stems = set([f.name.split('__')[0] for f in landmark_files])

        ann_stems = seg_stems & landmark_stems
        files = []
        for stem in sorted(ann_stems):
            seg_fs = [fs for fs in seg_files if fs[0].stem == stem][0]
            landmark_f = [f for f in landmark_files if f.name.split('__')[0] == stem][0]
            files.append((*seg_fs, landmark_f))

        return files

    def setup(self, stage: Optional[str]=None):
        rng = np.random.default_rng(self.seed)
        default_transforms = T.Compose(
            T.UniformDensityDownsample(self.uniform_density_voxel_size, inplace=True),
            T.GenerateProposals(self.proposal_points, self.max_proposals, rng=rng),
            self.default_transforms,
        )

        if stage is None or stage == 'fit':
            files = self._files('fit')
            print('Total number of files:', len(files))
            train_files, val_files = self._split(files)           
                                      
            train_transforms = T.Compose(
                T.RandomXAxisFlip(rng=rng),
                T.RandomScale(rng=rng),
                T.RandomZAxisRotate(rng=rng),
                default_transforms,
            )

            self.train_dataset = TeethLandDataset(
                stage='fit',
                seg_root=self.root,
                landmarks_root=self.landmarks_root,
                files=train_files,
                clean=self.clean,
                transform=train_transforms,
            )
            self.val_dataset = TeethLandDataset(
                stage='fit',
                seg_root=self.root,
                landmarks_root=self.landmarks_root,
                files=val_files,
                clean=self.clean,
                transform=default_transforms,
            )
    
    @property
    def num_classes(self) -> int:
        return 5

    def collate_fn(
        self,
        batch: List[Dict[str, TensorType[..., Any]]],
    ) -> Tuple[
        Path,
        TensorType['B', torch.bool],
        PointTensor,
        Tuple[PointTensor, PointTensor],       
    ]:
        scan_file, is_lower, x, points = super().collate_fn(batch)

        batch_dict = {key: [d[key] for d in batch] for key in batch[0]}        
        landmark_counts = torch.cat([
            torch.bincount(lands[:, 4].long(), minlength=points.shape[0])
            for lands, points in zip(batch_dict['landmarks'], batch_dict['points'])
        ])
        landmarks = PointTensor(
            coordinates=torch.cat(batch_dict['landmarks'])[:, :3],
            features=torch.cat(batch_dict['landmarks'])[:, 3].long(),
            batch_counts=landmark_counts,
        )

        return scan_file, is_lower, x, (landmarks, points)
    
    def transfer_batch_to_device(
        self,
        batch,
        device: torch.device,
        dataloader_idx: int,
    ) -> Tuple[PointTensor, Tuple[PointTensor, PointTensor]]:
        self.scan_file = batch[0]
        self.is_lower = batch[1].to(device)

        x, (landmarks, points) = batch[2:]
        x = x.to(device)
        landmarks = landmarks.to(device) if landmarks else landmarks
        points = points.to(device)

        return x, (landmarks, points)
