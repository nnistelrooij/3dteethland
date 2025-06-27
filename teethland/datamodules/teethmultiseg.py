from pathlib import Path
from typing import Any, List, Dict, Tuple

import torch
from torchtyping import TensorType

from teethland import PointTensor
from teethland.datamodules.teethbinseg import TeethBinSegDataModule


class TeethMultiSegDataModule(TeethBinSegDataModule):
    """Implements data module that loads tooth crops and segmentationsof the 3DTeethLand challenge."""
    
    @property
    def num_classes(self) -> int:
        return 11

    def collate_fn(
        self,
        batch: List[Dict[str, TensorType[..., Any]]],
    ) -> Tuple[
        Path,
        TensorType['B', torch.bool],
        PointTensor,
        PointTensor,     
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

        fg_tooth = torch.cat(batch_dict['labels']) == 1
        labels = torch.where(fg_tooth, torch.cat(batch_dict['attributes']) + 1, 0)        
        points = x.new_tensor(features=labels.flatten())

        return scan_file, is_lower, x, points
