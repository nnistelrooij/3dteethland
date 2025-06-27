from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pymeshlab
import torch
from torchtyping import TensorType

from teethland import PointTensor
from teethland.datamodules.teethinstseg import TeethInstSegDataModule
from teethland.data.datasets import TeethSegDataset
import teethland.data.transforms as T
from teethland.visualization import draw_point_clouds


class TeethAlignDataModule(TeethInstSegDataModule):

    def __init__(
        self,
        out_dir: Optional[Path]=None,
        **dm_cfg,
    ):
        super().__init__(**dm_cfg)

        self.out_dir = out_dir

    @property
    def num_channels(self) -> int:
        return 6

    def setup(self, stage: Optional[str]=None):
        rng = np.random.default_rng(self.seed)
        
        files = self._files(stage, exclude=[
            # Teeth3DS
            '0142CYK4_upper', '014WKP6A_upper', 'IUIE4BYI_upper', 'SZQ66Y5A_lower',
            'THGBYHX3_lower', '00OMSZGW_upper', '0132CR0A_upper', '015KRWDT_lower',
            '01CDZ2WA_upper', '01DXDP2R_upper', '01F4RGN8_lower', '01FVAE34_lower',
            '01J4YSDX_lower', '01M0RWA6_lower', '01MAU84T_lower', '0JN50XQR_lower',
            '0Y047OQK_lower', '230CAJXL_lower', '28Q7CM3T_upper', '2AK5QKZ7_lower',
            '2PNHPSAX_lower', '3OHU0Q5V_upper', '3Y8A14TI_lower', '4G9LHQ2X_lower',
            '4G9LHQ2X_upper', '4J24X0ES_upper', '4W9X0QQI_lower', '4W9X0QQI_upper',
            '6PNM29W7_lower', '9M60SBBP_lower', 'AY0DEPFN_lower', 'B5GFZIRW_lower',
            'BC0OWLNW_lower', 'BWWLPBF7_upper', 'CVXIS1C2_lower', 'E552BK9X_lower',
            'EVWJL2PL_upper', 'F92OKIOI_lower', 'GSICB2I6_lower', 'H79I7NHI_lower',
            'IFVHVDFO_upper', 'IUIE4BYI_lower', 'K23X2RAU_upper', 'KNIH5AER_lower',
            'KNIH5AER_upper', 'KSHNN3DV_lower', 'LSMGKLAH_upper', 'MGYSCSI9_lower',
            'QOCAWJXM_lower', 'R7SB5B5N_lower', 'S0AON6PZ_lower', 'S0AON6PZ_upper',
            'T21AWMTR_lower', 'TZRY3HIX_lower', 'VY1R353X_lower', 'W33KWDRX_lower',
            'WBR4UERA_lower', 'WRT7Y7P9_lower', 'WRT7Y7P9_upper', 'YJXIBVIM_lower',
            'ameziani_lower', '016FSM14_lower', '01FPTYH2_lower',
            # China scans
            '20221011_full_lower', '202210151_full_lower', '202210151_full_upper', '20221015_full_lower',
            '20221016_full_lower', '20221020_full_lower', '20221020_full_upper', '20221031_full_upper',
            '20221112_full_upper', '20221115_full_upper', '20221116_full_lower', '20221118_full_upper',
            '20221212_full_lower', '2022221_full_upper', '2022227_full_lower', '2022227_full_upper',
            '2022312_full_lower', '2022312_full_upper', '2022315_full_lower', '20223261_full_lower',
            '202235 - with implant_full_lower', '2022411_full_lower', '2022519_full_lower', '2022519_full_upper',
            '2022526_full_upper', '2022527_full_lower', '2022616_full_lower', '20226181_full_upper',
            '202261_full_lower', '2022625_full_lower', '2022627_full_lower', '2022710_full_lower',
            '2022710_full_upper', '2022716_full_lower', '2022720_full_lower', '2022720_full_upper',
            '2022722_full_upper', '2022724_full_lower', '2022724_full_upper', '2022727_full_upper',
            '202278_full_lower', '2022815_full_lower', '2022815_full_upper', '202285_full_lower',
            '202285_full_upper', '2022951_full_lower', '202295_full_lower', '202295_full_upper',
            '2023110_full_lower', '2023114_full_lower', '2023114_full_upper', '2023211_full_upper',
            '2023214_full_lower', '2023311_full_lower', '202333_full_upper', '2023422_full_lower',
            '202342_full_lower', '2023717_full_upper', '2023718_full_lower', '202378_full_lower',
            '2023818_full_lower', '2023928_full_upper', '202414_full_lower', '202456_full_upper',
        ] if stage == 'fit' else [])
        print('Total number of files:', len(files))

        if stage is None or stage == 'fit':
            train_files, val_files = self._split(files)

            transforms = T.Compose(
                T.RandomPartial(rng=rng) if self.rand_partial else dict,
                T.RandomRotate(rng=rng),
                self.default_transforms,
            )

            self.train_dataset = TeethSegDataset(
                stage='fit',
                root=self.root,
                files=train_files,
                norm=True,
                clean=False,
                transform=transforms,
                pre_transform=T.AlignUpForward(),
            )
            self.val_dataset = TeethSegDataset(
                stage='fit',
                root=self.root,
                files=val_files,
                norm=True,
                clean=False,
                transform=transforms,
                pre_transform=T.AlignUpForward(),
            )
        elif stage is None or stage == 'predict':
            self.pred_dataset = TeethSegDataset(
                stage='predict',
                root=self.root,
                files=files,
                norm=True,
                clean=True,
                transform=self.default_transforms,
            )

    def save_aligned_mesh(
        self,
        affine: TensorType[4, 4, torch.float32],
    ):
        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(str(self.root / self.scan_file))
        vertices = ms.current_mesh().vertex_matrix()
        triangles = ms.current_mesh().face_matrix()

        affine = (affine @ self.affine).cpu().numpy()
        vertices_homo = np.column_stack((vertices, np.ones_like(vertices[:, :1])))
        vertices_aligned = (vertices_homo @ affine.T)[:, :3]

        mesh = pymeshlab.Mesh(
            vertex_matrix=vertices_aligned,
            face_matrix=triangles,
        )
        ms.add_mesh(mesh)

        out_file = self.out_dir / self.scan_file.split('/')[-1]
        ms.save_current_mesh(str(out_file))

    def collate_fn(
        self,
        batch: List[Dict[str, TensorType[..., Any]]],
    ):
        batch_dict = {key: [d[key] for d in batch] for key in batch[0]}

        scan_file = batch_dict['scan_file'][0]
        affine = batch_dict['affine'][0]

        # collate input points and features
        point_counts = torch.stack(batch_dict['point_count'])
        x = PointTensor(
            coordinates=torch.cat(batch_dict['points']),
            features=torch.cat(batch_dict['features']),
            batch_counts=point_counts,
        )
        x = x[self.collate_downsample(
            batch_dict['point_count'],
            batch_dict['ud_downsample_idxs'],
            batch_dict['ud_downsample_count'],
        )]

        if 'dir_up' not in batch_dict:
            return scan_file, affine, x

        dir_up = torch.stack(batch_dict['dir_up'])
        dir_fwd = torch.stack(batch_dict['dir_fwd'])
        trans = torch.stack(batch_dict['trans'])

        instances = PointTensor(
            coordinates=torch.cat(batch_dict['points']),
            features=torch.cat(batch_dict['instances']) - 1,
            batch_counts=point_counts,
        )
        instances = instances[self.collate_downsample(
            batch_dict['point_count'],
            batch_dict['ud_downsample_idxs'],
            batch_dict['ud_downsample_count'],
        )]
        
        return x, (dir_up, dir_fwd, trans, instances)

    def transfer_batch_to_device(
        self,
        batch,
        device: torch.device,
        dataloader_idx: int,
    ) -> Union[
        PointTensor,
        Tuple[PointTensor, Tuple[
            TensorType[3, torch.float32],
            TensorType[3, torch.float32],
            TensorType[3, torch.float32],
            PointTensor,
        ]],
    ]:
        if self.trainer.state.fn == 'predict':
            self.scan_file = batch[0]
            self.affine = batch[1].to(device)

            return batch[2].to(device)
        
        x, (dir_up, dir_fwd, trans, instances) = batch
        x = x.to(device)
        dir_up = dir_up.to(device)
        dir_fwd = dir_fwd.to(device)
        trans = trans.to(device)
        instances = instances.to(device)

        return x, (dir_up, dir_fwd, trans, instances)
