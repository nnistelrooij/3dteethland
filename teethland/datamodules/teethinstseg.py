import json
from time import perf_counter
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import cv2
import numpy as np
from numpy.typing import NDArray
from pytorch_lightning.trainer.states import RunningStage
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from torchtyping import TensorType

from teethland import PointTensor
from teethland.datamodules.teethseg import TeethSegDataModule
from teethland.data.datasets import TeethSegDataset
import teethland.data.transforms as T
from teethland.visualization import draw_point_clouds


class TeethInstSegDataModule(TeethSegDataModule):
    """Data module to load intraoral scans with teeth instances."""

    def __init__(
        self,
        batch: Optional[Tuple[int, int]],
        uniform_density_voxel_size: int,
        distinguish_left_right: bool,
        distinguish_upper_lower: bool,
        m3_as_m2: bool,
        random_partial: bool,
        with_color: bool,
        **dm_cfg,
    ):
        super().__init__(**dm_cfg)

        self.default_transforms = T.Compose(
            T.UniformDensityDownsample(uniform_density_voxel_size[0]),
            T.XYZAsFeatures(),
            T.NormalAsFeatures(),
            (T.ColorAsFeatures() if with_color else dict),
            T.ToTensor(),
        )
        
        self.batch = None if batch is None else slice(*batch)
        self.distinguish_left_right = distinguish_left_right
        self.distinguish_upper_lower = distinguish_upper_lower
        self.m3_as_m2 = m3_as_m2
        self.rand_partial = random_partial
        self.is_lower = None
        self.with_color = with_color

    def setup(self, stage: Optional[str]=None):
        rng = np.random.default_rng(self.seed)

        if stage is None or stage == 'fit':
            files = self._files('fit')
            print('Total number of files:', len(files))
            train_files, val_files = self._split(files)

            train_transforms = T.Compose(
                T.Compose(
                    T.RandomPartial(rng, do_translate=False),
                    T.ZScoreNormalize(None, 1),
                    T.PoseNormalize(),
                    T.InstanceCentroids(),
                ) if self.rand_partial else dict,
                T.RandomXAxisFlip(rng=rng),
                T.RandomScale(rng=rng),
                T.RandomZAxisRotate(rng=rng),
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
                transform=self.default_transforms,
            )

        if stage is None or stage == 'predict':
            files = self._files('predict', exclude=[])
            self.pred_dataset = TeethSegDataset(
                stage='predict',
                root=self.root,
                files=files if self.batch is None else files[self.batch],
                norm=self.norm,
                clean=self.clean,
                transform=self.default_transforms,
            )

    @property
    def num_channels(self) -> int:
        return 6 + 3 * self.with_color

    @property
    def num_classes(self) -> int:
        factor = 1 if self.filter in ['lower', 'upper'] or not self.distinguish_upper_lower else 2
        factor *= 2 if self.distinguish_left_right else 1
        number = 7 if self.m3_as_m2 else 8
        return factor * number

    def teeth_labels_to_classes(
        self,
        labels: Union[
            NDArray[np.int64],
            TensorType['N', torch.int64],
        ]
    ) -> Union[
        NDArray[np.int64],
        TensorType['N', torch.int64],
    ]:
        if isinstance(labels, np.ndarray):
            classes = labels.copy()
        elif isinstance(labels, torch.Tensor):
            classes = labels.clone()
        else:
            raise ValueError(
                f'Expected np.ndarray or torch.Tensor, got {type(labels)}.',
            )

        classes[(11 <= labels) & (labels <= 18)] -= 11
        classes[(21 <= labels) & (labels <= 28)] -= 13 if self.distinguish_left_right else 21
        if self.m3_as_m2:
            classes[labels == 18] = 6
            classes[(21 <= labels) & (labels <= 28)] -= 1 if self.distinguish_left_right else 0
            classes[labels == 28] = 13 if self.distinguish_left_right else 6

        classes[(31 <= labels) & (labels <= 38)] -= 31
        classes[(41 <= labels) & (labels <= 48)] -= 33 if self.distinguish_left_right else 41
        if self.m3_as_m2:
            classes[labels == 38] = 6
            classes[(41 <= labels) & (labels <= 48)] -= 1 if self.distinguish_left_right else 0
            classes[labels == 48] = 13 if self.distinguish_left_right else 6
        
        return classes
    
    def determine_rot_matrix(
        self,
        centroids: PointTensor,
        verbose: bool=False,
    ):
        X, Y, _ = centroids.C.T.cpu().numpy()
        _, _, degrees = cv2.fitEllipse(np.column_stack((X, Y)))
        if degrees > 90:
            degrees = degrees - 180
        angle = -degrees / 180 * np.pi
        cosval, sinval = np.cos(angle), np.sin(angle)

        R = np.array([
            [cosval, -sinval, 0, 0],
            [sinval, cosval,  0, 0],
            [0,       0,      1, 0],
            [0,       0,      0, 1],
        ])
        R = torch.from_numpy(R).to(centroids.C)        

        if verbose:
            import matplotlib.pyplot as plt
            new_coords = centroids.C @ R.T
            new_coords = new_coords.cpu().numpy()
            plt.scatter(X, Y, label='Original')
            plt.scatter(new_coords[:, 0], new_coords[:, 1], label='Rotated')
            # plt.contour(X_coord, Y_coord, Z_coord, levels=[1], colors=('r'), linewidths=2)
            # plt.scatter(xs, ys)
            plt.legend()
            plt.show(block=True)


        return R
    
    def determine_seqence(self, classes):
        directions = classes.C / torch.linalg.norm(classes.C, dim=-1, keepdim=True)
        cos_angles = torch.einsum('ni,mi->nm', directions, directions)

        idxs = torch.full((classes.C.shape[0],), -1).to(classes.C.device)
        inverse = torch.full_like(idxs, -1)
        idxs[0] = classes.C[:, 1].argmax()  # most posterior
        inverse[idxs[0]] = 0
        for i in range(1, idxs.shape[0]):
            dots = cos_angles[idxs[i - 1], inverse == -1]
            next_idx = torch.nonzero(inverse == -1)[dots.argmax(), 0]
            idxs[i] = next_idx
            inverse[next_idx] = i

        return idxs, inverse
    
    def determine_transition_probabilities(
        self,
        classes,
        idxs,
    ):
        with open('fdi_pair_distrs.json', 'r') as f:
            pair_normals = json.load(f)
        means = torch.tensor(pair_normals['means']).to(classes.F.device)
        covs = torch.tensor(pair_normals['covs']).to(classes.F.device)

        offset = 16 * self.is_lower[0]
        normals = MultivariateNormal(
            loc=means[offset:offset + 16, offset:offset + 16, :2],
            covariance_matrix=covs[offset:offset + 16, offset:offset + 16, :2, :2],
        )

        trans_log_probs = torch.zeros(idxs.shape[0] - 1, 16, 16).to(classes.F)
        for i, (idx1, idx2) in enumerate(zip(idxs[:-1], idxs[1:])):
            offsets = classes.C[idx2] - classes.C[idx1]
            trans_log_probs[i] = normals.log_prob(offsets[:2])

        return trans_log_probs
    
    def dynamic_programming(
        self,
        classes,
        idxs,
        trans_log_probs,
        tooth_factor: float=4.0,
    ):
        log_probs = torch.log(classes.F.softmax(dim=-1))

        q = torch.zeros_like(log_probs)
        q[0] = -tooth_factor * log_probs[idxs[0]]

        up = torch.arange(16).to(classes.F.device)
        p = torch.zeros_like(q).long()
        p[0] = up

        for i in range(1, classes.F.shape[0]):
            for j in range(16):
                prev_costs = q[i - 1]
                trans_costs = -trans_log_probs[i - 1, :, j]
                trans_costs[j] = 40

                costs = prev_costs + trans_costs
                m = costs.amin()
                q[i, j] = m - tooth_factor * log_probs[idxs[i], j]
                p[i, j] = costs.argmin()

        path = q[-1].argmin(keepdim=True)
        for i in range(p.shape[0] - 1):
            path = torch.cat((p[None, -1 - i, path[0]], path))

        return path, q[-1].min()
    
    def teeth_classes_to_labels(
        self,
        classes: PointTensor,
        method: Literal['argmax', 'mincost']='mincost',
    ):
        labels = torch.zeros(0, dtype=torch.int64).to(classes.F.device)
        for b in range(classes.batch_size):
            if method == 'argmax':
                numbers = torch.argmax(classes.batch(b).F, axis=-1)
            elif method == 'mincost':
                idxs, inverse = self.determine_seqence(classes.batch(b))
                trans_log_probs = self.determine_transition_probabilities(classes.batch(b), idxs)
                path, min_cost = self.dynamic_programming(classes.batch(b), idxs, trans_log_probs)
                numbers = path[inverse]

            fdis = 11 + 20 * self.is_lower[0] + 10 * (numbers // 8) + (numbers % 8)
            labels = torch.cat((labels, fdis))
            
        return classes.new_tensor(features=labels)

    def collate_downsample(
        self,
        point_counts: List[TensorType[torch.int64]],
        downsample_idxs: List[TensorType['m', torch.int64]],
        downsample_counts: List[TensorType[torch.int64]],
    ) -> TensorType['M', torch.int64]:
        point_counts = torch.stack(point_counts)
        downsample_idxs = torch.cat(downsample_idxs)
        downsample_counts = torch.stack(downsample_counts)
        batch_offsets = point_counts.cumsum(dim=-1) - point_counts
        batch_offsets = batch_offsets.repeat_interleave(downsample_counts)
        downsample_idxs += batch_offsets
        
        return downsample_idxs

    def collate_fn(
        self,
        batch: List[Dict[str, TensorType[..., Any]]],
    ) -> Tuple[
        TensorType['B', torch.bool],
        Union[
            Dict[str, Union[
                TensorType['V', 3, torch.float32],
                TensorType['T', 3, torch.int64],
                TensorType['B', torch.int64],
            ]],
            PointTensor,
        ],
        Union[
            PointTensor,
            Tuple[PointTensor, PointTensor],
        ],        
    ]:
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
        instance_counts = torch.stack(batch_dict['instance_count']) - 1
        instances = PointTensor(
            coordinates=torch.cat(instance_centroids),
            features=self.teeth_labels_to_classes(torch.cat(instance_labels)),
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

    def _transfer_fit_batch_to_device(
        self,
        x: Union[
            Dict[str, Union[
                TensorType['V', 3, torch.float32],
                TensorType['T', 3, torch.int64],
                TensorType['B', torch.int64],
            ]],
            PointTensor,
        ],
        y: Tuple[PointTensor, PointTensor],
        device: torch.device,
    ) -> Tuple[PointTensor, Tuple[PointTensor, PointTensor]]:
        instances, points = y

        return x.to(device), (instances.to(device), points.to(device))

    def _transfer_predict_batch_to_device(
        self,
        x: Union[
            Dict[str, Union[
                TensorType['V', 3, torch.float32],
                TensorType['T', 3, torch.int64],
                TensorType['B', torch.int64],
            ]],
            PointTensor,
        ],
        y: PointTensor,
        device: torch.device,
    ) -> Tuple[PointTensor, PointTensor]:
        return x.to(device), y.to(device)

    def transfer_batch_to_device(
        self,
        batch,
        device: torch.device,
        dataloader_idx: int,
    ) -> Union[
        Tuple[PointTensor, PointTensor],
        Tuple[PointTensor, Tuple[PointTensor, PointTensor]],
    ]:
        self.scan_file = batch[0]
        self.is_lower = batch[1].to(device)

        x, y = batch[2:]

        if isinstance(y, PointTensor):
            return self._transfer_predict_batch_to_device(x, y, device)
        else:
            return self._transfer_fit_batch_to_device(x, y, device)
