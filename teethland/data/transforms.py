import copy
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import networkx
from numpy.typing import ArrayLike, NDArray
from scipy.spatial.transform import Rotation
from scipy.special import softmax
from scipy.stats import multivariate_normal, truncnorm
from sklearn.decomposition import PCA
import torch
from torch_scatter import scatter_mean, scatter_min, scatter_max
from torchtyping import TensorType


class Compose:

    def __init__(
        self,
        *transforms: List[Callable[..., Dict[str, Any]]],
    ):
        self.transforms = transforms

    def __call__(
        self,
        **data_dict: Dict[str, Any],
    ) -> Dict[str, Any]:
        for t in self.transforms:
            data_dict = t(**data_dict)
        
        return data_dict

    def __repr__(self) -> str:
        return '\n'.join([
            self.__class__.__name__ + '(',
            *[
                '    ' + repr(t).replace('\n', '\n    ') + ','
                for t in self.transforms
            ],
            ')',
        ])


class ToTensor:

    def __init__(
        self,
        bool_dtypes: List[np.dtype]=[bool, np.bool_],
        int_dtypes: List[np.dtype]=[int, np.int16, np.uint16, np.int32, np.int64],
        float_dtypes: List[np.dtype]=[float, np.float32, np.float64],
    ) -> None:
        self.bool_dtypes = bool_dtypes
        self.int_dtypes = int_dtypes
        self.float_dtypes = float_dtypes

    def __call__(
        self,
        **data_dict: Dict[str, Any],
    ) -> Dict[str, TensorType[..., Any]]:
        for k, v in data_dict.items():
            dtype = v.dtype if isinstance(v, np.ndarray) else type(v)
            if dtype in self.bool_dtypes:
                data_dict[k] = torch.tensor(copy.copy(v), dtype=torch.bool)
            elif dtype in self.int_dtypes:
                data_dict[k] = torch.tensor(copy.copy(v), dtype=torch.int64)            
            elif dtype in self.float_dtypes:
                data_dict[k] = torch.tensor(copy.copy(v), dtype=torch.float32)
            elif dtype == str:
                data_dict[k] = v
            else:
                raise ValueError(
                    'Expected a scalar or list or NumPy array with elements of '
                    f'{self.bool_dtypes + self.int_dtypes + self.float_dtypes},'
                    f' but got {dtype}.'
                )
            
        return data_dict

    def __repr__(self) -> str:
        return self.__class__.__name__ + '()'


class RandomZAxisRotate:

    def __init__(
        self,
        max_degrees: float=45,
        rng: Optional[np.random.Generator]=None,
    ) -> None:
        self.max_angle = max_degrees / 180 * np.pi
        self.rng = np.random.default_rng() if rng is None else rng

    def __call__(
        self,
        points: NDArray[Any],
        **data_dict: Dict[str, Any],
    ) -> Dict[str, Any]:
        angle = self.rng.uniform(-self.max_angle, self.max_angle)
        cosval, sinval = np.cos(angle), np.sin(angle)

        R = np.array([
            [cosval,  -sinval, 0],
            [sinval, cosval, 0],
            [0,       0,      1],
        ])
        data_dict['points'] = points @ R.T

        if 'normals' in data_dict:
            data_dict['normals'] = data_dict['normals'] @ R.T

        if 'landmark_coords' in data_dict:
            data_dict['landmark_coords'] = data_dict['landmark_coords'] @ R.T
        if 'instance_centroids' in data_dict:
            data_dict['instance_centroids'] = data_dict['instance_centroids'] @ R.T
        
        return data_dict
    
    def __repr__(self) -> str:
        return '\n'.join([
            self.__class__.__name__ + '(',
            f'    max_angle: {self.max_angle * 180 / np.pi} degrees,',
            ')',
        ])


class RandomScale(object):

    def __init__(
        self,
        low: float=0.95,
        high: float=1.05,
        rng: Optional[np.random.Generator]=None,
    ) -> None:
        self.low = low
        self.high = high
        self.rng = np.random.default_rng() if rng is None else rng

    def __call__(
        self,
        points: NDArray[Any],
        **data_dict: Dict[str, Any],
    ) -> Dict[str, Any]:
        scale = self.rng.uniform(self.low, self.high)
        data_dict['points'] = points * scale

        if 'landmark_coords' in data_dict:
            data_dict['landmark_coords'] = data_dict['landmark_coords'] * scale
        if 'instance_centroids' in data_dict:
            data_dict['instance_centroids'] = data_dict['instance_centroids'] * scale
        
        return data_dict

    def __repr__(self) -> str:
        return '\n'.join([
            self.__class__.__name__ + '(',
            f'    low: {self.low},',
            f'    high: {self.high},',
            ')',
        ])


class RandomJitter(object):

    def __init__(
        self,
        sigma: float=0.005,
        clip: float=0.02,
        rng: Optional[np.random.Generator]=None,
    ) -> None:
        self.sigma = sigma
        self.clip = clip / sigma
        self.rng = np.random.default_rng() if rng is None else rng

    def __call__(
        self,        
        points: NDArray[Any],
        **data_dict: Dict[str, Any],
    ) -> Dict[str, Any]:
        jitter = truncnorm.rvs(
            -self.clip, self.clip,
            scale=self.sigma,
            size=points.shape,
            random_state=self.rng,
        )
        data_dict['points'] = points + jitter

        return data_dict
    
    def __repr__(self) -> str:
        return '\n'.join([
            self.__class__.__name__ + '(',
            f'    sigma: {self.sigma},',
            f'    clip: {self.clip},',
            ')',
        ])


class RandomXAxisFlip(object):

    def __init__(
        self,
        prob: float=0.5,
        rng: Optional[np.random.Generator]=None,
    ) -> None:
        self.prob = prob
        self.rng = np.random.default_rng() if rng is None else rng

        self.label_map = np.arange(86)
        self.label_map[11:19] = np.arange(21, 29)
        self.label_map[21:29] = np.arange(11, 19)
        self.label_map[31:39] = np.arange(41, 49)
        self.label_map[41:49] = np.arange(31, 39)
        self.label_map[51:56] = np.arange(61, 66)
        self.label_map[61:66] = np.arange(51, 56)
        self.label_map[71:76] = np.arange(81, 86)
        self.label_map[81:86] = np.arange(71, 76)

    def __call__(
        self,
        points: NDArray[Any],
        **data_dict: Dict[str, Any],
    ) -> Dict[str, Any]:
        if self.rng.random() >= self.prob:
            data_dict['flip'] = False
            data_dict['points'] = points

            return data_dict

        data_dict['flip'] = True
        points[:, 0] = -points[:, 0]
        data_dict['points'] = points

        if 'normals' in data_dict:
            data_dict['normals'][:, 0] = -data_dict['normals'][:, 0]

        if 'landmark_coords' in data_dict:
            landmarks = data_dict['landmark_coords'][:, 0]
            data_dict['landmark_coords'][:, 0] = -landmarks

        if 'instance_centroids' in data_dict:
            centroids = data_dict['instance_centroids'][:, 0]
            data_dict['instance_centroids'][:, 0] = -centroids

        if 'labels' in data_dict:
            data_dict['instance_labels'] = self.label_map[data_dict['instance_labels']]
            data_dict['labels'] = self.label_map[data_dict['labels']]

        return data_dict

    def __repr__(self) -> str:
        return '\n'.join([
            self.__class__.__name__ + '(',
            f'    prob: {self.prob},',
            ')',
        ])
    

class RandomShiftCentroids:

    def __init__(
        self,
        pos_sample: float=0.30,  # 0.6666 ** 3
        rng: Optional[np.random.Generator]=None,
    ) -> None:
        self.pos_sample = pos_sample
        self.rng = np.random.default_rng() if rng is None else rng

    def __call__(
        self,
        points: NDArray[Any],
        instances: NDArray[Any],
        instance_centroids: NDArray[Any],
        **data_dict: Dict[str, Any],
    ) -> Dict[str, Any]:
        out = []
        for i, mean in enumerate(instance_centroids):
            if (instances == i).sum() <= 5:
                out.append(mean)
                continue

            cov = np.cov(points[instances == i].T)
            samples = multivariate_normal.rvs(mean, cov, size=1000, random_state=self.rng)
            probs = multivariate_normal.pdf(samples, mean, cov)
            
            pos_mask = probs >= np.quantile(probs, 1 - self.pos_sample)
            out.append(samples[pos_mask][0])
        
        data_dict['points'] = points
        data_dict['instances'] = instances
        data_dict['instance_centroids'] = np.stack(out)

        return data_dict        

    def __repr__(self) -> str:
        return '\n'.join([
            self.__class__.__name__ + '(',
            f'    pos_sample: {self.pos_sample},',
            ')',
        ])


class PoseNormalize:

    def __call__(
        self,
        points: NDArray[Any],
        normals: NDArray[Any],
        **data_dict: Dict[str, Any],
    ) -> Dict[str, Any]:
        pca = PCA()
        pca.fit(points)
        R = pca.components_
        
        # disallow reflections
        if np.linalg.det(R) < 0:
            R = np.array([
                [-1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
            ]) @ R

        # rotate 180 degrees around y-axis if points are upside down
        if (normals @ R.T)[:, 2].mean() < 0:
            R = np.array([
                [-1, 0, 0],
                [0, 1, 0],
                [0, 0, -1],
            ]) @ R
        
        # rotate points to principal axes of decreasing explained variance
        data_dict['points'] = points @ R.T
        data_dict['normals'] = normals @ R.T
        
        if 'landmark_coords' in data_dict:
            data_dict['landmark_coords'] = data_dict['landmark_coords'] @ R.T

        T = np.eye(4)
        T[:3, :3] = R
        data_dict['affine'] = T @ data_dict.get('affine', np.eye(4))

        return data_dict

    def __repr__(self) -> str:
        return self.__class__.__name__ + '()'


class ZScoreNormalize:

    def __init__(
        self,
        mean: Optional[Union[float, ArrayLike]]=None,
        std: Optional[Union[float, ArrayLike]]=None,
    ) -> None:
        self.mean_ = mean
        self.std_ = std

    def mean(
        self,
        points: NDArray[Any],
    ) -> Union[float, ArrayLike]:
        if self.mean_ is None:
            return points.mean(axis=0)

        return self.mean_

    def std(
        self,
        points: NDArray[Any],
    ) -> Union[float, ArrayLike]:
        if self.std_ is None:
            return points.std(axis=0)

        return self.std_

    def affine(
        self,
        points: NDArray[Any],
    ) -> NDArray[Any]:
        trans = np.eye(4)
        trans[:3, 3] -= self.mean(points)

        scale = np.eye(4)
        scale[np.diag_indices(3)] /= self.std(points)

        return scale @ trans

    def __call__(
        self,
        points: NDArray[Any],
        **data_dict: Dict[str, Any],
    ) -> Dict[str, Any]:
        affine = self.affine(points)
        points_hom = np.column_stack((points, np.ones_like(points[:, 0])))
        data_dict['points'] = (points_hom @ affine.T)[:, :3]

        if 'landmark_coords' in data_dict:
            landmarks = data_dict['landmark_coords']
            landmarks_hom = np.column_stack((landmarks, np.ones_like(landmarks[:, 0])))
            data_dict['landmark_coords'] = (landmarks_hom @ affine.T)[:, :3]

        data_dict['affine'] = affine @ data_dict.get('affine', np.eye(4))

        return data_dict

    def __repr__(self) -> str:
        return '\n'.join([
            self.__class__.__name__ + '(',
            f'    mean={self.mean_},',
            f'    std={self.std_},',
            ')',
        ])


class XYZAsFeatures:

    def __call__(
        self,
        points: NDArray[Any],
        **data_dict: Dict[str, Any],
    ) -> Dict[str, Any]:
        data_dict['points'] = points

        if 'features' in data_dict:
            data_dict['features'] = np.concatenate(
                (data_dict['features'], points), axis=-1,
            )
        else:
            data_dict['features'] = points

        return data_dict

    def __repr__(self) -> str:
        return self.__class__.__name__ + '()'


class NormalAsFeatures:

    def __init__(
        self,
        eps: float=1e-8,
    ):
        self.eps = eps

    def __call__(
        self,
        normals: NDArray[Any],
        **data_dict: Dict[str, Any],
    ) -> Dict[str, Any]:
        normals = normals / (np.linalg.norm(normals, axis=-1, keepdims=True) + self.eps)

        if 'features' in data_dict:
            data_dict['features'] = np.concatenate(
                (data_dict['features'], normals), axis=-1,
            )
        else:
            data_dict['features'] = normals        
            
        data_dict['normals'] = normals

        return data_dict

    def __repr__(self) -> str:
        return self.__class__.__name__ + '()'


class ColorAsFeatures:

    def __call__(
        self,
        colors: NDArray[Any],
        **data_dict: Dict[str, Any],
    ) -> Dict[str, Any]:
        if 'features' in data_dict:
            data_dict['features'] = np.concatenate(
                (data_dict['features'], colors), axis=-1,
            )
        else:
            data_dict['features'] = colors        
            
        data_dict['colors'] = colors

        return data_dict

    def __repr__(self) -> str:
        return self.__class__.__name__ + '()'


class CentroidOffsetsAsFeatures:

    def __call__(
        self,
        points: NDArray[Any],
        centroids: NDArray[Any],
        **data_dict: Dict[str, Any],
    ) -> Dict[str, Any]:
        diffs = points - centroids[:, None]

        if 'features' in data_dict:
            data_dict['features'] = np.concatenate(
                (data_dict['features'], diffs), axis=-1,
            )
        else:
            data_dict['features'] = diffs        
            
        data_dict['points'] = points
        data_dict['centroids'] = centroids

        return data_dict

    def __repr__(self) -> str:
        return self.__class__.__name__ + '()'


class UniformDensityDownsample:

    def __init__(
        self,
        voxel_size: float,
        inplace: bool=False,
    ) -> None:
        self.voxel_size = voxel_size
        self.inplace = inplace
    
    def __call__(
        self,
        points: NDArray[Any],
        **data_dict: Dict[str, Any],
    ):
        pos_points = points - points.min(axis=0)
        discrete_coords = (pos_points / self.voxel_size).astype(int)

        voxel_centers = self.voxel_size * (discrete_coords + 0.5)
        sq_dists = np.sum((pos_points - voxel_centers) ** 2, axis=-1)

        factors = discrete_coords.max(axis=0) + 1
        factors = factors.cumprod() / factors
        vertex_voxel_idxs = np.sum(discrete_coords * factors, axis=-1)
        _, vertex_voxel_idxs = np.unique(
            vertex_voxel_idxs, return_inverse=True,
        )

        argmin = scatter_min(
            src=torch.from_numpy(sq_dists),
            index=torch.from_numpy(vertex_voxel_idxs),
        )[1].numpy()

        if 'ud_downsample_idxs' in data_dict:
            data_dict['ud_downsample_idxs_1'] = data_dict['ud_downsample_idxs'] 
            data_dict['ud_downsample_count_1'] = data_dict['ud_downsample_count']
            data_dict['ud_downsample_idxs_2'] = argmin
            data_dict['ud_downsample_count_2'] = argmin.shape[0]
        else:
            data_dict['ud_downsample_idxs'] = argmin
            data_dict['ud_downsample_count'] = argmin.shape[0]

        if not self.inplace:
            data_dict['points'] = points
            return data_dict

        data_dict['points'] = points[argmin]
        data_dict['point_count'] = argmin.shape[0]

        for key in ['features', 'labels', 'types', 'instances', 'normals', 'colors', 'attributes']:
            if key not in data_dict:
                continue

            data_dict[key] = data_dict[key][argmin]

        return data_dict

    def __repr__(self) -> str:
        return '\n'.join([
            self.__class__.__name__ + '(',
            f'    voxel_size={self.voxel_size},',
            f'    inplace={self.inplace},',
            ')',
        ])



class BoundaryAwareDownsample(UniformDensityDownsample):

    def __init__(
        self,
        voxel_size: float,
        sample_ratio: float,
        min_points: int=10_000,
        inplace: bool=False,
        rng: Optional[np.random.Generator]=None,
    ) -> None:
        super().__init__(voxel_size, inplace)

        self.sample_ratio = sample_ratio
        self.min_points = min_points
        self.rng = np.random.default_rng() if rng is None else rng

    def __call__(
        self,
        confidences: NDArray[Any],
        **data_dict: Dict[str, Any],
    ) -> Dict[str, Any]:
        # first get sample of points with uniform density
        data_dict = super().__call__(**data_dict)
        sample_idxs = data_dict['ud_downsample_idxs']
        sample_count = data_dict['ud_downsample_count']

        # determine boundary-aware sample of remaining points
        rand_idxs = self.rng.choice(
            a=sample_count,
            size=max(int(sample_count * self.sample_ratio), self.min_points),
            replace=False,
            p=softmax(-np.abs(confidences[sample_idxs] / 4)),
        )

        data_dict['ba_downsample_idxs'] = sample_idxs[rand_idxs]
        data_dict['ba_downsample_count'] = rand_idxs.shape[0]

        if not self.inplace:
            data_dict['confidences'] = confidences
            return data_dict

        data_dict['confidences'] = confidences[sample_idxs][rand_idxs]
        data_dict['points'] = data_dict['points'][rand_idxs]
        data_dict['point_count'] = rand_idxs.shape[0]

        for key in ['features', 'labels', 'types', 'instances', 'attributes']:
            if key not in data_dict:
                continue

            data_dict[key] = data_dict[key][rand_idxs]

        return data_dict

    def __repr__(self) -> str:
        return '\n'.join([
            self.__class__.__name__ + '(',
            f'    voxel_size={self.voxel_size},',
            f'    sample_ratio={self.sample_ratio},',
            f'    min_points={self.min_points},',
            f'    inplace={self.inplace},',
            ')',
        ])
    

class InstanceCentroids:

    def __call__(
        self,
        points: NDArray[Any],
        **data_dict: Dict[str, Any],
    ) -> Dict[str, Any]:
        # no-op if ground-truth data is unavailable
        if 'labels' not in data_dict:
            data_dict['points'] = points

            return data_dict

        labels, types, instances = data_dict['labels'], data_dict['types'], data_dict['instances']

        instance_centroids = scatter_mean(
            src=torch.from_numpy(points),
            index=torch.from_numpy(instances),
            dim=0,
        ).numpy()
        
        instance_labels = scatter_max(
            src=torch.from_numpy(labels),
            index=torch.from_numpy(instances),
            dim=0,
        )[0].numpy()
        instance_types = scatter_max(
            src=torch.from_numpy(types),
            index=torch.from_numpy(instances),
            dim=0,
        )[0].numpy()

        data_dict['points'] = points
        data_dict['instance_centroids'] = instance_centroids
        data_dict['instance_labels'] = instance_labels
        data_dict['instance_types'] = instance_types
        data_dict['instance_count'] = instance_labels.shape[0]

        return data_dict
    
    def __repr__(self) -> int:
        return self.__class__.__name__ + '()'


class MatchLandmarksAndTeeth:

    def __init__(
        self,
        move: float=0.04,
    ):
        from teethland.data.datasets import TeethLandDataset
        self.landmark_classes = TeethLandDataset.landmark_classes
        self.move = move

    def move_landmarks(
        self,
        landmark_coords: NDArray[Any],
        landmark_classes: NDArray[Any],
        points: NDArray[Any],
        labels: NDArray[Any],
        instances: NDArray[Any],
        **data_dict,
    ):
        moved_coords = landmark_coords.copy()

        mesial = landmark_classes == self.landmark_classes['Mesial']
        distal = landmark_classes == self.landmark_classes['Distal']

        # move mesial landmarks of central incisors
        mesial_coords = landmark_coords[mesial]
        dists = np.linalg.norm(mesial_coords[None] - mesial_coords[:, None], axis=-1)
        dists = np.where(dists > 0, dists, 1e6)
        if dists.min() < 4 * self.move:
            mesial_idxs = np.unravel_index(dists.argmin(), dists.shape)
            mesial_idxs = np.nonzero(mesial)[0][list(mesial_idxs)]
            mesial[mesial_idxs] = False
            if landmark_coords[mesial_idxs][:, 1].ptp() < 2 * self.move:
                if landmark_coords[mesial_idxs[0], 0] < landmark_coords[mesial_idxs[1], 0]:
                    moved_coords[mesial_idxs[0], 0] -= self.move
                    moved_coords[mesial_idxs[1], 0] += self.move
                else:
                    moved_coords[mesial_idxs[1], 0] -= self.move
                    moved_coords[mesial_idxs[0], 0] += self.move

        # move pairs of mesial and distal landmarks
        mesial_coords = moved_coords[mesial]
        distal_coords = moved_coords[distal]
        dists = np.linalg.norm(mesial_coords[:, None, :2] - distal_coords[None, :, :2], axis=-1)
        for mesial_idx, distal_idx in zip(*np.nonzero(dists < 5 * self.move)):
            diff = mesial_coords[mesial_idx, :2] - distal_coords[distal_idx, :2]
            diff /= np.linalg.norm(diff)

            mesial_idx = np.nonzero(mesial)[0][mesial_idx]
            distal_idx = np.nonzero(distal)[0][distal_idx]
            
            moved_coords[mesial_idx, :2] += self.move * diff
            moved_coords[distal_idx, :2] -= self.move * diff

        # move inner points of front elements away from center
        max_y = points[(labels > 0) & (labels % 10 <= 3), 1].max() - self.move
        for i, (coords, cls) in enumerate(zip(landmark_coords, landmark_classes)):
            if coords[1] > max_y or cls != self.landmark_classes['InnerPoint']:
                continue

            dists = np.linalg.norm(points - coords, axis=-1)
            min_dists = scatter_min(
                src=torch.from_numpy(dists),
                index=torch.from_numpy(instances),
                dim=0,
            )[0][1:].numpy()
            if (min_dists < self.move).sum() > 1:
                continue

            diff = coords[:2] / np.linalg.norm(coords[:2])
            moved_coords[i, :2] = coords[:2] + self.move * diff

        return moved_coords  

    def __call__(
        self,
        points: NDArray[Any],
        **data_dict: Dict[str, Any],
    ) -> Dict[str, Any]:
        # no-op if ground-truth data is unavailable
        if 'labels' not in data_dict:
            data_dict['points'] = points

            return data_dict
        
        # match each landmark to an instance after moving it inward
        moved_coords = self.move_landmarks(points=points, **data_dict)        
        tooth_points = np.where(data_dict['labels'][:, None] > 0, points, 1e6)
        dists = np.linalg.norm(
            tooth_points[None] - moved_coords[:, None],
        axis=-1)
        instance_idxs = data_dict['instances'][dists.argmin(1)]

        data_dict['points'] = points
        data_dict['landmark_instances'] = instance_idxs

        return data_dict
    
    def __repr__(self) -> int:
        return '\n'.join([
            self.__class__.__name__ + '(',
            f'    move={self.move},',
            ')',
        ])
    

class GenerateProposals:

    def __init__(
        self,
        proposal_points: int,
        max_proposals: int,
        rng: Optional[np.random.Generator]=None,
        label_as_instance: bool=False,
    ):
        self.proposal_points = proposal_points
        self.max_proposals = max_proposals
        self.rng = rng if rng is not None else np.random.default_rng()
        self.label_as_instance = label_as_instance

    def __call__(
        self,
        points: NDArray[Any],
        **data_dict: Dict[str, Any],
    ) -> Dict[str, Any]:
        # no-op if ground-truth data is unavailable
        if 'instances' not in data_dict:
            data_dict['points'] = points

            return data_dict
        
        _, counts = np.unique(data_dict['instances'], return_counts=True)
        if counts[1:].max() > self.proposal_points:
            print('Tooth points:', counts[1:].max(), ', Max points:', self.proposal_points)
        
        unique_instances = np.unique(data_dict['instances'])[1:]
        instance_idxs = np.sort(self.rng.choice(
            unique_instances,
            size=min(self.max_proposals, unique_instances.shape[0]),
            replace=False,
        ))

        centroids = data_dict['instance_centroids'][instance_idxs]
        if 'landmark_coords' in data_dict:
            coords = data_dict['landmark_coords']
            classes = data_dict['landmark_classes']
            instances = data_dict['landmark_instances']
            landmark_mask = np.any(instances[None] == instance_idxs[:, None], axis=0)
            landmark_idxs = np.nonzero(landmark_mask)[0][np.argsort(instances[landmark_mask])]

            instance_map = np.full((instance_idxs.max() + 1,), -1)
            instance_map[instance_idxs] = np.arange(instance_idxs.shape[0])
            instances = instance_map[instances[landmark_idxs]]

            landmarks = np.column_stack((
                coords[landmark_idxs],
                classes[landmark_idxs],
                instances,
            ))
            data_dict['landmarks'] = landmarks

        dists = np.linalg.norm(points[None] - centroids[:, None], axis=-1)
        point_idxs = np.argsort(dists, axis=1)[:, :self.proposal_points]
        labels = (data_dict['instances'][point_idxs] == instance_idxs[:, None]).astype(int)
        if self.label_as_instance:
            fg_mask = data_dict['labels'][point_idxs] == instance_idxs[:, None]
            labels = labels + fg_mask

        data_dict['points'] = points[point_idxs]
        data_dict['normals'] = data_dict['normals'][point_idxs]
        if 'colors' in data_dict: data_dict['colors'] = data_dict['colors'][point_idxs]
        if 'attributes' in data_dict: data_dict['attributes'] = data_dict['attributes'][point_idxs]
        data_dict['labels'] = labels
        data_dict['centroids'] = centroids
        data_dict['point_count'] = np.array([self.proposal_points]).repeat(centroids.shape[0])
        data_dict['point_idxs'] = point_idxs
        data_dict['instance_count'] = centroids.shape[0]

        return data_dict

    def __repr__(self) -> str:
        return '\n'.join([
            self.__class__.__name__ + '(',
            f'    proposal_points={self.proposal_points},',
            f'    max_proposals={self.max_proposals},',
            ')',
        ])
    

class AlignUpForward:

    def __init__(
        self,
        basis: NDArray[Any]=np.array([
            [-1, 0, 0],
            [0, -1, 0],
            [0, 0, 1],
        ]),
    ):
        self.basis = basis
    
    def __call__(
        self,
        **data_dict: Dict[str, Any],
    ) -> Dict[str, Any]:
        # no-op if ground-truth data is unavailable
        if 'instances' not in data_dict:
            return data_dict
        
        is_front = np.isin(data_dict['instance_labels'], [11, 21, 31, 41])
        is_left = np.isin(data_dict['instance_labels'], [16, 36])
        is_right = np.isin(data_dict['instance_labels'], [26, 46])

        front_c = data_dict['instance_centroids'][is_front].mean(0)
        left_c = data_dict['instance_centroids'][is_left][0]
        right_c = data_dict['instance_centroids'][is_right][0]

        dir_up = np.cross(left_c - front_c, right_c - front_c)
        dir_up /= np.linalg.norm(dir_up)

        dir_right = right_c - left_c
        dir_right /= np.linalg.norm(dir_right)

        lhs = front_c - left_c
        dotp = lhs @ dir_right
        back_c = left_c + dir_right * dotp
        dir_forward = front_c - back_c
        dir_forward /= np.linalg.norm(dir_forward)

        T = np.eye(4)
        T[:3, :3] = self.basis @ np.stack((dir_right, dir_forward, dir_up))

        data_dict['points'] = data_dict['points'] @ T[:3, :3].T
        data_dict['normals'] = data_dict['normals'] @ T[:3, :3].T
        data_dict['affine'] = T @ data_dict.get('affine', np.eye(4))
        data_dict['instance_centroids'] = data_dict['instance_centroids'] @ T[:3, :3].T

        data_dict['dir_right'] = self.basis[0]
        data_dict['dir_fwd'] = self.basis[1]
        data_dict['dir_up'] = self.basis[2]
        data_dict['trans'] = np.zeros(3)

        return data_dict

    def __repr__(self) -> str:
        return '\n'.join([
            self.__class__.__name__ + '(',
            f'    basis={self.basis},',
            ')',
        ])
    

class RandomRotate:

    def __init__(
        self,
        rng: Optional[np.random.Generator],
    ):
        self.rng = rng if rng is not None else np.random.default_rng()
        self.pose_normalize = PoseNormalize()

    def __call__(
        self,
        points: NDArray[Any],
        normals: NDArray[Any],
        instance_centroids: NDArray[Any],
        **data_dict: Dict[str, Any],
    ):
        degrees = self.rng.random(3) * 360 - 180
        r = Rotation.from_euler('xyz', angles=degrees, degrees=True)
        T = np.eye(4)
        T[:3, :3] = r.as_matrix()

        # apply PCA to get a rough position
        T = self.pose_normalize(
            points=points @ T[:3, :3].T,
            normals=normals @ T[:3, :3].T,
            affine=T,
        )['affine']

        data_dict['points'] = points @ T[:3, :3].T
        data_dict['normals'] = normals @ T[:3, :3].T
        data_dict['affine'] = T @ data_dict.get('affine', np.eye(4))
        data_dict['instance_centroids'] = instance_centroids @ T[:3, :3].T

        if 'dir_right' in data_dict:
            data_dict['dir_right'] = data_dict['dir_right'] @ T[:3, :3].T
            data_dict['dir_up'] = data_dict['dir_up'] @ T[:3, :3].T
            data_dict['dir_fwd'] = data_dict['dir_fwd'] @ T[:3, :3].T
            data_dict['trans'] = data_dict['trans'] @ T[:3, :3].T

        return data_dict

    def __repr__(self) -> str:
        return self.__class__.__name__ + '()'


class RandomPartial:

    def __init__(
        self,
        rng: Optional[np.random.Generator]=None,
        count_range: Tuple[int, int]=(2, 12),
        keep_radius: float=0.40,  # approximatly 14mm teeth
        p: float=0.9,
        skew: float=-0.9,
        min_points: int=0,
        do_translate: bool=True,
        do_planes: bool=True,
        do_single_component: bool=True,
    ):
        self.rng = rng if rng is not None else np.random.default_rng()
        self.range = count_range
        self.keep_radius = keep_radius
        self.p = p
        self.min_points = min_points
        self.do_translate = do_translate
        self.do_planes = do_planes
        self.do_single_component = do_single_component

        num_counts = count_range[1] - count_range[0] + 1
        middle_idx = (num_counts - 1) // 2
        probs = np.full((num_counts,), 1 / num_counts)
        for i in range(num_counts):
            bias = np.trunc(i - (num_counts - 1) / 2) / middle_idx
            probs[i] += bias * probs[middle_idx] * skew

        self.probs = probs

    def determine_inside(
        self,
        points,
        centroids,
    ):
        normal = centroids[-1] - centroids[0]
        normal /= np.linalg.norm(normal)
        point1 = centroids[0] - self.keep_radius * normal
        point2 = centroids[-1] + self.keep_radius * normal
        plane1 = np.concatenate((normal, [-normal @ point1]))
        plane2 = np.concatenate((-normal, [normal @ point2]))

        middle_point = (centroids[0] + centroids[-1]) / 2
        normal = np.cross(normal, middle_point + [0, 0, 1] - centroids[0])
        normal /= np.linalg.norm(normal)
        point1 = centroids[((centroids - middle_point) @ normal).argmin()] - self.keep_radius * normal
        point2 = centroids[((centroids - middle_point) @ normal).argmax()] + self.keep_radius * normal
        plane3 = np.concatenate((normal, [-normal @ point1]))
        plane4 = np.concatenate((-normal, [normal @ point2]))

        # determine points inside planes
        coords_homo = np.column_stack((points, np.ones(points.shape[0])))
        is_inside = (
            ((coords_homo @ plane1) >= 0)
            & ((coords_homo @ plane2) >= 0)
            & ((coords_homo @ plane3) >= 0)
            & ((coords_homo @ plane4) >= 0)
        )

        return is_inside

    def single_connected_component(
        self,
        points,
        triangles,
        mask,
        tooth_coord,
    ):    
        # get vertex mask and triangles
        inside_triangles = triangles[np.all(mask[triangles], axis=-1)]
        
        vertex_mask = np.zeros_like(mask)
        vertex_mask[inside_triangles.flatten()] = True
        
        vertex_map = np.full((points.shape[0],), -1)
        vertex_map[vertex_mask] = np.arange(vertex_mask.sum())
        inside_triangles = vertex_map[inside_triangles]

        # determine component idxs
        edges = np.concatenate((
            inside_triangles[:, [0, 1]],
            inside_triangles[:, [0, 2]],
            inside_triangles[:, [1, 2]],
        ))        
        G = networkx.Graph(list(edges))
        dists = np.linalg.norm(points[vertex_mask] - tooth_coord, axis=-1)
        comp_idxs = np.array(list(networkx.node_connected_component(G, dists.argmin())))

        # get final mask
        final_mask = np.zeros_like(mask)
        final_mask[np.nonzero(vertex_mask)[0][comp_idxs]] = True

        return final_mask

    def __call__(
        self,
        points: NDArray[Any],
        triangles: NDArray[Any],
        **data_dict: Dict[str, Any],
    ):
        if self.rng.random() > self.p:
            data_dict['points'] = points
            data_dict['triangles'] = triangles
            return data_dict
        
        # determine a sequence based on FDI labels
        fdis = data_dict['instance_labels']
        q1 = (fdis > 10) & (fdis < 20)
        q2 = (fdis > 20) & (fdis < 30)
        q3 = (fdis > 30) & (fdis < 40)
        q4 = (fdis > 40) & (fdis < 50)        
        sort_idxs = np.concatenate((
            np.nonzero(q1)[0][np.argsort(fdis[q1])[::-1]],
            np.nonzero(q2)[0][np.argsort(fdis[q2])],
            np.nonzero(q3)[0][np.argsort(fdis[q3])[::-1]],
            np.nonzero(q4)[0][np.argsort(fdis[q4])],
        ))
        while True:
            # sample number of teeth
            count_range = np.arange(self.range[0], min(self.range[1], sort_idxs.shape[0]) + 1)
            probs = self.probs[:count_range.shape[0]] / sum(self.probs[:count_range.shape[0]])
            num_teeth = self.rng.choice(count_range, p=probs)

            # sample consecutive teeth
            start_tooth = self.rng.integers(sort_idxs.shape[0] - num_teeth, endpoint=True)
            tooth_idxs = sort_idxs[start_tooth:start_tooth + num_teeth]

            centroids = data_dict['instance_centroids'][tooth_idxs]
            if self.do_planes:
                # determine points inside of four planes
                vertex_mask = self.determine_inside(points, centroids)

                # determine largest area with connected triangles
                if self.do_single_component:
                    vertex_mask = self.single_connected_component(
                        points, triangles, vertex_mask, centroids[0],
                    )
            else:
                # keep the points close to centroids of selected teeth
                dists = np.linalg.norm(points[None] - centroids[:, None], axis=-1).min(0)
                vertex_mask = dists < self.keep_radius

            if vertex_mask.sum() >= self.min_points:
                break

        # update triangles
        if self.do_single_component:
            vertex_map = np.full((points.shape[0],), -1)
            vertex_map[vertex_mask] = torch.arange(vertex_mask.sum())
            triangles = vertex_map[triangles]
            triangles = triangles[np.all(triangles >= 0, axis=-1)]

        data_dict['points'] = points[vertex_mask]
        data_dict['normals'] = data_dict['normals'][vertex_mask]
        data_dict['colors'] = data_dict['colors'][vertex_mask]
        data_dict['labels'] = data_dict['labels'][vertex_mask]
        data_dict['types'] = data_dict['types'][vertex_mask]
        data_dict['attributes'] = data_dict['attributes'][vertex_mask]
        data_dict['instances'] = data_dict['instances'][vertex_mask]
        data_dict['point_count'] = vertex_mask.sum()
        data_dict['triangles'] = triangles
        data_dict['triangle_count'] = triangles.shape[0]

        if not self.do_translate:
            return data_dict

        trans = -points[vertex_mask].mean(0)
        data_dict['points'] = data_dict['points'] + trans
        data_dict['instance_centroids'] = data_dict['instance_centroids'] + trans
        T = np.eye(4)
        T[:3, 3] = trans
        data_dict['affine'] = T @ data_dict.get('affine', np.eye(4))
        data_dict['trans'] = data_dict.get('trans', np.zeros(3)) + trans

        return data_dict

    def __repr__(self) -> str:
        return '\n'.join([
            self.__class__.__name__ + '(',
            f'    range={self.range},',
            f'    keep_radius={self.keep_radius},',
            f'    p={self.p},',
            f'    min_points={self.min_points},',
            f'    do_translate={self.do_translate},',
            f'    do_planes={self.do_planes},',
            f'    do_single_component={self.do_single_component},',
            f'    probs={self.probs},',
            ')',
        ])
