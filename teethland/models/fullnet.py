import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import pymeshlab
import pytorch_lightning as pl
import torch
from torch_scatter import scatter_mean
from torchtyping import TensorType

import teethland
from teethland import PointTensor
import teethland.data.transforms as T
from teethland.cluster import learned_region_cluster
import teethland.nn as nn


def tta_nms(
    clusters: PointTensor,
    labels: PointTensor,
    iou_thresh: float=0.8,
):
    clusters1, clusters2 = clusters.batch(0), clusters.batch(1)

    num_instances = labels.batch_counts.sum()
    ious = torch.zeros(num_instances, num_instances).to(clusters.F.device)
    for i in range(labels.batch_counts[0]):
        mask1 = clusters1.F == i
        for j in range(labels.batch_counts[1]):
            mask2 = clusters2.F == (labels.batch_counts[0] + j)

            inter = (mask1 & mask2).sum()
            union = (mask1 | mask2).sum()
            iou = inter / (union + 1e-6)
            ious[i, labels.batch_counts[0] + j] = iou

    keep = torch.ones(labels.batch_counts.sum()).bool().to(clusters.F.device)
    for index, iou in enumerate(ious):
        if not keep[index]:
            continue

        condition = iou >= iou_thresh
        keep = keep & ~condition

    return torch.nonzero(keep)[:, 0]


def instance_nms(
    probs: PointTensor,
    point_idxs: TensorType['K', 'N', torch.int64],
    conf_thresh: float=0.5,
    iou_thresh: float=0.3,
    score_thresh: float=0.5,
):
    fg_point_idxs, scores = [], torch.empty(0).to(probs.F)
    for i in range(probs.batch_size):
        fg_mask = probs.batch(i).F >= conf_thresh
        fg_idxs = point_idxs[i][fg_mask]
        fg_point_idxs.append(set(fg_idxs.cpu().tolist()))

        score = probs.batch(i).F[fg_mask].mean()
        scores = torch.cat((scores, score[None]))

    sort_index = torch.argsort(scores, descending=True)
    fg_point_idxs = [fg_point_idxs[idx.item()] for idx in sort_index]
    
    ious = torch.zeros(probs.batch_size, probs.batch_size).to(probs.F)
    for i in range(probs.batch_size):
        for j in range(i + 1, probs.batch_size):
            inter = len(fg_point_idxs[i] & fg_point_idxs[j])
            union = len(fg_point_idxs[i] | fg_point_idxs[j])
            iou = inter / (union + 1e-6)
            ious[i, j] = iou

    keep = scores[sort_index] >= score_thresh
    for index, iou in enumerate(ious):
        if not keep[index]:
            continue

        condition = iou >= iou_thresh
        keep = keep & ~condition

    return sort_index[keep].unique()


class FullNet(pl.LightningModule):

    def __init__(
        self,
        align: Dict[str, Any],
        instseg: Dict[str, Any],
        single_tooth: Dict[str, Any],
        proposal_points: int,
        is_panoptic: bool,
        with_attributes: bool,
        dbscan_cfg: dict[str, Any],
        do_align: bool,
        tta: bool,
        stage2_iters: int,
        post_process_seg: bool,
        post_process_labels: bool,
        out_dir: Path,
        **kwargs,
    ) -> None:
        super().__init__()

        # align stage
        if do_align:
            ckpt = align.pop('checkpoint_path')
            self.align_backbone = nn.StratifiedTransformer(
                in_channels=min(6, kwargs['in_channels']),
                out_channels=None,
                **instseg,
            )
            self.load_ckpt(self.align_backbone, ckpt)

            self.align_head = nn.MLP(self.align_backbone.enc_channels, 256, 9)
            self.load_ckpt(self.align_head, ckpt)
            
        # instance segmentation stage
        ckpt = instseg.pop('checkpoint_path')
        self.instances_model = nn.StratifiedTransformer(
            in_channels=min(6, kwargs['in_channels']),
            out_channels=[6, 1, None] + ([None] if is_panoptic else []),
            **instseg,
        )
        self.load_ckpt(self.instances_model, ckpt)

        self.fdi_model = nn.MaskedAveragePooling(
            num_features=self.instances_model.out_channels[-1 - is_panoptic],
            out_channels=kwargs['num_classes'],
        )
        try:
            self.load_ckpt('fdi_model', ckpt)
        except:
            self.load_ckpt(self.fdi_model, ckpt)
        if is_panoptic:
            self.type_model = nn.MaskedAveragePooling(
                num_features=self.instances_model.out_channels[-1],
                out_channels=4,
            )
            self.load_ckpt('type_model', ckpt)
        
        # landmark prediction stage
        ckpt = single_tooth.pop('checkpoint_path')
        self.single_tooth_model = nn.StratifiedTransformer(
            in_channels=3 + kwargs['in_channels'],
            # out_channels=[1] + ([1] if is_panoptic or with_attributes else []) + ([None] if with_attributes else []),
            out_channels=11,
            **single_tooth,
        )
        self.load_ckpt(self.single_tooth_model, ckpt)

        if with_attributes:
            self.attribute_model = nn.MaskedAveragePooling(
                num_features=self.single_tooth_model.out_channels[-1],
                out_channels=4,
            )
            self.load_ckpt(self.attribute_model, ckpt)

        self.gen_proposals = T.GenerateProposals(proposal_points, max_proposals=100)
        self.do_align = do_align
        self.is_panoptic = is_panoptic
        self.with_attributes = with_attributes
        self.tta = tta
        self.stage2_iters = stage2_iters
        self.post_process_seg = post_process_seg
        self.post_process_labels = post_process_labels
        self.dbscan_cfg = dbscan_cfg
        self.only_dentalnet = stage2_iters == 0
        self.out_dir = out_dir
        from teethland.data.datasets import TeethLandDataset
        self.landmark_class2label = {
            v: k for k, v in TeethLandDataset.landmark_classes.items()
        }

    def load_ckpt(self, model, ckpt: str):
        if not ckpt:
            return
        
        ckpt = torch.load(ckpt)['state_dict']
        
        if isinstance(model, str):
            state_dict = self.__getattr__(model).state_dict()
            ckpt = {k.split('.', 1)[1]: v for k, v in ckpt.items() if model in k}
            self.__getattr__(model).load_state_dict(ckpt)
            self.__getattr__(model).requires_grad_(False)
        else:
            state_dict = model.state_dict()            
            ckpt = {k.split('.', 1)[1]: v for k, v in ckpt.items()}
            ckpt = {k: v for k, v in ckpt.items() if k in state_dict}
            model.load_state_dict(ckpt)
            model.requires_grad_(False)

    def align_stage(
        self,
        x: PointTensor,
    ) -> Tuple[PointTensor, TensorType[4, 4, torch.float32]]:
        # downsample
        x_down = x[x.cache['instseg_downsample_idxs']]
        x_down = x_down.new_tensor(features=x_down.F[:, :6])
        
        encoding = self.align_backbone(x_down)
        embeddings = scatter_mean(encoding.F, encoding.batch_indices, dim=0)
        embeddings = PointTensor(
            coordinates=torch.zeros(x.batch_size, 3).to(x.C),
            features=embeddings,
        )
        preds = self.align_head(embeddings)

        dir_up = preds.F[:, :3] / torch.linalg.norm(preds.F[:, :3], dim=-1, keepdim=True)
        dir_fwd = preds.F[:, 3:6] / torch.linalg.norm(preds.F[:, 3:6], dim=-1, keepdim=True)
        trans = preds.F[:, 6:]

        # make two vectors orthogonal
        dots = torch.einsum('bi,bi->b', dir_up, dir_fwd)
        dir_up -= dots[:, None] * dir_fwd

        # determine non-reflecting rotation matrix to standard basis
        pred_right = torch.cross(dir_fwd, dir_up, dim=-1)
        R = torch.stack((pred_right, dir_fwd, dir_up))[:, 0]
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
        T[:3, 3] = -trans @ R.T

        # apply transformation to input
        coords_homo = torch.column_stack((x.C, torch.ones_like(x.C[:, 0])))
        out = x.new_tensor(
            coordinates=(coords_homo @ T.T)[:, :3],
            features=torch.column_stack((
                (coords_homo @ T.T)[:, :3],
                x.F[:, 3:6] @ T[:3, :3].T,
                x.F[:, 6:],
            )),
        )
        out.cache = x.cache

        return out, T

    def instances_stage(
        self,
        x: PointTensor,
    ) -> Tuple[
        PointTensor, 
        Union[PointTensor, Tuple[PointTensor, PointTensor]],
        PointTensor,
    ]:
        # downsample
        x_down = x[x.cache['instseg_downsample_idxs']]
        x_down = x_down.new_tensor(features=x_down.F[:, :6])

        # test-time augmentation by horizontally flipping
        if self.tta:
            x_down_flip = x_down.clone()
            x_down_flip._coordinates[:, 0] *= -1
            x_down_flip.F[:, 0] *= -1
            x_down_flip.F[:, 3] *= -1

            x_down = teethland.stack((x_down, x_down_flip))
        
        # forward pass
        if self.is_panoptic:
            _, (spatial_embeds, seeds, features, features2) = self.instances_model(x_down)
        else:
            _, (spatial_embeds, seeds, features) = self.instances_model(x_down)

        # cluster
        offsets = spatial_embeds.new_tensor(features=spatial_embeds.F[:, :3])
        sigmas = spatial_embeds.new_tensor(features=spatial_embeds.F[:, 3:])
        clusters = learned_region_cluster(
            offsets, sigmas, seeds,
        )
        clusters._coordinates[clusters.batch_counts[0]:, 0] *= -1

        _, classes = self.fdi_model(features, clusters)
        labels = self.trainer.datamodule.teeth_classes_to_labels(
            classes, method='mincost' if self.post_process_labels else 'argmax',
        )

        if self.tta:
            keep_idxs = tta_nms(clusters, labels)
            clusters.F[~torch.any(clusters.F == keep_idxs[:, None], dim=0)] = -1
            labels = labels[keep_idxs]

            if self.only_dentalnet:
                clusters = clusters.batch(0).new_tensor(
                    features=torch.maximum(clusters.batch(0).F, clusters.batch(1).F),
                )
                labels._batch_counts = labels.batch_counts.sum()[None]

            clusters.F = torch.unique(clusters.F, return_inverse=True)[1] - 1

        if self.is_panoptic:
            features = features, features2

        # interpolate clusters back to original scan
        instances = clusters.interpolate(teethland.stack([x for _ in labels.batch_counts]))

        return instances, features, labels
    
    def landmarks_process(
        self,
        proposals,
        point_offsets,
        keep_idxs,
        labels,
        affine,
    ):        
        print('start landmarks')
        # apply NMS selection
        proposals = proposals.batch(keep_idxs)
        points_offsets = preds[1:]
        points_offsets = [offsets.batch(keep_idxs) for offsets in points_offsets]

        # process point-level landmarks
        landmarks_list = []
        for i, offsets in enumerate(points_offsets):
            kpt_mask = offsets.F[:, 0] < 0.15  # 2.5 mm
            coords = proposals.C + offsets.F[:, 1:]
            dists = torch.clip(offsets.F[:, 0], 0, 0.15)
            weights = (0.15 - dists) / 0.15
            landmarks = PointTensor(
                coordinates=coords[kpt_mask],
                features=weights[kpt_mask],
                batch_counts=torch.bincount(
                    input=proposals.batch_indices[kpt_mask],
                    minlength=proposals.batch_size,
                ),
            )
            landmarks = landmarks.cluster(**self.dbscan_cfg)
            landmarks = landmarks.new_tensor(features=torch.column_stack((landmarks.F, 
                torch.full((landmarks.C.shape[0],), i).to(coords.device),
            )))
            landmarks_list.append(landmarks)
        landmarks = teethland.cat(landmarks_list)

        landmarks = self.trainer.datamodule.process_landmarks(labels, affine, landmarks)

        return landmarks
    
    def single_tooth_stage(
        self,
        x: PointTensor,
        instances: PointTensor,
        features: Union[PointTensor, Tuple[PointTensor, PointTensor]],
        labels: PointTensor,
        affine: TensorType[4, 4, torch.float32],
    ) -> Tuple[PointTensor, PointTensor, Optional[PointTensor]]:
        x_down = x[x.cache['landmarks_downsample_idxs']]
        instances = teethland.stack(([
            instances.batch(i)[x.cache['landmarks_downsample_idxs']]
            for i in range(instances.batch_size)
        ]))
        
        # generate proposals based on predicted instances
        for _ in range(self.stage2_iters):
            data_dict = {
                'points': x_down.C.cpu().numpy(),
                'instances': instances.F.cpu().numpy(),
                'instance_centroids': labels.C.cpu().numpy(),
                'normals': x_down.F[:, 3:6].cpu().numpy(),
                'colors': x_down.F[:, 6:].cpu().numpy(),
            }
            data_dict = self.gen_proposals(**data_dict)
            points = torch.from_numpy(data_dict['points']).to(x.C)
            normals = torch.from_numpy(data_dict['normals']).to(x.C)
            colors = torch.from_numpy(data_dict['colors']).to(x.C)
            centroids = torch.from_numpy(data_dict['instance_centroids']).to(x.C)
            proposals = PointTensor(
                coordinates=points.reshape(-1, 3),
                features=torch.column_stack((
                    points.reshape(-1, 3),
                    normals.reshape(-1, 3),
                    colors.reshape(-1, 3) if colors.numel() else points.reshape(-1, 3)[:, :0],
                    (points - centroids[:, None]).reshape(-1, 3),
                )),
                batch_counts=torch.tensor([points.shape[1]]*points.shape[0]).to(x.C.device),
            )

            # run the proposals through the models
            _, preds = self.single_tooth_model(proposals)
            seg = preds[0] if isinstance(preds, list) else preds

            if seg.F.shape[1] == 1:
                probs = seg.new_tensor(features=torch.sigmoid(seg.F[:, 0]))
            else:
                probs = seg.new_tensor(features=1 - torch.softmax(seg.F, dim=-1)[:, 0])

            # apply non-maximum suppression to remove redundant instances
            point_idxs = torch.from_numpy(data_dict['point_idxs']).to(seg.F.device)
            keep_idxs = instance_nms(probs, point_idxs)
            instances.F[~torch.any(instances.F == keep_idxs[:, None], dim=0)] = -1
            instances.F = torch.unique(instances.F, return_inverse=True)[1] - 1
            labels = labels[keep_idxs]
            probs = probs.batch(keep_idxs)

            # update the centroids for the second run
            labels._coordinates = scatter_mean(
                src=probs.C[probs.F >= 0.5],
                index=probs.batch_indices[probs.F >= 0.5],
                dim=0,
            )

        # interpolate segmentations to original points
        instances = torch.full_like(x.F[:, 0], -1).long()
        max_probs = torch.zeros_like(instances).float()
        for b in range(probs.batch_size):
            interp = probs.batch(b).interpolate(x, dist_thresh=0.03).F
            instances = torch.where(interp > max_probs, b, instances)
            max_probs = torch.maximum(max_probs, interp)
        instances = x.new_tensor(features=instances)
        if self.post_process_seg:
            instances = self.trainer.datamodule.process_instances(instances, max_probs)
        else:
            instances.F = torch.where(max_probs >= 0.5, instances.F, -1)
        
        _, inverse, counts = torch.unique(instances.F, return_inverse=True, return_counts=True)
        instances.F[(counts < 16)[inverse]] = -1
        instances.F = torch.unique(instances.F, return_inverse=True)[1] - 1
        
        if self.is_panoptic:
            features, features2 = features

        clusters = instances[x.cache['instseg_downsample_idxs']]
        _, classes = self.fdi_model(features.batch(0), clusters)
        fdis = self.trainer.datamodule.teeth_classes_to_labels(
            classes, method='mincost' if self.post_process_labels else 'argmax',
        )

        if seg.F.shape[1] > 1:  # multi-class semantic segmentation
            seg = seg.batch(keep_idxs)

            max_probs = torch.zeros((instances.F.shape[0], seg.F.shape[1])).to(x.C)
            for b in range(seg.batch_size):
                probs = torch.softmax(seg.batch(b).F, axis=1)
                interp = seg.batch(b).new_tensor(features=probs).interpolate(x, dist_thresh=0.03).F
                max_probs = torch.maximum(max_probs, interp)

            instances = instances.new_tensor(features=torch.column_stack(
                (instances.F, max_probs),
            ))

        if not isinstance(preds, list):
            return instances, fdis, None

        if self.is_panoptic:
            _, classes = self.type_model(features2.batch(0), clusters)
            labels = fdis.new_tensor(features=torch.column_stack(
                (fdis.F, classes.F.argmax(-1)),
            ))
        
        if len(preds) in [2, 3]:  # fracture or caries segmentation
            seg_probs = preds[1].new_tensor(features=torch.sigmoid(preds[1].F[:, 0]))
            seg_probs = seg_probs.batch(keep_idxs)
            
            max_probs = torch.zeros_like(instances.F).float()
            for b in range(seg_probs.batch_size):
                interp = seg_probs.batch(b).interpolate(x, dist_thresh=0.03).F
                max_probs = torch.maximum(max_probs, interp)

            instances = instances.new_tensor(features=torch.column_stack(
                (instances.F, max_probs),
            ))

        if len(preds) > 3:  # landmark detection        
            landmarks = self.landmarks_process(proposals, preds[1:], keep_idxs, labels, affine)

            return instances, labels, landmarks

        if len(preds) == 3:  # caries type classification
            caries_classes = torch.full_like(max_probs, -1, dtype=torch.int64)
            caries_clusters = torch.full_like(max_probs, -1, dtype=torch.int64)
            for b in range(preds[2].batch_size):
                caries = preds[1].batch(b)
                fg_probs = torch.sigmoid(caries.F[:, 0])
                fg = proposals.batch(b)[fg_probs >= 0.1]
                if not fg:
                    continue

                cluster_idxs = torch.full_like(fg_probs, -1, dtype=torch.int64)
                cluster_idxs[fg_probs >= 0.1] = fg.cluster(
                    min_points=10, return_index=True, ignore_noisy=True,
                )
                if torch.all(cluster_idxs == -1):
                    continue

                clusters = proposals.batch(b).new_tensor(features=cluster_idxs)
                _, inst_classes = self.attribute_model(preds[2].batch(b), clusters)
                inst_classes = inst_classes.new_tensor(features=inst_classes.F.argmax(-1))
                
                point_classes = torch.where(cluster_idxs >= 0, inst_classes.F[cluster_idxs], -1)
                point_classes = caries.new_tensor(features=point_classes)
                caries_classes = torch.maximum(
                    caries_classes, point_classes.interpolate(x, dist_thresh=0.03).F,
                )
                
                cluster_idxs[cluster_idxs >= 0] += caries_clusters.max() + 1
                cluster_idxs = caries.new_tensor(features=cluster_idxs)
                caries_clusters = torch.maximum(
                    caries_clusters, cluster_idxs.interpolate(x, dist_thresh=0.03).F,
                )
            
            instances = instances.new_tensor(features=torch.column_stack(
                (instances.F, caries_clusters, caries_classes),
            ))
            
        return instances, fdis, None

    def forward(
        self,
        x: PointTensor,
    ) -> Tuple[TensorType['N', torch.float32], PointTensor, PointTensor, PointTensor]:
        # stage 1
        if self.do_align:
            x, affine = self.align_stage(x)
        else:
            affine = torch.eye(4).to(x.C)

        # stage 2
        instances, features, labels = self.instances_stage(x)
        if torch.all(instances.F == -1) or self.only_dentalnet:
            return torch.zeros_like(x.C), instances, labels, None
        
        # stage 3
        instances, labels, landmarks = self.single_tooth_stage(x, instances, features, labels, affine)

        return instances.batch(0), labels.batch(0), landmarks
    
    def predict_step(
        self,
        batch: Tuple[
            TensorType['B', 'C', 'D', 'H', 'W', torch.float32],
            Path,
            TensorType[4, 4, torch.float32],
            TensorType[3, torch.int64],
        ],
        batch_idx: int,
    ):
        try:
            instances, classes, landmarks = self(batch)

            self.save_segmentation(instances, classes)
            self.save_landmarks(landmarks)
        except Exception as e:
            return

    def save_segmentation(
        self,
        points: PointTensor,
        labels: PointTensor,
    ):
        # make output directory
        path = Path(self.trainer.datamodule.scan_file)
        if self.out_dir.name:
            out_file = self.out_dir / path.with_suffix('.json')
        else:
            out_file = self.trainer.datamodule.root / path.with_suffix('.json')
        out_file.parent.mkdir(parents=True, exist_ok=True)

        # save transformed mesh
        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(str(self.trainer.datamodule.root / path))
        ms.save_current_mesh(str(out_file.parent / path.name))

        # determine per-point labels and instances
        instances = points.F if points.F.ndim == 1 else points.F[:, 0].long()
        out_dict = {
            'instances': (torch.unique(instances, return_inverse=True)[1] - 1).cpu().tolist(),
            'labels': torch.zeros_like(instances).cpu().tolist(),
        }

        if points.F.ndim == 2:
            extra = points.F[:, 1:]
            out_dict['probs'] = extra.cpu().tolist()

        if torch.any(instances >= 0):
            if labels.F.ndim == 1:
                labels = torch.where(instances >= 0, labels.F[instances], 0)
            else:
                out_dict['extra'] = labels.F[:, 1:].cpu().tolist()
                labels = torch.where(instances >= 0, labels.F[instances, 0], 0)
            
            out_dict['labels'] = labels.long().cpu().tolist()

        # save as JSON file
        with open(out_file, 'w') as f:
            json.dump(out_dict, f)

    def save_landmarks(self, landmarks: Optional[PointTensor]):
        if landmarks is None:
            return
        
        template = {
            'version': '1.1',
            'description': 'landmarks',
            'key': self.trainer.datamodule.scan_file,
            'objects': [],
        }
        for i, (coords, (score, cls), instance) in enumerate(zip(
            landmarks.C, landmarks.F, landmarks.batch_indices,
        )):
            landmark = {
                'key': f'uuid_{i}',
                'score': score.cpu().item(),
                'class': self.landmark_class2label[cls.cpu().item()],
                'coord': coords.cpu().tolist(),
                'instance_id': instance.cpu().item()
            }
            template['objects'].append(landmark)

        out_name = Path(template['key'][:-4] + '__kpt.json')
        if self.out_dir.name:
            out_file = self.out_dir / out_name.name
        else:
            out_file = self.trainer.datamodule.root / out_name
        with open(out_file, 'w') as f:
            json.dump(template, f, indent=2)
