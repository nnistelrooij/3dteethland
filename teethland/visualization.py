import math
from pathlib import Path
from typing import List, Optional

import numpy as np
import open3d
import pymeshlab
import torch


palette = torch.tensor([
    [174, 199, 232],
    [152, 223, 138],
    [31, 119, 180],
    [255, 187, 120],
    [188, 189, 34],
    [140, 86, 75],
    [255, 152, 150],
    [214, 39, 40],
    [197, 176, 213],
    [148, 103, 189],
    [196, 156, 148], 
    [23, 190, 207], 
    [247, 182, 210], 
    [219, 219, 141], 
    [255, 127, 14], 
    [158, 218, 229], 
    [44, 160, 44], 
    [112, 128, 144], 
    [227, 119, 194], 
    [82, 84, 163],
    [100, 100, 100],
], dtype=torch.uint8)


def draw_mesh(
    file: Path,
    pt,
) -> None:
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(str(file))
    mesh = ms.current_mesh()
    mesh.compact()

    vertices = mesh.vertex_matrix()
    triangles = mesh.face_matrix()

    mesh = open3d.geometry.TriangleMesh(
        vertices=open3d.utility.Vector3dVector(vertices),
        triangles=open3d.utility.Vector3iVector(triangles),
    )
    mesh.compute_vertex_normals()

    if pt.has_features:
        pt = pt.new_tensor(features=pt.F - pt.F.amin() - 1)
        mesh.vertex_colors = open3d.utility.Vector3dVector(palette[pt.F] / 255)

    open3d.visualization.draw_geometries([mesh], width=1600, height=900)


def draw_point_clouds(
    pt,
    output_type: Optional[str]=None,
) -> Optional[List[open3d.geometry.PointCloud]]:
    pt = pt.to('cpu')
    length = math.sqrt(pt.batch_size)
    nrows = math.ceil(length)
    ncols = round(length)

    x_range = pt.C[:, 0].amax() - pt.C[:, 0].amin()
    y_range = pt.C[:, 1].amax() - pt.C[:, 1].amin()
    geometries = []
    for row in range(nrows):
        for col in range(ncols):
            batch_idx = ncols * row + col
            if batch_idx >= pt.batch_size:
                continue

            points = pt.batch(batch_idx).C
            points[:, 0] += 1.2 * row * x_range
            points[:, 1] += 1.2 * col * y_range
            pcd = open3d.geometry.PointCloud(
                points=open3d.utility.Vector3dVector(points.cpu()),
            )

            if not pt.has_features or pt.F.dim() != 1:
                geometries.append(pcd)
                continue

            feats = pt.batch(batch_idx).F
            if feats.dtype in [torch.float32, torch.float64]:
                colors = feats.float()
                colors = colors - colors.amin()
                colors /= colors.amax()
                colors = colors.expand(3, -1).T.cpu()
            elif feats.dtype in [torch.bool, torch.int32, torch.int64]:
                feats = feats.long()
                classes = feats - feats.amin() - 1
                classes[classes >= 0] %= palette.shape[0] - 1
                colors = palette[classes] / 255

            pcd.colors = open3d.utility.Vector3dVector(colors.cpu().detach())
            geometries.append(pcd)

    if output_type == 'tensorboard':
        return to_dict_batch(geometries[:1])

    open3d.visualization.draw_geometries(geometries, width=1600, height=900)


def draw_landmarks(
    vertices: np.ndarray,
    landmarks: np.ndarray,
    labels:  Optional[np.ndarray]=None,
    normals: Optional[np.ndarray]=None,
    triangles: Optional[np.ndarray]=None,
    point_size: float=0.025
):
    if triangles is not None:
        geom = open3d.geometry.TriangleMesh(
            vertices=open3d.utility.Vector3dVector(vertices),
            triangles=open3d.utility.Vector3iVector(triangles),
        )
        geom.compute_vertex_normals()
    else:
        geom = open3d.geometry.PointCloud(
            open3d.utility.Vector3dVector(vertices),
        )
        if normals is not None:
            geom.normals = open3d.utility.Vector3dVector(normals)
        geom.colors = open3d.utility.Vector3dVector(np.full((vertices.shape[0], 3), 100 / 255))

    balls = []
    for landmark in landmarks:
        ball = open3d.geometry.TriangleMesh.create_sphere(radius=point_size)
        ball.translate(landmark[:3])
        if landmark.shape[0] == 3:
            ball.paint_uniform_color([0.0, 0.0, 1.0])
        else:
            ball.paint_uniform_color(palette[int(landmark[-1])].numpy() / 255)
        ball.compute_vertex_normals()
        balls.append(ball)

    open3d.visualization.draw_geometries([geom, *balls], width=1600, height=900)


def check_predictions(
    # mesh_root: Path=Path('/home/mkaailab/Documents/IOS/Maud Wijbrandts/results_test'),
    # mesh_root: Path=Path('/home/mkaailab/Documents/IOS/Zainab Bousshimad/data/root_first50_review_fdis'),
    mesh_root: Path=Path('/mnt/diag/CBCT/fusion/IOS arches'),
    extensions: List[str]=['obj', 'ply', 'stl'],
    filter: List[str]=['16_lower', '16_upper'],
    # ann_root: Path=Path('/home/mkaailab/Documents/IOS/Maud Wijbrandts/LU_C_FDI&Toothtype_root'),
    # ann_root: Path=Path('/home/mkaailab/Documents/IOS/Katja Vos/reviewed_annotations'),
    # ann_root: Path=Path('/home/mkaailab/Documents/IOS/Zainab Bousshimad/data/root_first50_review_fdis'),
    ann_root: Path=Path('/mnt/diag/CBCT/fusion/IOS arches'),
    verbose: bool=True,
    save_to_storage: bool=False,
):
    mesh_files = sorted([f for ext in extensions for f in mesh_root.glob(f'**/*.{ext}')])
    ann_files = sorted(ann_root.glob('**/*er.json'))
    for ann_file in ann_files:
        ann_mesh_files = [f for f in mesh_files if f.stem == ann_file.stem]
        if not ann_mesh_files:
            continue
        mesh_file = ann_mesh_files[0]

        if filter and mesh_file.stem not in filter:
            continue

        # load mesh
        print(mesh_file)
        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(str(mesh_file))
        mesh = ms.current_mesh()
        mesh.compact()
        vertices = mesh.vertex_matrix()
        triangles = mesh.face_matrix()

        # load annotation
        with open(ann_file, 'rb') as f:
            ann = json.load(f)
        # labels = np.array([0 if not attrs else attrs[0] for attrs in ann['attributes']])
        # labels = np.array([0 if not point['assigned_ids'] else point['assigned_ids'][0] for point in ann])
        
        labels = np.array(ann['labels'])
        instances = np.array(ann['instances'])
        unique, index = np.unique(instances, return_index=True)

        # print sorted tooth labels
        centroids = np.zeros((0, 3))
        for idx in unique[1:]:
            centroid = vertices[instances == idx].mean(0)
            centroids = np.concatenate((centroids, [centroid]))
        if centroids.shape[0]:
            inst_labels = labels[index[1:]]
            print(inst_labels[np.argsort(centroids[:, 0])])

        # initialize Open3D mesh
        mesh = open3d.geometry.TriangleMesh(
            vertices=open3d.utility.Vector3dVector(vertices),
            triangles=open3d.utility.Vector3iVector(triangles),
        )
        mesh.compute_vertex_normals()

        # add instance colors
        fdis = np.where(labels > 0, (labels - 11) % 20, -1)
        mesh.vertex_colors = open3d.utility.Vector3dVector(palette[fdis] / 255)
        if verbose:
            open3d.visualization.draw_geometries([mesh], width=1600, height=900)
        if save_to_storage:
            open3d.io.write_triangle_mesh(str(mesh_file.parent / f'fdis_{mesh_file.stem}.ply'), mesh)

        # add type colors
        if 'extra' in ann:
            extra = np.array(ann['extra'])[:, 0]
            types = np.where(instances >= 0, extra[instances], -1)
            mesh.vertex_colors = open3d.utility.Vector3dVector(palette[types] / 255)
            if verbose:
                open3d.visualization.draw_geometries([mesh], width=1600, height=900)
            if save_to_storage:
                open3d.io.write_triangle_mesh(str(mesh_file.parent / f'types_{mesh_file.stem}.ply'), mesh)

        # add fracture segmentation
        if 'probs' in ann and len(ann['probs'][0]) == 1:
            probs = np.array(ann['probs'])[:, 0]
            if 'extra' in ann:  # only show fractures on opbouw teeth
                probs[(types == -1) | (types > 0)] = 0

            red = np.column_stack((np.ones_like(probs), np.zeros_like(probs), np.zeros_like(probs)))
            gray = np.tile(np.full_like(probs, 100 / 255)[:, None], (1, 3))
            colors = probs[:, None] * red + (1 - probs[:, None]) * gray
            mesh.vertex_colors = open3d.utility.Vector3dVector(colors)
            if verbose:
                open3d.visualization.draw_geometries([mesh], width=1600, height=900)
            if save_to_storage:
                open3d.io.write_triangle_mesh(str(f'{mesh_file.parent.as_posix()}/fractures_{mesh_file.stem}.ply'), mesh)

        # combine tooth wear segmentations
        probs_palette = np.array([
            [122, 143, 182],
            [69, 22, 0],
            [166, 188, 153],
            [157, 23, 42],
            [88, 45, 131],
            [219, 167, 164],
            [27, 179, 103],
            [171, 20, 168],
            [78, 149, 165],
        ])
        if 'probs' in ann and len(ann['probs'][0]) > 1:
            probs = np.array(ann['probs'])[:, 2:]
            probs[probs < 0.5] = 0.0
            color = (probs[:, :, None] * probs_palette).sum(1) / 255
            gray = np.tile(np.full_like(probs[:, :1], 100 / 255), (1, 3))

            colors = color + gray * (1 - probs.sum(1, keepdims=True))
            mesh.vertex_colors = open3d.utility.Vector3dVector(colors)
            if verbose:
                open3d.visualization.draw_geometries([mesh], width=1600, height=900)
            if save_to_storage:
                open3d.io.write_triangle_mesh(str(f'{mesh_file.parent.as_posix()}/wear_{mesh_file.stem}.ply'), mesh)

        if 'attributes' in ann:
            attrs = np.array([max(attrs) if attrs else 0 for attrs in ann['attributes']])
            onehots = np.zeros((attrs.size, 10))
            onehots[np.arange(attrs.size), attrs] = 1

            color = (onehots[:, 1:, None] * probs_palette).sum(1) / 255
            gray = np.tile(np.full_like(onehots[:, :1], 100 / 255), (1, 3))

            colors = color + gray * onehots[:, :1]
            mesh.vertex_colors = open3d.utility.Vector3dVector(colors)
            if verbose:
                open3d.visualization.draw_geometries([mesh], width=1600, height=900)
            if save_to_storage:
                open3d.io.write_triangle_mesh(str(f'{mesh_file.parent.as_posix()}/attrs_{mesh_file.stem}.ply'), mesh)


def check_landmarks():
    seg_root = Path('/mnt/diag/IOS/3dteethseg/full_dataset/lower_upper')
    landmarks_root = Path('/home/mkaailab/Documents/IOS/3dteethland/code/preds/3dteethland')

    for landmarks_file in sorted(landmarks_root.glob('**/*__kpt.json')):
        stem = landmarks_file.stem.split('__')[0]
        ann_file = next(seg_root.glob(f'**/{stem}.json'))
        mesh_file = next(seg_root.glob(f'**/{stem}.obj'))

        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(str(mesh_file))

        vertices = ms.current_mesh().vertex_matrix()
        triangles = ms.current_mesh().face_matrix()

        with open(ann_file, 'rb') as f:
            ann = json.load(f)

        labels = np.array(ann['labels'])
        _, inverse = np.unique(labels, return_inverse=True)
        print(stem)
        print(_)

        labels = np.clip(labels % 10, a_min=0, a_max=7)

        mesh = open3d.geometry.TriangleMesh(
            vertices=open3d.utility.Vector3dVector(vertices),
            triangles=open3d.utility.Vector3iVector(triangles),
        )
        mesh.compute_vertex_normals()
        mesh.vertex_colors = open3d.utility.Vector3dVector(palette[inverse - 1] / 255)
        mesh.vertex_colors = open3d.utility.Vector3dVector(np.full((inverse.shape[0], 3), 100 / 255))


        with open(landmarks_file, 'r') as f:
            landmarks = json.load(f)['objects']
        landmark_coords = np.array([landmark['coord'] for landmark in landmarks])
        landmark_scores = np.array([landmark['score'] for landmark in landmarks])
        landmark_classes = np.array([landmark['class'] for landmark in landmarks])
        _, landmark_classes = np.unique(landmark_classes, return_inverse=True)

        mask = landmark_scores >= 0.3
        landmarks = np.column_stack((landmark_coords[mask], landmark_classes[mask]))
        
        balls = []
        for landmark in landmarks:
            ball = open3d.geometry.TriangleMesh.create_sphere(radius=0.03 * 17.3281)
            ball.translate(landmark[:3])
            ball.paint_uniform_color(palette[int(landmark[-1])].numpy() / 255)
            ball.compute_vertex_normals()
            balls.append(ball)

        open3d.visualization.draw_geometries([*balls, mesh])





if __name__ == '__main__':
    import json
    from pathlib import Path

    import numpy as np
    import open3d
    import pymeshlab

    check_predictions()
    # check_landmarks()

    