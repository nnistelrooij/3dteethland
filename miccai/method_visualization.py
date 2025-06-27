import json
from pathlib import Path

import numpy as np
import open3d
import open3d.visualization
import pymeshlab
import torch
from torch_scatter import scatter_mean
import yaml

from teethland import PointTensor
import teethland.data.transforms as T
from teethland.models import DentalNet, LandmarkNet, BinSegNet
from teethland.visualization import draw_landmarks

palette = np.array([
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
], dtype=np.uint8)


def load_open3d_mesh(root, case, arch):
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(str(root / case / f'{case}_{arch}.ply'))
    vertices = ms.current_mesh().vertex_matrix()
    normals = ms.current_mesh().vertex_normal_matrix()

    colors = np.full((vertices.shape[0], 3), 100 / 255)

    ud_sample = T.UniformDensityDownsample(voxel_size=0.025 * 17.3281)
    out = ud_sample(points=vertices)
    sample_idxs = out['ud_downsample_idxs']

    mesh = open3d.geometry.PointCloud(
        points=open3d.utility.Vector3dVector(vertices[sample_idxs]),
        # triangles=open3d.utility.Vector3iVector(triangles),
    )
    mesh.normals = open3d.utility.Vector3dVector(normals[sample_idxs])
    # mesh.compute_vertex_normals()
    mesh.colors = open3d.utility.Vector3dVector(colors[sample_idxs])

    return mesh


def load_point_tensors(root, case, arch):
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(str(root / case / f'{case}_{arch}.ply'))
    vertices = ms.current_mesh().vertex_matrix()
    normals = ms.current_mesh().vertex_normal_matrix()

    normalize = T.ZScoreNormalize(None, 17.3281)
    vertices = normalize(points=vertices)['points']

    with open(root / case / f'{case}_{arch}.json', 'r') as f:
        miccai_dict = json.load(f)
    labels = np.array(miccai_dict['labels'])
    _, instances = np.unique(labels, return_inverse=True)

    points = PointTensor(
        coordinates=torch.tensor(vertices, dtype=torch.float32),
        features=torch.tensor(np.column_stack((vertices, normals))).float(),
    ).to('cuda')
    instances = PointTensor(
        coordinates=torch.tensor(vertices, dtype=torch.float32),
        features=torch.from_numpy(instances - 1),
    ).to('cuda')
    classes = PointTensor(
        coordinates=torch.tensor(vertices, dtype=torch.float32),
        features=torch.from_numpy(labels),
    ).to('cuda')
    
    return points, instances, classes


def interpolate_foreground(mesh, pt):
    ud_sample = T.UniformDensityDownsample(voxel_size=0.025)
    out = ud_sample(points=pt.C.numpy())
    sample_idxs = torch.from_numpy(out['ud_downsample_idxs'])
    
    pt_down = pt.new_tensor(features=(pt.F > 0).float())[sample_idxs]
    pt_interp = pt_down.to('cuda').interpolate(pt.to('cuda'), k=10)
    foreground = pt_interp.F.cpu().numpy()[:, None]

    colors = torch.empty((pt.C.shape[0], 3))
    colors = np.array([[255, 0, 0]]) * foreground + np.array([[100, 100, 100]]) * (1 - foreground)
    mesh.vertex_colors = open3d.utility.Vector3dVector(colors / 255)

    return mesh



def show_instances(instances):
    colors = palette[instances.F.cpu().numpy()] / 255
    mesh = load_open3d_mesh(root, case, arch)
    mesh.colors = open3d.utility.Vector3dVector(colors)

    open3d.visualization.draw_geometries([mesh])



def show_classes(classes):

    classes.F[(11 <= classes.F) & (classes.F <= 17)] -= 10
    classes.F[classes.F == 18] = 7
    classes.F[(21 <= classes.F) & (classes.F <= 27)] -= 20
    classes.F[classes.F == 28] = 7
    classes.F[(71 <= classes.F) & (classes.F <= 75)] -= 63
    classes.F[(31 <= classes.F) & (classes.F <= 37)] -= 30
    classes.F[classes.F == 38] = 7
    classes.F[(81 <= classes.F) & (classes.F <= 85)] -= 73
    classes.F[(41 <= classes.F) & (classes.F <= 47)] -= 40
    classes.F[classes.F == 48] = 7

    colors = palette[-classes.F.cpu().numpy() - 1] / 255
    mesh = load_open3d_mesh(root, case, arch)
    mesh.colors = open3d.utility.Vector3dVector(colors)

    open3d.visualization.draw_geometries([mesh])





def seed_map(points, seeds):
    # seeds = seeds.interpolate(points)
    foreground = seeds.F[:, 0]
    foreground = foreground.float()
    foreground = foreground - foreground.amin()
    foreground /= foreground.amax()
    colors = torch.full((foreground.shape[0], 3), 100 / 255)
    colors[:, 0] = foreground.clip(100/255, 1)

    mesh = load_open3d_mesh(root, case, arch)
    mesh.colors = open3d.utility.Vector3dVector(colors.cpu().detach().numpy())

    open3d.visualization.draw_geometries([mesh])


def show_3d(points, instances, data, threshold=8):
    data = data.new_tensor(features=torch.where(
        instances.F[:, None] >= 0, data.F, 0.0,
    ))
    # data_interp = data.interpolate(points)

    data_norm = data.F.float()
    data_norm = data_norm - data_norm.amin(0)
    data_norm /= data_norm.amax(0)
    colors = torch.full((data_norm.shape[0], 3), 100 / 255).cuda()
    colors[torch.any(data.F > 0, -1)] = data_norm[torch.any(data.F > 0, -1)]

    mesh = load_open3d_mesh(root, case, arch)
    mesh.colors = open3d.utility.Vector3dVector(colors.cpu().detach().numpy())

    open3d.visualization.draw_geometries([mesh])


def show_tooth_crop(points, instances):

    gen_proposals = T.GenerateProposals(proposal_points=10_000, max_proposals=16)

    instance_centroids = scatter_mean(
        src=instances.C,
        index=instances.F + 1,
        dim=0,
    )
    
    data_dict = {
        'points': instances.C.cpu().numpy(),
        'instances': instances.F.cpu().numpy() + 1,
        'instance_centroids': instance_centroids.cpu().numpy(),
        'normals': points[points.cache['cp_downsample_idxs']].F[:, 3:].cpu().numpy(),
    }
    data_dict = gen_proposals(**data_dict)

    points = data_dict['points'][4]
    normals = data_dict['normals'][4]
    pcd = open3d.geometry.PointCloud(points=open3d.utility.Vector3dVector(points))
    pcd.normals = open3d.utility.Vector3dVector(normals)
    pcd.colors = open3d.utility.Vector3dVector(np.full((normals.shape[0], 3), 100 / 255))

    open3d.visualization.draw_geometries([pcd])

    diffs = points - data_dict['centroids'][4, None]
    pt = PointTensor(
        torch.from_numpy(points),
        torch.from_numpy(np.column_stack((points, normals, diffs))),
    ).to('cuda')


    data_norm = diffs
    data_norm = data_norm - data_norm.min(0)
    data_norm /= data_norm.max(0)

    pcd.colors = open3d.utility.Vector3dVector(data_norm)
    open3d.visualization.draw_geometries([pcd])

    net = BinSegNet.load_from_checkpoint(
        in_channels=9,
        **config['model']['binseg']
    ).to('cuda')
    seg = net(pt)
    
    foreground = torch.clip(seg.F[:, 0], -2, 2)
    foreground = foreground.float()
    foreground = foreground - foreground.amin()
    foreground /= foreground.amax()
    colors = torch.full((foreground.shape[0], 3), 100 / 255)
    colors[:, 0] = foreground.clip(100/255, 1)

    pcd.colors = open3d.utility.Vector3dVector(colors.cpu().detach().numpy())
    open3d.visualization.draw_geometries([pcd])


    foreground = -cusps.F[:, 0]
    foreground = foreground.float()
    foreground = foreground - foreground.amin()
    foreground /= foreground.amax()
    colors = foreground.clip(100/255, 1)
    colors = torch.column_stack((colors, colors, colors))

    pcd.colors = open3d.utility.Vector3dVector(colors.cpu().detach().numpy())
    open3d.visualization.draw_geometries([pcd])

    data_norm = cusps.F[:, 1:].float()
    data_norm = data_norm - data_norm.amin(0)
    data_norm /= data_norm.amax(0)

    pcd.colors = open3d.utility.Vector3dVector(data_norm.cpu().detach().numpy())
    open3d.visualization.draw_geometries([pcd])

    mask = cusps.F[:, 0] < 0.1
    landmarks = cusps.C[mask] + cusps.F[mask, 1:]
    draw_landmarks(cusps.C.cpu().numpy(), landmarks.cpu().detach().numpy(), normals=normals, point_size=0.025)

    kpt_mask = cusps.F[:, 0] < 0.12  # 2.5 mm
    coords = cusps.C + cusps.F[:, 1:]
    dists = torch.clip(cusps.F[:, 0], 0, 0.12)
    weights = (0.12 - dists) / 0.12
    landmarks = PointTensor(
        coordinates=coords[kpt_mask],
        features=weights[kpt_mask],
        batch_counts=torch.bincount(
            input=cusps.batch_indices[kpt_mask],
            minlength=cusps.batch_size,
        ),
    )
    landmarks = landmarks.cluster(**config['model']['dbscan_cfg'])
    draw_landmarks(cusps.C.cpu().numpy(), landmarks.C.cpu().detach().numpy(), normals=normals, point_size=0.025)

landmark_classes = {
    'Mesial': 0,
    'Distal': 1,
    'FacialPoint': 2,
    'OuterPoint': 3,
    'InnerPoint': 4,
    'Cusp': 5,
}
def final_result(classes):
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(str(root / case / f'{case}_{arch}.ply'))
    vertices = ms.current_mesh().vertex_matrix()
    normals = ms.current_mesh().vertex_normal_matrix()

    with open(root / case / f'{case}_{arch}.json', 'r') as f:
        miccai_dict = json.load(f)
    labels = np.array(miccai_dict['labels'])
    labels[(11 <= labels) & (labels <= 17)] -= 10
    labels[labels == 18] = 7
    labels[(21 <= labels) & (labels <= 27)] -= 20
    labels[labels == 28] = 7

    labels[(71 <= labels) & (labels <= 75)] -= 63
    labels[(31 <= labels) & (labels <= 37)] -= 30
    labels[labels == 38] = 7
    labels[(81 <= labels) & (labels <= 85)] -= 73
    labels[(41 <= labels) & (labels <= 47)] -= 40
    labels[labels == 48] = 7

    colors = palette[-labels - 1] / 255
    mesh = open3d.geometry.PointCloud(
        points=open3d.utility.Vector3dVector(vertices),
        # triangles=open3d.utility.Vector3iVector(triangles),
    )
    mesh.normals = open3d.utility.Vector3dVector(normals)
    # mesh.compute_vertex_normals()
    mesh.colors = open3d.utility.Vector3dVector(colors)

    with open('/home/mkaailab/Documents/IOS/3dteethland/data/3DTeethLand_landmarks_train/Batch_1_3_21_22/01J9K9S6_upper__kpt.json', 'r') as f:
        kpt_dict = json.load(f)

    landmarks = []
    for landmark in kpt_dict['objects']:
        label = landmark_classes[landmark['class']]
        coords = landmark['coord']

        landmarks.append(coords + [label])

    landmarks = np.array(landmarks)
    balls = []
    for landmark in landmarks:
        ball = open3d.geometry.TriangleMesh.create_sphere(radius=0.04 * 17.3281)
        ball.translate(landmark[:3])
        if landmark.shape[0] == 3:
            ball.paint_uniform_color([0.0, 0.0, 1.0])
        else:
            ball.paint_uniform_color(palette[int(landmark[-1])] / 255)
        ball.compute_vertex_normals()
        balls.append(ball)


    open3d.visualization.draw_geometries([mesh, *balls])









if __name__ == '__main__':
    root = Path('/home/mkaailab/Documents/IOS/3dteethland/data/lower_upper/')
    root = Path('/home/mkaailab/Documents/IOS/Brazil/cases')
    case, arch = '01J9K9S6', 'upper'
    case, arch = 'TVSR5QBQ', 'lower'
    
    with open('teethland/config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)        
    net = DentalNet.load_from_checkpoint(
        in_channels=6,
        num_classes=12,
        **config['model']['mixedseg'],
    ).to('cuda')

    points, instances, classes = load_point_tensors(root, case, arch)
    ud_sample = T.UniformDensityDownsample(voxel_size=0.01)
    out = ud_sample(points=points.C.cpu().numpy())
    sample_idxs = torch.from_numpy(out['ud_downsample_idxs']).to('cuda')
    points.cache['cp_downsample_idxs'] = sample_idxs
    instances = instances[sample_idxs]
    classes = classes[sample_idxs]
    offsets, sigmas, seeds, prototypes, _ = net(points, instances)

    mesh = open3d.io.read_triangle_mesh(str(root / case / f'{case}_{arch}.ply'))
    mesh.compute_vertex_normals()
    mesh.vertex_colors = open3d.utility.Vector3dVector(np.full((points.C.shape[0], 3), 100 / 255))
    open3d.visualization.draw_geometries([mesh])
    
    mesh = load_open3d_mesh(root, case, arch)
    open3d.visualization.draw_geometries([mesh])

    # final_result(classes)
    show_tooth_crop(points, instances)
    show_instances(instances)
    show_classes(classes)
    seed_map(points, seeds)
    show_3d(points, instances, offsets, threshold=8)
    show_3d(points, instances, sigmas, threshold=0.25)




