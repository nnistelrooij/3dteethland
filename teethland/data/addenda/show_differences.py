import json
from pathlib import Path

import numpy as np
import open3d
import open3d.visualization
import pymeshlab

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


if __name__ == '__main__':
    root = Path('/mnt/diag/IOS/3dteethseg/challenge_dataset')

    ms = pymeshlab.MeshSet()

    for i, addenda_file in enumerate(root.glob('addenda/*.json')):
        mesh_file = sorted(root.glob(f'**/{addenda_file.stem}.obj'))[-1]
        ms.load_new_mesh(str(mesh_file))
        vertices = ms.current_mesh().vertex_matrix()
        triangles = ms.current_mesh().face_matrix()

        print(i, addenda_file.stem)
        with open(addenda_file, 'r') as f:
            addenda_dict = json.load(f)

        add_labels = np.array(addenda_dict['labels'])
        add_instances = np.array(addenda_dict['instances']) - 1

        centroids = []
        for idx in np.unique(add_instances)[1:]:
            centroid = vertices[add_instances == idx].mean(0)
            centroids.append(centroid)
        centroids = np.stack(centroids)
        inst_labels = add_labels[np.unique(add_instances, return_index=True)[1][1:]]
        print('addenda:', inst_labels[np.argsort(centroids[:, 0])])

        original_file = sorted(root.glob(f'**/{addenda_file.name}'))[-1]
        with open(original_file, 'r') as f:
            original_dict = json.load(f)

        ori_labels = np.array(original_dict['labels'])
        ori_instances = np.array(original_dict['instances']) - 1
        
        centroids = []
        for idx in np.unique(ori_instances)[1:]:
            centroid = vertices[ori_instances == idx].mean(0)
            centroids.append(centroid)
        centroids = np.stack(centroids)
        inst_labels = ori_labels[np.unique(ori_instances, return_index=True)[1][1:]]
        print('original:', inst_labels[np.argsort(centroids[:, 0])])

        add_mesh = open3d.geometry.TriangleMesh(
            open3d.utility.Vector3dVector(vertices),
            open3d.utility.Vector3iVector(triangles),
        )
        add_mesh.compute_vertex_normals()
        add_mesh.vertex_colors = open3d.utility.Vector3dVector(palette[add_instances] / 255)


        ori_mesh = open3d.geometry.TriangleMesh(
            open3d.utility.Vector3dVector(vertices),
            open3d.utility.Vector3iVector(triangles),
        )
        ori_mesh.compute_vertex_normals()
        ori_mesh.translate((80, 0, 0))
        ori_mesh.vertex_colors = open3d.utility.Vector3dVector(palette[ori_instances] / 255)

        open3d.visualization.draw_geometries([add_mesh, ori_mesh], width=1600, height=900)
