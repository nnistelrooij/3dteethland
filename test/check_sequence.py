
import json
from pathlib import Path

import numpy as np
import pymeshlab
from tqdm import tqdm

def determine_seqence(centroids):
    directions = centroids / np.linalg.norm(centroids, axis=-1, keepdims=True)
    cos_angles = np.einsum('ni,mi->nm', directions, directions)

    idxs = np.full((centroids.shape[0],), -1)
    inverse = np.full_like(idxs, -1)
    idxs[0] = centroids[:, 1].argmax()
    inverse[idxs[0]] = 0
    for i in range(1, idxs.shape[0]):
        dots = cos_angles[idxs[i - 1], inverse == -1]
        next_idx = np.nonzero(inverse == -1)[0][dots.argmax()]
        idxs[i] = next_idx
        inverse[next_idx] = i

    return idxs, inverse


if __name__ == '__main__':
    mesh_root = Path('/home/mkaailab/Documents/IOS/partials/full_dataset/align_full')
    labels_root = mesh_root.parent / 'complete_full'

    ms = pymeshlab.MeshSet()

    t = tqdm(sorted(mesh_root.glob('**/*.ply')))
    for mesh_file in t:
        t.set_description(mesh_file.name)
        ms.load_new_mesh(str(mesh_file))

        vertices = ms.current_mesh().vertex_matrix()

        with open(next(labels_root.glob('**/' + mesh_file.with_suffix('.json').name)), 'r') as f:
            teeth3ds_dict = json.load(f)

        instances = np.array(teeth3ds_dict['instances'])
        labels = np.array(teeth3ds_dict['labels'])

        centroids = []
        fdis = []
        for label in np.unique(teeth3ds_dict['instances'])[1:]:
            centroid = vertices[instances == label].mean(0)
            centroids.append(centroid)
            fdis.append(labels[instances == label][0])
        centroids = np.array(centroids)
        fdis = np.array(fdis)

        idxs, inverse = determine_seqence(centroids)
        sequence_fdis = fdis[idxs]

        sequence_fdis[(10 < sequence_fdis) & (sequence_fdis < 20)] -= 11
        sequence_fdis[(20 < sequence_fdis) & (sequence_fdis < 30)] -= 21
        sequence_fdis[(30 < sequence_fdis) & (sequence_fdis < 40)] -= 31
        sequence_fdis[(40 < sequence_fdis) & (sequence_fdis < 50)] -= 41

        middle_idx = 0
        for i in range(sequence_fdis.shape[0] - 1):
            if sequence_fdis[i] - sequence_fdis[i + 1] <= 0:
                break
            middle_idx += 1

        middle2_idx = sequence_fdis.shape[0] - 1
        for i in range(sequence_fdis.shape[0] - 1):
            if sequence_fdis[-1 - i] - sequence_fdis[-2 - i] <= 0:
                break
            middle2_idx -= 1

        if middle2_idx > middle_idx + 1:
            print(sequence_fdis, mesh_file)

        
