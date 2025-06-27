from collections import defaultdict
import json
import multiprocessing as mp
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pymeshlab
from tqdm import tqdm


def process_scan(
    files: Tuple[Path, Path],
):
    mesh_file, label_file = files

    with open(label_file, 'r') as f:
        teeth3ds_dict = json.load(f)
    instances = np.array(teeth3ds_dict['instances'])
    labels = np.array(teeth3ds_dict['labels'])
        
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(str(mesh_file))
    vertices = ms.current_mesh().vertex_matrix()

    fdis = np.zeros(0, dtype=int)
    centroids = np.zeros((0, 3))
    for idx in np.unique(instances)[1:]:
        inst_mask = instances == idx

        fdi = labels[inst_mask][0] - 11
        fdi -= 2 if (fdi % 20) >= 10 else 0
        fdi -= 4 if fdi >= 20 else 0

        centroid = vertices[inst_mask].mean(0)

        fdis = np.concatenate((fdis, [fdi]))
        centroids = np.concatenate((centroids, [centroid]))

    # determine tooth pair offsets
    pair_offsets = defaultdict(list)
    for i in range(fdis.shape[0]):
        for j in range(i + 1, fdis.shape[0]):
            offsets = centroids[j] - centroids[i]

            # track offsets in both directions
            fdi1, fdi2 = fdis[i], fdis[j]
            pair_offsets[fdi1, fdi2].append(offsets)
            pair_offsets[fdi2, fdi1].append(-offsets)

            # track offsets in both reflections
            fdi1 = fdi1 - 8 if (fdi1 % 16) >= 8 else fdi1 + 8
            fdi2 = fdi2 - 8 if (fdi2 % 16) >= 8 else fdi2 + 8
            offsets = offsets.copy()
            offsets[0] *= -1
            pair_offsets[fdi1, fdi2].append(offsets)
            pair_offsets[fdi2, fdi1].append(-offsets)

    return pair_offsets


def remove_outliers(
    pair_offsets,
    axes: List[int]=[0, 1],
    poly_degree: int=3,
    max_diffs: List[int]=[18, 18],  # upper, lower
):
    reg_functions = []
    for is_arch_lower in [False, True]:
        gap2dists = defaultdict(lambda: np.zeros(0))
        for fdi1 in range(16):
            for fdi2 in range(fdi1, 16):
                offsets = pair_offsets[16 * is_arch_lower + fdi1, 16 * is_arch_lower + fdi2]
                dists = np.linalg.norm(offsets[:, axes], axis=-1)
                lr1 = 7 - fdi1 if fdi1 < 8 else fdi1
                lr2 = 7 - fdi2 if fdi2 < 8 else fdi2
                gap = np.abs(lr2 - lr1)
                gap2dists[gap] = np.concatenate((gap2dists[gap], dists))

        xs = np.arange(16).repeat([len(gap2dists[gap]) for gap in range(16)])
        ys = [dist for gap in range(16) for dist in gap2dists[gap]]
        fit = np.polyfit(xs, ys, poly_degree)
        reg_functions.append(np.poly1d(fit))

    clean_pair_offsets = defaultdict(list)
    for (fdi1, fdi2), offsets in pair_offsets.items():
        if fdi1 // 16 != fdi2 // 16:
            continue

        mod_fdi1, mod_fdi2 = fdi1 % 16, fdi2 % 16
        lr1 = 7 - mod_fdi1 if mod_fdi1 < 8 else mod_fdi1
        lr2 = 7 - mod_fdi2 if mod_fdi2 < 8 else mod_fdi2
        gap = np.abs(lr2 - lr1)
        expected_dist = reg_functions[fdi1 // 16](gap)

        dists = np.linalg.norm(offsets[:, axes], axis=-1)
        keep_offsets = [
            direction
            for direction, dist in zip(offsets, dists)
            if np.abs(expected_dist - dist) <= max_diffs[fdi1 >= 16]
        ]
        keep_offsets = np.stack(keep_offsets) if keep_offsets else np.zeros((0, 3))
        clean_pair_offsets[fdi1, fdi2] = keep_offsets

    return clean_pair_offsets


def determine_fdi_pair_distributions(
    mesh_files: List[Path],
    label_files: List[Path],
    verbose: bool=False,
    num_processes: int=32,
):
    # get all tooth pair offsets from training data
    pair_offsets = defaultdict(lambda: np.zeros((0, 3)))
    with mp.Pool(num_processes) as p:
        i = p.imap_unordered(process_scan, zip(mesh_files, label_files))
        for offsets in tqdm(i, total=len(mesh_files)):
            for k, v in offsets.items():
                pair_offsets[k] = np.concatenate((pair_offsets[k], v))

    # remove offsets between incorrect annotations
    pair_offsets = remove_outliers(pair_offsets)

    # model offsets per FDI pair as multi-variate Gaussian
    pair_means = np.zeros((32, 32, 3))
    pair_covs = np.tile(np.eye(3)[None, None], (32, 32, 1, 1))
    for fdi1 in range(32):
        for fdi2 in range(32):
            if fdi1 == fdi2 or fdi1 // 16 != fdi2 // 16:
                continue

            offsets = pair_offsets[fdi1, fdi2]
            if len(offsets) == 0:
                fdi1_ = fdi1 + (16 if fdi1 < 16 else -16)
                fdi2_ = fdi2 + (16 if fdi2 < 16 else -16)
                offsets = pair_offsets[fdi1_, fdi2_]
                print(fdi1, fdi2)
            pair_means[fdi1, fdi2] = np.mean(offsets, axis=0)
            pair_covs[fdi1, fdi2] = np.cov(offsets.T)

    if not verbose:
        return pair_means, pair_covs

    for i, name in zip(range(3), 'xyz'):
        fig, axs = plt.subplots(2, 2)
        fig.suptitle(name)
        axs[0, 0].imshow(pair_means[:16, :16, i])
        axs[0, 0].set_title('Upper jaw means')
        axs[0, 0].axis('off')
        axs[0, 1].imshow(pair_covs[:16, :16, i, i])
        axs[0, 1].set_title('Upper jaw SDs')
        axs[0, 1].axis('off')
        axs[1, 0].imshow(pair_means[16:, 16:, i])
        axs[1, 0].set_title('Lower jaw means')
        axs[1, 0].axis('off')
        axs[1, 1].imshow(pair_covs[16:, 16:, i, i])
        axs[1, 1].set_title('Lower jaw SDs')
        axs[1, 1].axis('off')
        plt.show(block=True)

    return pair_means, pair_covs


if __name__ == '__main__':
    # get all files from storage
    root = Path('/home/mkaailab/Documents/IOS/partials/full_dataset')
    root = Path('/mnt/diag/IOS/3dteethseg/')

    mesh_files = [
        *sorted(root.glob('challenge_dataset/aligned/**/*.obj')),
        # *sorted(root.glob('result_complete/**/*full*.ply'))
    ]
    label_files = [
        # *sorted(root.glob('complete_partial/**/*.json')),
        # *sorted(root.glob('complete_full/**/*.json')),
        *sorted(root.glob('full_dataset/lower_upper/cases/**/*.json'))
    ]

    # determine FDI pair distributions only on train data
    # with open('full_fold_0.txt', 'r') as f:
    #     stems = [Path(l.strip()).stem for l in f.readlines() if l.strip()]
    # with open('partial_fold_0.txt', 'r') as f:
    #     stems += [Path(l.strip()).stem for l in f.readlines() if l.strip()]
    # mesh_files = [f for f in mesh_files if f.stem not in stems]
    # label_files = [f for f in label_files if f.stem not in stems]
    stems = set(f.stem for f in mesh_files) & set(f.stem for f in label_files)
    mesh_files = [f for f in mesh_files if f.stem in stems]
    label_files = [f for f in label_files if f.stem in stems]

    # determine means and covariance matrices of multi-varite Gaussians
    pair_means, pair_covs = determine_fdi_pair_distributions(mesh_files, label_files)

    # save modeled tooth pair distributions to storage
    out = {'means': pair_means.tolist(), 'covs': pair_covs.tolist()}
    with open('3dteethseg_fdi_pair_distrs.json', 'w') as f:
        json.dump(out, f)
