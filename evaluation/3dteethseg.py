from functools import partial
import math
import multiprocessing as mp
from pathlib import Path
import json

import numpy as np
import pymeshlab
from scipy.optimize import linear_sum_assignment
import scipy.spatial.distance as compute_dist_matrix
from sklearn.metrics import f1_score
import traceback
from tqdm import tqdm


def compute_tooth_size(points, centroid):
    size = np.sqrt(np.sum((centroid - points) ** 2, axis=0))
    return size


def calculate_jaw_TSA(gt_instances, pred_instances):
    """
    Teeth segmentation accuracy (TSA): is computed as the average F1-score over all instances of teeth point clouds.
    The F1-score of each tooth instance is measured as: F1=2*(precision * recall)/(precision+recall)

    Returns F1-score per jaw
    -------

    """
    gt_instances[gt_instances != 0] = 1
    pred_instances[pred_instances != 0] = 1
    return f1_score(gt_instances, pred_instances, average='micro')


def extract_centroids(instance_label_dict):
    centroids_list = []
    for k, v in instance_label_dict.items():
        centroids_list.append((v["centroid"]))
    return centroids_list


def centroids_pred_to_gt_attribution(gt_instance_label_dict, pred_instance_label_dict):
    gt_cent_list = extract_centroids(gt_instance_label_dict)
    pred_cent_list = extract_centroids(pred_instance_label_dict)
    M = compute_dist_matrix.cdist(gt_cent_list, pred_cent_list)
    row_ind, col_ind = linear_sum_assignment(M)

    matching_dict = {list(gt_instance_label_dict.keys())[i]: list(pred_instance_label_dict.keys())[j]
                    for i,j in zip(row_ind,col_ind)}

    return matching_dict


def calculate_jaw_TLA(gt_instance_label_dict, pred_instance_label_dict, matching_dict):

    """
    Teeth localization accuracy (TLA): mean of normalized Euclidean distance between ground truth (GT) teeth centroids and the closest localized teeth
    centroid. Each computed Euclidean distance is normalized by the size of the corresponding GT tooth.
    In case of no centroid (e.g. algorithm crashes or missing output for a given scan) a nominal penalty of 5 per GT
    tooth will be given. This corresponds to a distance 5 times the actual GT tooth size. As the number of teeth per
    patient may be variable, here the mean is computed over all gathered GT Teeth in the two testing sets.
    Parameters
    ----------
    matching_dict
    gt_instance_label_dict
    pred_instance_label_dict

    Returns
    -------
    """
    TLA = 0
    for inst, info in gt_instance_label_dict.items():
        if inst in matching_dict.keys():

            TLA += np.linalg.norm((gt_instance_label_dict[inst]['centroid'] - pred_instance_label_dict[matching_dict
            [inst]]['centroid']) / gt_instance_label_dict[inst]['tooth_size'])
        else:
            TLA += 5 * np.linalg.norm(gt_instance_label_dict[inst]['tooth_size'])

    return TLA/len(gt_instance_label_dict.keys())


def calculate_jaw_TIR(gt_instance_label_dict, pred_instance_label_dict, matching_dict, threshold=0.5):
    """
    Teeth identification rate (TIR): is computed as the percentage of true identification cases relatively to all GT
    teeth in the two testing sets. A true identification is considered when for a given GT Tooth,
    the closest detected tooth centroid : is localized at a distance under half of the GT tooth size, and is
    attributed the same label as the GT tooth
    Returns
    -------

    """
    tir = 0
    for gt_inst, pred_inst in matching_dict.items():
        dist = np.linalg.norm((gt_instance_label_dict[gt_inst]["centroid"]-pred_instance_label_dict[pred_inst]["centroid"])
                         /gt_instance_label_dict[gt_inst]['tooth_size'])
        if dist < threshold and gt_instance_label_dict[gt_inst]["label"]==pred_instance_label_dict[pred_inst]["label"]:
            tir += 1
    return tir/len(matching_dict)


def calculate_metrics(gt_label_dict, pred_label_dict):
    # read gt labels
    # with open(gt_labels_path) as f:
    #     gt_label_dict = json.load(f)
    gt_instances = np.array(gt_label_dict['instances'])
    gt_labels = np.array(gt_label_dict['labels'])

    u_instances = np.unique(gt_instances)
    u_instances = u_instances[u_instances != 0]

    pred_instances = np.array(pred_label_dict['instances'])
    pred_labels = np.array(pred_label_dict['labels'])

    # check if one instance match exactly one label else this instance(label) will be attributed to gingiva 0
    pred_instance_label_dict = {}
    u_pred_instances = np.unique(pred_instances)
    # delete 0 instance
    u_pred_instances = u_pred_instances[u_pred_instances != 0]
    for pred_inst in u_pred_instances:
        pred_label_inst = pred_labels[pred_instances == pred_inst]
        nb_predicted_labels_per_inst = np.unique(pred_label_inst)
        if len(nb_predicted_labels_per_inst) == 1:
            # compute predicted tooth center
            pred_verts = gt_label_dict["mesh_vertices"][pred_instances == pred_inst]
            pred_center = np.mean(pred_verts, axis=0)
            # tooth_size = compute_tooth_size(pred_verts, pred_center)
            pred_instance_label_dict[str(pred_inst)] = {"label": pred_label_inst[0], "centroid": pred_center}

        else:
            pred_labels[pred_instances == pred_inst] = 0
            pred_instances[pred_instances == pred_inst] = 0

    gt_instance_label_dict = {}
    for l in u_instances:
        gt_lbl = gt_labels[gt_instances == l]
        label = np.unique(gt_lbl)

        assert len(label) == 1
        # compute gt tooth center and size

        gt_verts = gt_label_dict["mesh_vertices"][gt_instances == l]
        gt_center = np.mean(gt_verts, axis=0)
        tooth_size = compute_tooth_size(gt_verts, gt_center)
        gt_instance_label_dict[str(l)] = {"label": label[0], "centroid": gt_center, "tooth_size": tooth_size}

    matching_dict = centroids_pred_to_gt_attribution(gt_instance_label_dict, pred_instance_label_dict)

    try:
        jaw_TLA = calculate_jaw_TLA(gt_instance_label_dict, pred_instance_label_dict, matching_dict)

    except Exception as e:
        print("error in jaw TLA calculation")
        print(str(e))
        print(traceback.format_exc())
        jaw_TLA = 0

    try:
        jaw_TSA = calculate_jaw_TSA(gt_instances, pred_instances)
    except Exception as e:
        print("error in jaw TSA calculation")
        print(str(e))
        print(traceback.format_exc())
        jaw_TSA = 0

    try:
        jaw_TIR = calculate_jaw_TIR(gt_instance_label_dict, pred_instance_label_dict, matching_dict)
    except Exception as e:
        print("error in jaw TIR calculation")
        print(str(e))
        print(traceback.format_exc())
        jaw_TIR = 0

    return jaw_TLA, jaw_TSA, jaw_TIR


def get_teeth_vertices(mesh, labels_path):
    with open(labels_path) as f:
        label_dict = json.load(f)
    labels = label_dict['instances']
    u_labels = np.unique(labels)
    u_labels = u_labels[u_labels != 0]
    teeth_list = []
    teeth_centers = []
    for l in u_labels:
        verts = mesh.vertices[labels == l]
        teeth_centers.append(np.mean(verts, axis=0))
        teeth_list.append(mesh.vertices[labels == l])
    return teeth_list, teeth_centers


def process_scan(gt_files, pred_filename):
    with open(pred_filename, 'r') as f:
        pred_label_dict = json.load(f)
    pred_label_dict['instances'] = (np.array(pred_label_dict['instances']) + 1).tolist()

    # load ground-truth segmentations
    gt_filename = [f for f in gt_files if f.name == pred_filename.name][0]
    with open(gt_filename, 'r') as f:
        gt_label_dict = json.load(f)

    instances = np.array(gt_label_dict['instances'])
    _, instances, counts = np.unique(instances, return_inverse=True, return_counts=True)

    labels = np.array(gt_label_dict['labels'])
    labels[(counts < 100)[instances]] = 0
    instances[(counts < 100)[instances]] = 0
    gt_label_dict['labels'] = labels.tolist()
    gt_label_dict['instances'] = instances.tolist()
    
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(str(gt_filename.with_suffix('.ply')))
    vertices = ms.current_mesh().vertex_matrix()
    gt_label_dict['mesh_vertices'] = vertices

    jaw_TLA, jaw_TSA, jaw_TIR = calculate_metrics(gt_label_dict, pred_label_dict)

    return math.exp(-jaw_TLA), jaw_TSA, jaw_TIR



if __name__ == "__main__":
    # gt_dir = Path('/home/mkaailab/Documents/IOS/Brazil/cases')    
    gt_dir = Path('/mnt/diag/IOS/3dteethseg/full_dataset/lower_upper')
    pred_dir = Path('3dteethseg_2m2')
    
    gt_dir = Path('/home/mkaailab/Documents/IOS/Brazil/cases')
    pred_dir = Path('mixed_ios')
    TLA, TSA, TIR = [], [], []

    gt_files = sorted(gt_dir.glob(f'**/*.json'))
    pred_files = sorted(pred_dir.glob('*'))

    with mp.Pool(16) as p:
        i = p.imap_unordered(partial(process_scan, gt_files), pred_files)
        for tla, tsa, tir in tqdm(i, total=len(pred_files)):
            
            TLA.append(tla)
            TSA.append(tsa)
            TIR.append(tir)

    score = (np.mean(TSA) + np.mean(TLA) + np.mean(TIR))/3
    print("TSA : {} +- {}".format(np.mean(TSA), np.std(TSA)))
    print("TLA : {} +- {}".format(np.mean(TLA), np.std(TLA)))
    print("TIR : {} +- {}".format(np.mean(TIR), np.std(TIR)))
    print(" score : ", score)

    # export metrics to /output/metrics.json
    score_dict = {
        "global": score,
        "TSA": np.mean(TSA),
        "TLA": np.mean(TLA),
        "TIR": np.mean(TIR)
    }

    with open('/output/metrics.json', 'w') as fp:
        json.dump(score_dict, fp)

