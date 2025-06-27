import json
import multiprocessing as mp
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, f1_score, jaccard_score
from tqdm import tqdm


def process_scan(
    pred_filename,
    score_thresh: float=0.0,
    iou_thresh: float=0.5,
):
    with open(pred_filename, 'r') as f:
        pred_label_dict = json.load(f)
    instances = np.array(pred_label_dict['instances'])
    if class_key == 'types':
        extra = np.array(pred_label_dict['extra'])[:, 0]
        types = np.where(instances >= 0, extra[instances], -1)            
        pred_label_dict['labels'] = types.tolist()
    

    # load ground-truth segmentations
    gt_filename = [f for f in gt_files if f.name == pred_filename.name][0]
    with open(gt_filename, 'r') as f:
        gt_label_dict = json.load(f)
    gt_label_dict['labels'] = gt_label_dict[class_key]
    
    pred_instances = np.array(pred_label_dict['instances'])
    if 'confidences' not in pred_label_dict:
        pred_label_dict['confidences'] = np.ones_like(pred_instances).tolist()
    pred_confidences = np.array(pred_label_dict['confidences'])
    pred_probs = []
    for label in np.unique(pred_instances):
        pred_probs.append(pred_confidences[pred_instances == label].mean())

    keep_points = (np.array(pred_probs) >= score_thresh)[pred_instances]
    pred_label_dict['instances'] = np.where(keep_points, pred_instances, 0)
    pred_label_dict['instances'] = np.unique(pred_label_dict['instances'], return_inverse=True)[1].tolist()
    pred_label_dict['labels'] = np.where(keep_points, pred_label_dict['labels'], 0)
    
    pred_instances = np.array(pred_label_dict['instances'])
    pred_point_idxs = []
    for label in np.unique(pred_instances)[1:]:
        point_idxs = np.nonzero(pred_instances == label)[0]
        pred_point_idxs.append(set(point_idxs.tolist()))

    gt_instances = np.array(gt_label_dict['instances'])
    gt_instances = np.unique(gt_instances, return_inverse=True)[1]

    gt_point_idxs = []
    for label in np.unique(gt_instances)[1:]:
        point_idxs = np.nonzero(gt_instances == label)[0]
        gt_point_idxs.append(set(point_idxs.tolist()))

    tooth_dices = []
    gt_labels, pred_labels = [], []
    ious = np.zeros((gt_instances.max(), max(pred_label_dict['instances'])))
    for i in range(gt_instances.max()):
        for j in range(max(pred_label_dict['instances'])):
            inter = len(gt_point_idxs[i] & pred_point_idxs[j])
            union = len(gt_point_idxs[i] | pred_point_idxs[j])
            iou = inter / union
            ious[i, j] = iou

            if iou < iou_thresh:
                continue
            
            dice = 2 * inter / (inter + union)
            tooth_dices.append(dice)
            gt_labels.append(gt_label_dict['labels'][next(iter(gt_point_idxs[i]))])
            pred_labels.append(pred_label_dict['labels'][next(iter(pred_point_idxs[j]))])

    fps = (np.max(ious, axis=0) < iou_thresh).sum()
    fns = (np.max(ious, axis=1) < iou_thresh).sum()
    tps = (max(gt_label_dict['instances']) + max(pred_label_dict['instances']) - fps - fns) / 2

    gt_points = gt_label_dict['labels']
    pred_points = pred_label_dict['labels']

    gt_labels = np.array(gt_labels)
    pred_labels = np.array(pred_labels)

    return tps, fps, fns, tooth_dices, gt_labels, pred_labels, gt_points, pred_points, pred_filename



if __name__ == "__main__":
    gt_dir = Path('/home/mkaailab/Documents/IOS/Brazil/cases')
    pred_dir = Path('/mnt/diag/IOS/mixed_ios/predictions/mixed_ios')

    stats = {
        'names': [], 'fps': [], 'fns': [], 'tps': [], 'dices': [],
        'gt_labels': [], 'pred_labels': [],
        'gt_points': [], 'pred_points': [],
    }

    pred_files = sorted(pred_dir.glob('**/*er.json'))
    # pred_files = [f for f in pred_files if len(re.split('\-|_', f.stem)[0]) == 4]
    # pred_files = [f for f in pred_files if len(re.split('\-|_', f.stem)[0]) == 5]
    # pred_files = [f for f in pred_files if len(re.split('\-|_', f.stem)[0]) > 5]

    # pred_files = [f for f in pred_files if (sample_dir / f.name).exists()]
    scan_ids = [f.stem for f in pred_files]
    _, unique_idxs = np.unique(scan_ids, return_index=True)
    pred_files = [pred_files[i] for i in unique_idxs]

    gt_files = sorted(gt_dir.glob('**/*er.json'))
    scan_ids = [f.stem for f in gt_files]
    _, unique_idxs = np.unique(scan_ids, return_index=True)
    gt_files = [gt_files[i] for i in unique_idxs]

    stems = set([f.stem for f in gt_files]) & set([f.stem for f in pred_files])
    gt_files = [f for f in gt_files if f.stem in stems]
    pred_files = [f for f in pred_files if f.stem in stems]
    
    # thresh = determine_optimal_threshold(pred_files, gt_files)
    class_key = 'labels'

    failures = []
    with mp.Pool(16) as p:
        i = p.imap_unordered(process_scan, pred_files)
        for out in tqdm(i, total=len(pred_files)):
            tps, fps, fns, tooth_dices, gt_labels, pred_labels, gt_points, pred_points, pred_filename = out

            # if False:
            if fps or fns or not np.all(gt_labels == pred_labels):
                failures.append((pred_filename, fps, fns, np.sum(gt_labels != pred_labels), gt_labels, pred_labels))
                
            stats['tps'].append(tps)
            stats['fps'].append(fps)
            stats['fns'].append(fns)
            stats['dices'].extend(tooth_dices)
            stats['gt_labels'].extend(gt_labels)
            stats['pred_labels'].extend(pred_labels)
            stats['gt_points'].extend(gt_points)
            stats['pred_points'].extend(pred_points)
            stats['names'].append(pred_filename.name)

    print('Tooth Precision:', sum(stats['tps']) / (sum(stats['tps']) + sum(stats['fps'])))
    print('Tooth Sensitivity:', sum(stats['tps']) / (sum(stats['tps']) + sum(stats['fns'])))
    print('Tooth F1:', 2 * sum(stats['tps']) / (2 * sum(stats['tps']) + sum(stats['fps']) + sum(stats['fns'])))
    print('Tooth Dice:', np.mean(stats['dices']))

    gt_labels = np.array(stats['gt_labels'])
    pred_labels = np.array(stats['pred_labels'])
    print('Tooth macro-F1:', f1_score(gt_labels, pred_labels, average='macro'))

    # macro_iou = jaccard_score(stats['gt_points'], stats['pred_points'], labels=np.unique(stats['gt_points'])[1:], average='macro')
    # print('macro-IoU:', macro_iou)

    # with open(pred_dir / f'{gt_dir.name}_failures.txt', 'w') as f:
    #     f.write('file,false_positive,false_negative,wrong_labels,gt_labels,pred_labels\n')
    #     for filename, fp, fn, count, gt_labels, pred_labels in failures:
    #         f.write(','.join([filename.stem, str(fp), str(fn), str(count), str(gt_labels), str(pred_labels)]) + '\n')

    upper_mask = np.isin(gt_labels // 10, np.array([1, 2, 5, 6]))

    fig, ax = plt.subplots(1, 1, figsize=(12, 7.2))
    cmd = ConfusionMatrixDisplay.from_predictions(
        gt_labels[upper_mask], pred_labels[upper_mask], ax=ax, text_kw={'fontsize': 8}, colorbar=False,
    )
    fig.canvas.draw()
    plt.savefig('cm_upper.png', dpi=600, bbox_inches='tight', pad_inches=None)
    plt.close()
    # plt.show(block=True)

    fig, ax = plt.subplots(1, 1, figsize=(12, 7.2))
    cmd = ConfusionMatrixDisplay.from_predictions(
        gt_labels[~upper_mask], pred_labels[~upper_mask], ax=ax, text_kw={'fontsize': 8}, colorbar=False,
    )
    fig.canvas.draw()
    plt.savefig('cm_lower.png', dpi=600, bbox_inches='tight', pad_inches=None)
    plt.close()
    # plt.show(block=True)
