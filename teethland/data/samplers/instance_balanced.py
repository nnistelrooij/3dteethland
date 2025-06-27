from collections import defaultdict
from itertools import combinations_with_replacement
import json
from typing import Literal

import numpy as np
from torch.utils.data import Dataset, WeightedRandomSampler


class InstanceBalancedSampler(WeightedRandomSampler):

    def __init__(
        self,
        dataset: Dataset,
        key: Literal['labels', 'attributes']='attributes',
    ):
        scan_labels = np.zeros((len(dataset), 86), dtype=int)
        for i, case_files in enumerate(dataset.files):
            with open(dataset.root / case_files[1], 'rb') as f:
                annotation = json.load(f)

            if key == 'labels':
                labels = np.array(annotation['labels'])
            elif key == 'attributes':
                attributes = [max(attrs) if attrs else 0 for attrs in annotation['attributes']]
                labels = np.array(attributes)

            _, instances, counts = np.unique(
                annotation['instances'],
                return_inverse=True, return_counts=True,
            )            
            labels[(counts < 30)[instances]] = 0
            
            labels = np.unique(labels)
            for label in labels[labels > 0]:
                scan_labels[i, label] += 1

        repeat_factors = self._get_repeat_factors(scan_labels)

        super().__init__(
            weights=repeat_factors,
            num_samples=int(np.ceil(sum(repeat_factors))),
            replacement=True,
        )

    def _get_repeat_factors(self, scan_labels, repeat_thr: float=0.1):
        # 1. For each category pair c, compute the fraction # of images
        #   that contain it: f(c)
        image_freq, inst_freq = defaultdict(int), defaultdict(int)
        num_instances = 0
        for idx in range(scan_labels.shape[0]):
            if not np.any(scan_labels[idx]):
                continue

            instances = np.arange(86).repeat(scan_labels[idx])
            pair_ids = set()
            for pair in combinations_with_replacement(instances, 2):
                pair_id = 2**pair[0] + 2**pair[1]
                inst_freq[pair_id] += 1
                num_instances += 1
                pair_ids.add(pair_id)

            for pair_id in pair_ids:
                image_freq[pair_id] += 1

        for k, v in image_freq.items():
            image_freq[k] = v / scan_labels.shape[0]
            inst_freq[k] = inst_freq[k] / num_instances

        # 2. For each pair c, compute the pair-level repeat factor:
        #    r(c) = max(1, sqrt(t/f(c)))
        pair_repeat = {
            pair_id: max(1.0, np.sqrt(repeat_thr / np.sqrt(img_freq * inst_freq[pair_id])))
            for pair_id, img_freq in image_freq.items()
        }

        # 3. For each image I, compute the image-level repeat factor:
        #    r(I) = max_{c in I} r(c)
        repeat_factors = []
        for idx in range(scan_labels.shape[0]):
            if not np.any(scan_labels[idx]):
                repeat_factors.append(1.0)
                continue

            instances = np.arange(86).repeat(scan_labels[idx])
            pair_ids = set()
            for pair in combinations_with_replacement(instances, 2):
                pair_id = 2**pair[0] + 2**pair[1]
                pair_ids.add(pair_id)

            repeat_factor = max(pair_repeat[pair_id] for pair_id in pair_ids)
            repeat_factors.append(repeat_factor)

        return repeat_factors
