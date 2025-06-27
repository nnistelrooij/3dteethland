import json

import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler
from torch_scatter import scatter_max


class ClassAwareSampler(WeightedRandomSampler):

    def __init__(
        self,
        dataset: Dataset,
    ):
        scan_labels = np.zeros((len(dataset), 86), dtype=int)
        for i, case_files in enumerate(dataset.files):
            with open(dataset.root / case_files[1], 'rb') as f:
                annotation = json.load(f)

            labels = np.array(annotation['labels'])
            instances = np.array(annotation['instances'])
            _, instances, counts = np.unique(
                instances, return_inverse=True, return_counts=True,
            )
            labels[(counts < 30)[instances]] = 0
            instances[(counts < 30)[instances]] = 0

            instance_labels = scatter_max(
                src=torch.from_numpy(labels),
                index=torch.from_numpy(instances),
                dim=0,
            )[0].numpy()

            for label in instance_labels[instance_labels > 0]:
                scan_labels[i, label] += 1

        repeat_factors = self._get_repeat_factors(scan_labels)

        super().__init__(
            weights=repeat_factors,
            num_samples=len(dataset),
            replacement=True,
        )

    def _get_repeat_factors(self, scan_labels):
        scan_labels[:, 0] = scan_labels.sum(1) == 0
        scan_labels = scan_labels[:, scan_labels.sum(0) > 0]

        class_counts = scan_labels.sum(0)
        scan_labels = scan_labels[:, np.argsort(class_counts)]
        class_counts = np.sort(class_counts)

        repeat_factors = np.zeros(scan_labels.shape[0])
        for i, labels in enumerate(scan_labels):
            rare_label = np.nonzero(labels)[0].min()
            repeat_factors[i] = 1 / class_counts[rare_label]

        return repeat_factors
