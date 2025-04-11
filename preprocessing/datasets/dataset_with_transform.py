import logging
from typing import List, Optional

import numpy as np
from omegaconf import OmegaConf

from preprocessing.datasets.base_dataset import BaseDataset
from preprocessing.datasets.base_transform import BaseTransform
from preprocessing.types import Action, Observations, TrainingSample

logger = logging.getLogger(__name__)


class DatasetWithTransform(BaseDataset):
    def __init__(
        self,
        datasets: List[BaseDataset],
        transforms: Optional[List[BaseTransform]] = None,
    ):
        assert (
            len(datasets) > 0
        ), "Need at least one dataset to use in DatasetWithTransform"
        self._datasets = datasets
        self._transforms = transforms or []
        self._offsets = np.cumsum([len(_d) for _d in self._datasets])
        self._offsets[1:] = self._offsets[:-1]
        self._offsets[0] = 0
        self._len = sum(len(_d) for _d in self._datasets)

    def __len__(self):
        return self._len

    def _global_index_to_dataset_index(self, global_idx):
        dataset_idx = self._get_dataset_index_from_sample_index(global_idx)
        idx_within_dataset = global_idx - self._offsets[dataset_idx]
        return dataset_idx, idx_within_dataset

    def _get_dataset_index_from_sample_index(self, idx: int):
        result = -1  # noqa: SIM113
        for offset in self._offsets:
            if offset > idx:
                return result
            result += 1
        return len(self._offsets) - 1

    def __getitem__(self, idx):
        dataset_idx, idx_within_dataset = self._global_index_to_dataset_index(idx)
        sample = self._datasets[dataset_idx][idx_within_dataset]
        for _transform in self._transforms:
            sample, _ = _transform.transform_sample(sample)
        return sample

    def sorted_idx_by_num_points(self):
        if len(self._datasets) == 1:
            return self._datasets[0].sorted_idx_by_num_points()

        idx_num_points = []
        for d in self._datasets:
            idx_num_points.extend(d.get_idx_to_num_points())

        return np.argsort(idx_num_points).tolist()
