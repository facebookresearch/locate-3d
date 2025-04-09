import glob
import json
import logging
import os
from pathlib import Path
from types import Observations, TrainingSample
from typing import Optional

import filelock
import numpy as np
import torch
from filelock import FileLock

logger = logging.getLogger(__name__)


class CachedDataset:
    def __init__(
        self,
        key,
        cache_path: Optional[str] = None,
        acquire_lock: bool = False,
    ):
        logger.warning(f"Using CachedDataset at {cache_path}")

        self.cache_path = cache_path + "/" + key
        assert os.path.exists(self.cache_path), f"{self.cache_path} does not exist."
        self._chown_group_id = os.stat(cache_path).st_gid
        self.obs_files_path = glob.glob(f"{self.cache_path}/*.obs")
        self.pt_files_path = glob.glob(f"{self.cache_path}/*.pt")
        self.files_path = self.obs_files_path + self.pt_files_path

        logger.info(
            f"Loaded the CachedDataset at path {self.cache_path} with {len(self.files_path)} samples"
        )

        self.num_points_path = Path(self.cache_path) / "num_points.json"
        self.acquire_lock = acquire_lock
        self.load_num_points(acquire_lock=acquire_lock)

    def load_num_points(self, acquire_lock=True):
        if self.num_points_path.exists():
            if acquire_lock:
                try:
                    with FileLock(
                        str(self.num_points_path) + ".lock", timeout=10
                    ), open(self.num_points_path) as f:
                        self.num_points = json.load(f)
                except json.decoder.JSONDecodeError as e:
                    logger.error(f"JSON decode error: {self.num_points_path}")
                    raise e
                except filelock._error.Timeout:
                    logger.error(
                        f"Timeout on filelock: {self.num_points_path}. Trying without lock (unsafe read):"
                    )
                    with open(self.num_points_path, "rb") as f:
                        self.num_points = json.load(f)
            else:
                with open(self.num_points_path, "rb") as f:
                    self.num_points = json.load(f)
        else:
            self.num_points = {}

    def get_idx_to_num_points(self):
        return [self.num_points[Path(f).stem] for f in self.files_path]

    def sorted_idx_by_num_points(self):
        keys = [Path(f).stem for f in self.files_path]
        values = [self.num_points[x] for x in keys]

        return np.argsort(values).tolist()

    def __len__(self) -> int:
        return len(self.files_path)

    def __getitem__(self, idx: int) -> Observations:
        sample_path = self.files_path[idx]
        assert Path(sample_path).suffix == ".pt" or Path(sample_path).suffix == ".pth"
        sample = TrainingSample.from_dict(
            torch.load(sample_path, map_location=torch.device("cpu"))
        )
        return sample

    def exists(self, scene_id: str):
        path = f"{self.cache_path}/{scene_id}.pt"
        return os.path.exists(path)

    def get_scene(self, scene_id: str):
        path = f"{self.cache_path}/{scene_id}.pt"
        if path in self.files_path:
            return self[self.files_path.index(path)]

        return None

    def clear(self):
        for _path in self.files_path:
            os.remove(_path)
        self.files_path = []

    def add(self, sample: Observations, allow_no_feats=False):
        assert sample.pointcloud is not None
        if not allow_no_feats:
            assert sample.pointcloud.features_reduced is not None
        identifier = sample.frame_history.scene_id

        path = f"{self.cache_path}/{identifier}.pt"
        if os.path.exists(path):
            return

        # assert isinstance(sample, Observations)
        path = f"{self.cache_path}/{identifier}.pt"
        torch.save(sample.to_dict(), path)
        os.chown(path, -1, self._chown_group_id)
        self.files_path.append(path)

        # save num_points
        with FileLock(str(self.num_points_path) + ".lock"):
            self.load_num_points(acquire_lock=False)
            self.num_points[identifier] = len(sample.pointcloud.points_reduced)
            with open(self.num_points_path, "w") as outfile:
                json.dump(self.num_points, outfile)
