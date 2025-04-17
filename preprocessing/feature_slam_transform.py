# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.


import logging
import os
from typing import Any, Dict, Optional, Tuple

import hydra
import torch
from datasets.base_transform import BaseTransform
from datasets.cached_dataset import CachedDataset
from omegaconf import DictConfig, OmegaConf

from preprocessing.type_utils import Action, Observations, PointCloudData
from preprocessing.voxelized_pointcloud import VoxelizedPointcloud

logger = logging.getLogger(__name__)
MAX_OBJ_NAME_LEN = 77


class FeatureSLAMTransform(torch.nn.Module, BaseTransform):
    def __init__(self, cfg: OmegaConf, overrides: Optional[Dict[str, Any]] = None):
        """
        Goes from a Observation to a latent tensor representation with shape (K, dim)
        """
        torch.nn.Module.__init__(self)
        BaseTransform.__init__(self, cfg, overrides)

    def forward(self, obs, goal=None):
        return self.transform_observations(obs, goal)[0]

    def _assert_observations(self, obs: Observations) -> None:
        """
        Run checks to make sure the agent can act on these observations
        """
        assert obs.frame_history is not None
        assert obs.pointcloud is None

    def _transform_observations(
        self, obs: Observations, goal: Optional[str] = None
    ) -> (Observations, Optional[Dict]):
        """
        Runs the observation transformation. For example, for the sparse
        voxel map, take in full frame observations and return object images
        """
        _id = obs.frame_history.scene_id
        for _d in self.cached_datasets:
            if _d.exists(_id):
                new_obs = _d.get_scene(_id).observations
                if self.cfg.keep_frame_history and not self.use_cache_frame_history:
                    new_obs.frame_history = obs.frame_history
                return new_obs, new_obs.pointcloud
        logger.warning(f"FeatureSLAMTransform cache miss for {_id}")

        vpc = VoxelizedPointcloud(**self.voxelized_pointcloud_kwargs)

        N = obs.frame_history.rgb.shape[0] // self.batch_size + 1
        for i in range(N):
            start, end = i * self.batch_size, (i + 1) * self.batch_size
            self.feature_slam.add_batch_image(
                vpc,
                obs.frame_history.rgb[start:end],
                obs.frame_history.depth_zbuffer[start:end],
                obs.frame_history.cam_to_world[start:end],
                obs.frame_history.cam_K[start:end],
                frame_path=obs.frame_history.view_id[start:end],
            )
        (
            points_reduced,
            features_reduced,
            weights_reduced,
            rgb_reduced,
        ) = vpc.get_pointcloud()

        pointcloud = PointCloudData(
            points_reduced=points_reduced,
            features_reduced=features_reduced,
            weights_reduced=weights_reduced,
            rgb_reduced=rgb_reduced,
        )

        new_observation = Observations(pointcloud=pointcloud)
        if self.cfg.keep_frame_history:
            new_observation.frame_history = obs.frame_history
        return new_observation, pointcloud

    def initialize(self):
        self.batch_size = self.cfg.get("unproject_frame_batch_size", 1)
        self.voxelized_pointcloud_kwargs = self.cfg.get("voxelized_pointcloud", {})

        self.use_cpp = self.cfg.get("use_cpp", True)
        self.use_cache_frame_history = self.cfg.get("use_cache_frame_history", False)
        self.do_not_use_cache = self.cfg.get("do_not_use_cache", False)
        self.cache_path = self.cfg.get("cache_path", None)

        skip_fields = ["frame_history"]
        if self.use_cache_frame_history:
            assert (
                not self.use_cpp
            ), "CPP loading does not currently support loading frame history"
            assert (
                self.cfg.keep_frame_history
            ), "Conflicting configs. Use_cache_frame_history is True but keep_frame_history is False"
            skip_fields = []

        cache_path = self.cfg.get("cache_path", None)
        if os.path.exists(cache_path) and not self.do_not_use_cache:
            default_cache_keys = os.listdir(cache_path)
        else:
            default_cache_keys = []
            logger.warning(f"COULD NOT LOAD DEFAULT CACHE PATH AT {cache_path}")

        self.cached_datasets = [
            CachedDataset(
                key=k,
                skip_fields=skip_fields,
                use_cpp=self.use_cpp,
                cache_path=cache_path,
            )
            for k in self.cfg.get("cache_keys", default_cache_keys)
        ]

        if isinstance(self.cfg.feature_slam, DictConfig):
            logger.info("instantiating feature slam", self.cfg.feature_slam)
            self.feature_slam = hydra.utils.instantiate(self.cfg.feature_slam)
            logger.info(
                f"Feature generator: {self.feature_slam.image_feature_generator}"
            )
        else:
            self.feature_slam = self.cfg.feature_slam

    def _process_action(self, episode_id: int, action: Action) -> Tuple[int, Action]:
        return episode_id, action
