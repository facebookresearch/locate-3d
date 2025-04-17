# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.


import logging
import os
import re
from typing import Any, Dict, List, Optional

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from preprocessing.voxelized_pointcloud import VoxelizedPointcloud

logger = logging.getLogger(__name__)
MAX_OBJ_NAME_LEN = 77


class FeatureLifter3D:

    def __init__(self, cfg):

        self.batch_size = cfg.get("unproject_frame_batch_size", 1)
        self.voxelized_pointcloud_kwargs = cfg.get("voxelized_pointcloud", {})

        logger.info("instantiating feature slam", cfg.feature_slam)
        self.feature_slam = hydra.utils.instantiate(cfg.feature_slam)
        logger.info(f"Feature generator: {self.feature_slam.image_feature_generator}")

    def lift_frames(self, camera_views: Dict):
        """
        Runs the observation transformation. For example, for the sparse
        voxel map, take in full frame observations and return object images
        """

        vpc = VoxelizedPointcloud(**self.voxelized_pointcloud_kwargs)

        N = camera_views["rgb"].shape[0] // self.batch_size + 1
        for i in range(N):
            start, end = i * self.batch_size, (i + 1) * self.batch_size
            self.feature_slam.add_batch_image(
                vpc,
                camera_views["rgb"][start:end],
                camera_views["depth_zbuffer"][start:end],
                camera_views["cam_to_world"][start:end],
                camera_views["cam_K"][start:end],
            )

        (
            points_reduced,
            features_reduced,
            weights_reduced,
            rgb_reduced,
        ) = vpc.get_pointcloud()

        return {
            "points_reduced": points_reduced,
            "features_reduced": features_reduced,
            "weights_reduced": weights_reduced,
            "rgb_reduced": rgb_reduced,
        }
