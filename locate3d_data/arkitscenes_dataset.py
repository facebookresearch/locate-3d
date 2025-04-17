# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import Any, Dict, List, Optional, Tuple, Union
from natsort import natsorted
from pathlib import Path
import numpy as np
import torch
import pyminiply
import json

from locate3d_data.data_utils import (
    get_image_from_path,
    get_depth_image_from_path,
    intrinsic_array_to_matrix,
    infer_sky_direction_from_poses,
    interpolate_camera_poses,
    rotate_frames_90_degrees_clockwise_about_camera_z,
    six_dim_pose_to_transform,
)


class ARKitScenesDataset:
    DEFAULT_HEIGHT = 192.0
    DEFAULT_WIDTH = 256.0
    DEPTH_SCALE_FACTOR = 0.001  # to MM
    frame_skip = 30

    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.root_dir = Path(self.dataset_path) / "raw"
        self.rgb_folder = "lowres_wide"
        self.depth_folder = "lowres_depth"
        self.intrinsics_folder = "lowres_wide_intrinsics"

    def get_scan(self, scene_id):
        # Construct possible paths to the PLY file
        training_path = os.path.join(
            self.dataset_path, "raw", "Training", scene_id, f"{scene_id}_3dod_mesh.ply"
        )
        validation_path = os.path.join(
            self.dataset_path,
            "raw",
            "Validation",
            scene_id,
            f"{scene_id}_3dod_mesh.ply",
        )

        # Check if the file exists in Training or Validation
        if os.path.exists(training_path):
            ply_path = training_path
        elif os.path.exists(validation_path):
            ply_path = validation_path
        else:
            raise FileNotFoundError(
                f"PLY file for scene {scene_id} in ARKitScenes dataset not found."
            )
        xyz, _, _, _, rgb = pyminiply.read(ply_path)
        xyz = torch.tensor(xyz)
        rgb = torch.tensor(rgb)

        return xyz, rgb

    def get_camera_views(self, scan_name):
        if (self.root_dir / "Training" / scan_name).exists():
            working_dir = self.root_dir / "Training" / scan_name
        elif (self.root_dir / "Validation" / scan_name).exists():
            working_dir = self.root_dir / "Validation" / scan_name
        else:
            raise FileNotFoundError(f"{scan_name} not found in Training or Validation")

        scene_rgb_dir = working_dir / self.rgb_folder
        scene_rgb_files = [str(s) for s in scene_rgb_dir.iterdir()]

        def get_endswith(f_list, endswith):
            return list(natsorted([s for s in f_list if s.endswith(endswith)]))

        frame_idxs = torch.arange(len(get_endswith(scene_rgb_files, ".png")))[
            :: self.frame_skip
        ]

        # RGB
        img_names = get_endswith(scene_rgb_files, ".png")[:: self.frame_skip]
        img_names = [
            img_name
            for img_name in img_names
            if os.path.exists(img_name.replace(self.rgb_folder, self.depth_folder))
        ]

        assert len(img_names) > 0, f"Found zero images for scene {scan_name}"

        images = []
        for i_name in img_names:
            img = get_image_from_path(
                i_name, height=self.DEFAULT_HEIGHT, width=self.DEFAULT_WIDTH
            )
            images.append(img)

        # Depth
        depth_names = [
            Path(img_name.replace(self.rgb_folder, self.depth_folder))
            for img_name in img_names
            if os.path.exists(img_name.replace(self.rgb_folder, self.depth_folder))
        ]

        depths = []
        for d_name in depth_names:
            depth = get_depth_image_from_path(
                d_name,
                height=self.DEFAULT_HEIGHT,
                width=self.DEFAULT_WIDTH,
                scale_factor=self.DEPTH_SCALE_FACTOR,
            )
            depths.append(depth)

        # Intrinsics
        img_timestamps = [
            float(os.path.basename(img_name).replace(".png", "").split("_")[-1])
            for img_name in img_names
        ]

        intrinsic_file_names = [
            img_name.replace(self.rgb_folder, self.intrinsics_folder).replace(
                ".png", ".pincam"
            )
            for img_name in img_names
            if os.path.exists(
                img_name.replace(self.rgb_folder, self.intrinsics_folder).replace(
                    ".png", ".pincam"
                )
            )
        ]
        assert len(intrinsic_file_names) == len(
            img_names
        ), f"Unequal number of color and intrinsic images for scene {scan_name} ({len(img_names)} != ({len(intrinsic_file_names)}))"

        intrinsics = []
        for i_name in intrinsic_file_names:
            intrinsic = np.loadtxt(i_name)
            intrinsic = torch.from_numpy(intrinsic_array_to_matrix(intrinsic))
            intrinsics.append(intrinsic)

        # Poses
        poses_file = working_dir / "lowres_wide.traj"
        timestamped_poses = np.loadtxt(poses_file)
        pose_timestamps, poses = timestamped_poses[:, 0], timestamped_poses[:, 1:]

        clipped_img_timestamps = np.clip(
            img_timestamps,
            min(pose_timestamps),
            max(pose_timestamps),
        )
        poses = interpolate_camera_poses(
            pose_timestamps,
            np.stack(
                [six_dim_pose_to_transform(pose_6d) for pose_6d in poses],
                axis=0,
            ),
            clipped_img_timestamps,
        )

        # Infer sky directions for ARKit from poses
        # This is a workaround for incorrect sky directions in the dataset
        sky_dir = infer_sky_direction_from_poses(poses)
        num_rotations = {"UP": 0, "DOWN": 2, "LEFT": 3, "RIGHT": 1}

        # Stack all data for processing
        images = torch.stack(images)
        depths = torch.stack(depths)
        poses = torch.from_numpy(poses).float()
        intrinsics = torch.stack(intrinsics).float()

        # Rotate frames based on inferred sky directions
        images, depths, poses, intrinsics = (
            rotate_frames_90_degrees_clockwise_about_camera_z(
                images,
                depths,
                poses,
                intrinsics,
                images.shape[3],
                images.shape[2],
                k=num_rotations[sky_dir],
            )
        )

        return {
            "cam_to_world": poses.float(),
            "cam_K": intrinsics.float(),
            "rgb": images.float(),
            "depth_zbuffer": depths.float(),
        }
