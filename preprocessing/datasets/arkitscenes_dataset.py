# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import logging
import os
from enum import Enum, auto
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
import torch
from natsort import natsorted

from preprocessing.datasets.base_dataset import BaseDataset
from preprocessing.datasets.pose_utils import (
    infer_sky_direction_from_poses,
    interpolate_camera_poses,
    six_dim_pose_to_transform,
)
from preprocessing.datasets.posed_rgbd_dataset import PosedRGBDDataset
from preprocessing.types import Observations, TrainingSample

logger = logging.getLogger(__name__)


CLASS_LABELS = [
    "bathtub",
    "bed",
    "build_in_cabinet",
    "cabinet",
    "chair",
    "dishwasher",
    "fireplace",
    "oven",
    "refrigerator",
    "shelf",
    "sink",
    "sofa",
    "stairs",
    "stool",
    "stove",
    "table",
    "toilet",
    "tv_monitor",
    "washer",
]

CLASS_IDS = list(range(len(CLASS_LABELS)))

CLASS_LABEL_TO_CLASS_ID = {label: i for i, label in enumerate(CLASS_LABELS)}

MAX_CLASS_IDX = max(CLASS_IDS)


class ARKitScenesModalities(Enum):
    ALL = auto()
    RGB = auto()
    DEPTH = auto()
    POSE = auto()
    INTRINSICS = auto()
    BBOX_3D = auto()


class ARKitScenesDataset(PosedRGBDDataset):
    """
    This class loads ARKitScenes scenes through the __getitem__ function.
    Which modalities are loaded (RGB, depth, bounding boxes, etc) can be set through
    the constructor using the ARKitScenesModalities enum.
    """

    DEPTH_SCALE_FACTOR = 0.001  # to MM
    modalities_enum = ARKitScenesModalities
    read_intrinsics_per_frame = True
    read_pose_per_frame = False
    dataset_name = "arkitscenes"

    def __init__(
        self,
        root_dir: Union[str, Path],
        split: str = "train",
        resolution: str = "lowres",
        modalities: Union[str, Tuple[modalities_enum]] = (
            ARKitScenesModalities.RGB,
            ARKitScenesModalities.DEPTH,
            ARKitScenesModalities.POSE,
            ARKitScenesModalities.INTRINSICS,
        ),
        **kwargs,
    ):
        """
        Data loader for ARKitScenes dataset.
        We use the lowres version of the dataset, which has a resolution of 256x192.
        """
        self.resolution = resolution
        if self.resolution == "lowres":
            kwargs["height"] = 192
            kwargs["width"] = 256
            self.rgb_folder = "lowres_wide"
            self.depth_folder = "lowres_depth"
            self.intrinsics_folder = "lowres_wide_intrinsics"
        elif self.resolution == "highres":
            kwargs["height"] = 1440
            kwargs["width"] = 1920
            self.rgb_folder = "wide"
            self.depth_folder = "highres_depth"
            self.intrinsics_folder = "wide_intrinsics"
        else:
            raise ValueError(f"Resolution {resolution} not supported")

        super().__init__(root_dir, split, modalities=modalities, **kwargs)

    def retrieve_scene_list(self, split: str = "train"):
        scene_metadata = pd.read_csv(os.path.join(self.root_dir, "../metadata.csv"))
        # Filter scenes based on split and is_in_threedod
        filtered_scene_metadata = scene_metadata[
            (scene_metadata["fold"] == os.path.basename(self.root_dir))
            & scene_metadata["is_in_threedod"]
        ]
        if self.resolution == "highres":
            # further filter scenes that have "is_in_upsampled" set to True
            filtered_scene_metadata = filtered_scene_metadata[
                filtered_scene_metadata["is_in_upsampling"]
            ]
        scene_list = filtered_scene_metadata["video_id"].apply(str).tolist()
        return scene_list

    def set_root_dir(self, root_dir: Union[str, Path], split: str = "train"):
        super().set_root_dir(root_dir, split)
        if split == "train":
            self.root_dir = self.root_dir / "Training"
        elif split == "val":
            self.root_dir = self.root_dir / "Validation"
        else:
            raise ValueError("split needs to be one of train or val")

    @staticmethod
    def get_sky_direction_in_frames(observations: Observations) -> str:
        """
        While ARKitScenes does provide sky directions in metadata.csv, these can be inaccurate as noted in https://github.com/naver/dust3r/blob/69192aa322d279438390c109b8b85d5b859b5cdd/datasets_preprocess/preprocess_arkitscenes.py#L332.
        We can instead infer the sky direction from the poses.
        """
        return infer_sky_direction_from_poses(observations.frame_history.cam_to_world)

    def find_data(self, scan_name: str):
        scene_rgb_dir = self.root_dir / scan_name / self.rgb_folder
        scene_rgb_files = [str(s) for s in scene_rgb_dir.iterdir()]

        def get_endswith(f_list, endswith):
            return list(natsorted([s for s in f_list if s.endswith(endswith)]))

        frame_idxs = torch.arange(len(get_endswith(scene_rgb_files, ".png")))[
            :: self.frame_skip
        ]

        # RGB
        img_names = get_endswith(scene_rgb_files, ".png")[:: self.frame_skip]

        assert len(img_names) > 0, f"Found zero images for scene {scan_name}"

        # Depth
        depth_names = [
            Path(img_name.replace(self.rgb_folder, self.depth_folder))
            for img_name in img_names
            if os.path.exists(img_name.replace(self.rgb_folder, self.depth_folder))
        ]
        img_names = [
            img_name
            for img_name in img_names
            if os.path.exists(img_name.replace(self.rgb_folder, self.depth_folder))
        ]

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

        bbox_annotations_file = (
            self.root_dir / scan_name / f"{scan_name}_3dod_annotation.json"
        )
        with open(bbox_annotations_file, "r") as f:
            bbox_annotations = json.load(f)

        poses_file = self.root_dir / scan_name / "lowres_wide.traj"
        timestamped_poses = np.loadtxt(poses_file)
        pose_timestamps, poses = timestamped_poses[:, 0], timestamped_poses[:, 1:]

        return {
            "img_paths": img_names,
            "depth_paths": depth_names,
            "img_timestamps": img_timestamps,
            "pose_values": poses,
            "pose_timestamps": pose_timestamps,
            "intrinsics_paths": intrinsic_file_names,
            "bbox_annotations": bbox_annotations,
            "frame_indexes": frame_idxs,
        }

    def get_intrinsics(self, data: dict, indices: List[int], **kwargs) -> torch.Tensor:
        assert len(indices) == 1
        intrinsic = np.loadtxt(data["intrinsics_paths"][indices[0]])
        return torch.from_numpy(intrinsic_array_to_matrix(intrinsic)).unsqueeze(0)

    def get_poses(self, data: dict, indices: List[int], **kwargs) -> torch.Tensor:
        clipped_img_timestamps = np.clip(
            data["img_timestamps"],
            min(data["pose_timestamps"]),
            max(data["pose_timestamps"]),
        )
        poses = interpolate_camera_poses(
            data["pose_timestamps"],
            np.stack(
                [six_dim_pose_to_transform(pose_6d) for pose_6d in data["pose_values"]],
                axis=0,
            ),
            [clipped_img_timestamps[i] for i in indices],
        )
        poses = torch.from_numpy(poses).float()
        return poses

    def get_other_global_fields(
        self, data: dict, indices: List[int], scan_name: str, **kwargs
    ) -> dict:
        # Load bounding boxes
        if ARKitScenesModalities.BBOX_3D in self.modalities:
            boxes_aligned, box_classes, box_obj_ids = load_3d_bboxes(
                data["bbox_annotations"]
            )
            return {
                "boxes_aligned": boxes_aligned,
                "box_classes": box_classes,
                "box_obj_ids": box_obj_ids,
            }
        return {}


def load_3d_bboxes(bbox_annotations) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Loads 3D bounding boxes from annotations dict

    Returns:
        bounds: FloatTensor of box bounds (xyz mins and maxes)
        label_id: IntTensor of class IDs
        obj_id: IntTensor of object IDs
    """
    bboxes = []
    labels = []
    obj_ids = []
    for obj_id, object_annotation in enumerate(bbox_annotations["data"]):
        bbox_info = object_annotation["segments"]["obbAligned"]
        bbox_centroid = torch.tensor(bbox_info["centroid"])
        bbox_axes_lengths = torch.tensor(bbox_info["axesLengths"])
        bbox_mins = bbox_centroid - bbox_axes_lengths / 2
        bbox_maxs = bbox_centroid + bbox_axes_lengths / 2
        bbox = torch.stack([bbox_mins, bbox_maxs], axis=1)
        bboxes.append(bbox)
        obj_ids.append(obj_id)
        labels.append(CLASS_LABEL_TO_CLASS_ID[object_annotation["label"]])
    if len(bboxes) == 0:
        bboxes = torch.tensor([])
    else:
        bboxes = torch.stack(bboxes, axis=0)

    obj_ids = torch.tensor(obj_ids)
    labels = torch.tensor(labels)
    return bboxes, labels, obj_ids


def intrinsic_array_to_matrix(intrinsics: np.ndarray):
    """
    Converts the loaded 6D array to a 4x4 transformation matrix
    Loaded 6D array has the format: [width height focal_length_x focal_length_y principal_point_x principal_point_y]
    """
    fx = intrinsics[2]
    fy = intrinsics[3]
    cx = intrinsics[4]
    cy = intrinsics[5]

    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    return K


class ARKitScenesTrainingSampleDataset(ARKitScenesDataset, BaseDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, idx: int) -> TrainingSample:
        """
        A dataset where each item is a dictionary with the following keys:
         - goal : None
         - action : None
         - Observations : The observation provided to the agent
        """
        raw_data = ARKitScenesDataset.__getitem__(self, idx)
        return TrainingSample(
            goal=None,
            observations=raw_data["observations"],
            action=None,
            info=None,
        )
