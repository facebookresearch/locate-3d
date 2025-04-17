# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.


import torch
import pyminiply
import json
from pathlib import Path
from typing import List, Union, Optional
from PIL import Image
from torchvision.transforms import Compose, InterpolationMode, PILToTensor, Resize


class ScanNetPPDataset:
    DEPTH_SCALE_FACTOR = 0.001  # to MM

    def __init__(
        self,
        dataset_path: Union[str, Path],
        frame_skip: int = 30,
        image_scale: float = 1.0 / 3.0,
    ):
        self.dataset_path = Path(dataset_path)
        self.frame_skip = frame_skip
        self.image_scale = image_scale

        # Image resizing setup
        IPHONE_SHAPE = (1440, 1920)
        new_shape = (
            round(IPHONE_SHAPE[0] * image_scale),
            round(IPHONE_SHAPE[1] * image_scale),
        )
        self.image_shape = new_shape
        self.resize_scale = (
            new_shape[0] / IPHONE_SHAPE[0],
            new_shape[1] / IPHONE_SHAPE[1],
        )
        self.image_to_tensor = Compose(
            [
                PILToTensor(),
                Resize(self.image_shape, InterpolationMode.BILINEAR, antialias=False),
            ]
        )
        self.depth_to_tensor = Compose(
            [
                PILToTensor(),
                Resize(self.image_shape, InterpolationMode.NEAREST, antialias=False),
            ]
        )

    def get_scan(self, scene_name):
        ppath = (
            self.dataset_path / "data" / scene_name / "scans" / "mesh_aligned_0.05.ply"
        )
        xyz, _, _, _, rgb = pyminiply.read(str(ppath))
        xyz = torch.tensor(xyz)
        rgb = torch.tensor(rgb)

        with open(
            self.dataset_path / "data" / scene_name / "scans" / "segments_anno.json",
            "r",
        ) as f:
            a = json.load(f)
        seg_groups = a["segGroups"]

        seg = torch.zeros_like(xyz[:, 0])
        seg -= 1

        for sg in seg_groups:
            assert sg["id"] == sg["objectId"]
            seg[sg["segments"]] = sg["id"]

        return xyz, rgb, seg

    def _load_json(self, scene_name, frame_idxs: Optional[List[int]] = None):
        scene_path = self.dataset_path / "data" / scene_name
        with open(scene_path / "iphone" / "pose_intrinsic_imu.json", "r") as f:
            rtk_json = json.load(f)
        frame_names = list(rtk_json.keys())
        if frame_idxs is None:
            frame_names = frame_names[:: self.frame_skip]
        else:
            frame_names = [frame_names[i] for i in frame_idxs]
        return frame_names, rtk_json

    def get_poses(
        self, scene_name, frame_idxs: Optional[List[int]] = None
    ) -> List[torch.Tensor]:
        frame_names, rtk_json = self._load_json(scene_name, frame_idxs)
        poses = []
        for frame_name in frame_names:
            frame = rtk_json[frame_name]
            pose = torch.tensor(frame["aligned_pose"])
            poses.append(pose)

        return torch.stack(poses).float()

    def get_intrinsics(
        self, scene_name, frame_idxs: Optional[List[int]] = None
    ) -> List[torch.Tensor]:
        frame_names, rtk_json = self._load_json(scene_name, frame_idxs)

        intrinsics = []
        for frame_name in frame_names:
            frame = rtk_json[frame_name]
            intrinsic = torch.tensor(frame["intrinsic"])
            intrinsic[0, 0] *= self.resize_scale[1]
            intrinsic[1, 1] *= self.resize_scale[1]
            intrinsic[0, 2] *= self.resize_scale[0]
            intrinsic[1, 2] *= self.resize_scale[0]
            intrinsics.append(intrinsic)

        return torch.stack(intrinsics).float()

    def get_images(
        self, scene_name, frame_idxs: Optional[List[int]] = None
    ) -> List[torch.Tensor]:
        scene_path = self.dataset_path / "data" / scene_name
        frame_names, _ = self._load_json(scene_name, frame_idxs)

        images = []
        for frame_name in frame_names:
            rgb_frame_path = scene_path / "iphone" / "rgb" / (frame_name + ".jpg")
            rgb = Image.open(rgb_frame_path)
            rgb = self.image_to_tensor(rgb).float() / 255
            images.append(rgb)

        return torch.stack(images)

    def get_depths(
        self, scene_name, frame_idxs: Optional[List[int]] = None
    ) -> List[torch.Tensor]:
        scene_path = self.dataset_path / "data" / scene_name
        frame_names, _ = self._load_json(scene_name, frame_idxs)

        depths = []
        for frame_name in frame_names:
            depth_frame_path = scene_path / "iphone" / "depth" / (frame_name + ".png")
            depth = Image.open(depth_frame_path)
            depth = self.depth_to_tensor(depth).float() * self.DEPTH_SCALE_FACTOR
            depths.append(depth)

        return torch.stack(depths)[:, 0]

    def get_camera_views(self, scene_name, frame_idxs: Optional[List[int]] = None):
        return {
            "cam_to_world": self.get_poses(scene_name, frame_idxs),
            "cam_K": self.get_intrinsics(scene_name, frame_idxs),
            "rgb": self.get_images(scene_name, frame_idxs),
            "depth_zbuffer": self.get_depths(scene_name, frame_idxs),
        }
