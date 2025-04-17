# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.


# copied from https://github.com/facebookresearch/home-robot/blob/main/src/home_robot/home_robot/mapping/voxel/feature/conceptfusion.py

import logging
from typing import List, Optional

import torch

from preprocessing.image_features.mask_embedding import FeatureImageGenerator
from preprocessing.voxelized_pointcloud import VoxelizedPointcloud

logger = logging.getLogger(__name__)


def unproject_masked_depth_to_xyz_coordinates(
    depth: torch.Tensor,
    pose: torch.Tensor,
    inv_intrinsics: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Returns the XYZ coordinates for a batch posed RGBD image.

    Args:
        depth: The depth tensor, with shape (B, 1, H, W)
        mask: The mask tensor, with the same shape as the depth tensor,
            where True means that the point should be masked (not included)
        inv_intrinsics: The inverse intrinsics, with shape (B, 3, 3)
        pose: The poses, with shape (B, 4, 4)

    Returns:
        XYZ coordinates, with shape (N, 3) where N is the number of points in
        the depth image which are unmasked
    """

    batch_size, _, height, width = depth.shape
    if mask is None:
        mask = torch.full_like(depth, fill_value=False, dtype=torch.bool)
    flipped_mask = ~mask

    # Gets the pixel grid.
    xs, ys = torch.meshgrid(
        torch.arange(0, width, device=depth.device),
        torch.arange(0, height, device=depth.device),
        indexing="xy",
    )
    xy = torch.stack([xs, ys], dim=-1)[None, :, :].repeat_interleave(batch_size, dim=0)
    xy = xy[flipped_mask.squeeze(1)]
    xyz = torch.cat((xy, torch.ones_like(xy[..., :1])), dim=-1)

    # Associates poses and intrinsics with XYZ coordinates.
    inv_intrinsics = inv_intrinsics[:, None, None, :, :].expand(
        batch_size, height, width, 3, 3
    )[flipped_mask.squeeze(1)]
    pose = pose[:, None, None, :, :].expand(batch_size, height, width, 4, 4)[
        flipped_mask.squeeze(1)
    ]
    depth = depth[flipped_mask]

    # Applies intrinsics and extrinsics.
    xyz = xyz.to(inv_intrinsics).unsqueeze(1) @ inv_intrinsics.permute([0, 2, 1])
    xyz = xyz * depth[:, None, None]
    xyz = (xyz[..., None, :] * pose[..., None, :3, :3]).sum(dim=-1) + pose[
        ..., None, :3, 3
    ]
    xyz = xyz.squeeze(1)

    return xyz


class UnprojectRGBDFeatureSLAM:
    NO_MASK_IDX = 255

    def __init__(
        self,
        image_feature_generator: Optional[FeatureImageGenerator] = None,
        min_depth: float = 0.1,
        max_depth: float = 4.0,
        device: Optional[str] = None,
    ) -> None:
        """
        Creates an UnprojectRGBDSLAM instance. Conceptfusion is one particular example, using a specific way of generating features

        Keeps a point cloud and adds featurized points to it.
         - clear() : Resets the point cloud
         - generate_mask_features() : takes an image and returns a feature image using various masks and the provided 2D encoder
         - build_scene() : Takes as input a dict(str, Tensor) of images, depths, poses and intrinsics to generate the featurized point cloud
         - add_image() : Increments the point cloud with a new RGB image, depth, pose and intrinsic.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda":
            device = f"cuda:{torch.cuda.current_device()}"
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.device = device
        self.image_feature_generator = image_feature_generator
        self.feat_dim = None

    def clear(self):
        pass

    @torch.no_grad()
    def add_image(
        self,
        voxel_map: VoxelizedPointcloud,
        image: torch.Tensor,
        depth: torch.Tensor,
        pose: torch.Tensor,
        intrinsic: torch.Tensor,
        features: Optional[torch.Tensor] = None,
        frame_path: Optional[int] = None,
    ):
        image = image.unsqueeze(0)
        depth = depth.unsqueeze(0)
        pose = pose.unsqueeze(0)
        intrinsic = intrinsic.unsqueeze(0)
        features = features.unsqueeze(0) if features is not None else None
        frame_path = [frame_path] if frame_path is not None else None
        return self.add_batch_image(
            voxel_map=voxel_map,
            image=image,
            depth=depth,
            pose=pose,
            intrinsic=intrinsic,
            features=features,
            frame_path=frame_path,
        )

    @torch.no_grad()
    def add_batch_image(
        self,
        voxel_map: VoxelizedPointcloud,
        image: torch.Tensor,
        depth: torch.Tensor,
        pose: torch.Tensor,
        intrinsic: torch.Tensor,
        features: Optional[torch.Tensor] = None,
    ):
        """
        image : (batch x color x width x height)
        depth : (batch x width x height)
        pose : (batch x4 x 4)
        intrinsics : (batch x4 x 4)
        intrinsics : (batch x4 x 4)
        """
        if image.shape[0] == 0:
            return voxel_map.get_pointcloud()
        rgb = image.permute(0, 2, 3, 1).to(self.device)
        depth = depth.unsqueeze(dim=3).permute(0, 3, 1, 2).to(self.device)

        camera_pose = pose.float().to(self.device)

        if features is None and self.image_feature_generator is not None:
            features = []
            for i in range(rgb.shape[0]):
                features.append(self.image_feature_generator.generate_features(rgb[i]))
            features = torch.cat(features, dim=0)

        original_intrinsics = intrinsic.to(self.device)

        xyz = unproject_masked_depth_to_xyz_coordinates(
            depth,
            camera_pose,
            original_intrinsics.inverse()[:, :3, :3],
        ).to(self.device)
        valid_depth = torch.full_like(depth, fill_value=True, dtype=torch.bool)
        if depth is not None:
            valid_depth = (depth > self.min_depth) & (depth < self.max_depth)
        valid_depth = valid_depth.flatten()
        xyz = xyz[valid_depth]
        if features is not None:
            features = features.reshape(-1, features.shape[-1])[valid_depth].to(
                self.device
            )
        rgb = rgb.to(self.device).reshape(-1, 3)[valid_depth]
        try:
            voxel_map.add(points=xyz, features=features, rgb=rgb)
        except IndexError:
            pass
        return voxel_map.get_pointcloud()
