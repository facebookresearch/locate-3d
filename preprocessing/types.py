import copy
from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Text, Tuple, Union

import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image
from torch import Tensor

from cortex.utils.pose.pose_utils import get_rotation_matrix_z


def get_rotation_matrix_z(k: int):
    """
    Generate a rotation matrix for k*90 degrees rotation around the Z-axis.

    Args:
        k (int): The number of 90-degree increments to rotate.

    Returns:
        torch.Tensor: A 4x4 rotation matrix.
    """
    theta = np.radians(90 * k)  # Convert degrees to radians
    cos = np.cos(theta)
    sin = np.sin(theta)

    return torch.tensor(
        [[cos, -sin, 0, 0], [sin, cos, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
        dtype=torch.float32,
    )


def rotate_intrinsics_90_degrees_clockwise_about_camera_z(cam_K, W, H, k):
    """
    Adjust the intrinsic matrix for camera rotation.

    Parameters:
    cam_K : torch.Tensor
        The original intrinsic matrix [batch_size, 3, 3].
    W : int
        The width of the image.
    H : int
        The height of the image.
    k : int
        Number of 90-degree clockwise rotations.

    Returns:
    cam_K : torch.Tensor
        The rotated intrinsic matrix [batch_size, 3, 3].
    """

    k = k % 4

    if k == 0:
        return cam_K

    f_x = cam_K[:, 0, 0].clone()
    f_y = cam_K[:, 1, 1].clone()
    c_x = cam_K[:, 0, 2].clone()
    c_y = cam_K[:, 1, 2].clone()

    assert (cam_K[:, 0, 1] == 0).all()
    assert (cam_K[:, 1, 0] == 0).all()
    assert (cam_K[:, 2, :2] == 0).all()
    assert (cam_K[:, 2, 2] == 1).all()

    if k == 1:  # 90-degree clockwise rotation
        cam_K[:, 0, 0] = f_y
        cam_K[:, 0, 2] = c_y
        cam_K[:, 1, 1] = f_x
        cam_K[:, 1, 2] = W - c_x

    elif k == 2:  # 180-degree rotation
        cam_K[:, 0, 2] = W - c_x
        cam_K[:, 1, 2] = H - c_y

    elif k == 3:  # 270-degree clockwise rotation
        cam_K[:, 0, 0] = f_y
        cam_K[:, 0, 2] = H - c_y
        cam_K[:, 1, 1] = f_x
        cam_K[:, 1, 2] = c_x

    else:
        raise ValueError(
            "k must be 1, 2, or 3 for 90, 180, or 270-degree rotation respectively."
        )

    return cam_K


def deepcopy_tensor_override(obj):
    to_ret = dict()
    for k in obj.__dict__:
        v = obj.__dict__[k]
        if isinstance(v, torch.Tensor):
            to_ret[k] = v.detach().clone()
        else:
            to_ret[k] = copy.deepcopy(v)
    return to_ret


def _get_field(data, key, ensure_len):
    """
    This is just a convenience function to grab a tensor from a dict and do some validation
    This will probably never get used outside of this file
    """
    res = data.get(key, [None] * ensure_len)
    assert (
        len(res) == ensure_len
    ), f"{key}: {res.shape=} got unexpected length (not {ensure_len=})"
    return res


@dataclass
class SemanticFeatureImage:
    #  H x W (uint8), 255 means no value, other values are mask idx
    semantic_mask: Image = None
    local_feature: Optional[Tensor] = None  # MASK_NUMBER x DIM
    no_mask_id: int = -1

    @classmethod
    def from_tensor(
        cls,
        semantic_mask: Tensor,
        local_feature: Optional[Tensor] = None,
        no_mask_id=-1,
    ):
        assert semantic_mask.dtype == torch.int32
        pil_semantic_mask = F.to_pil_image(
            semantic_mask.to(torch.int32)
        )  # torch.short not supported, so we use uint8
        if local_feature is not None:
            local_feature = local_feature.half()
        return cls(pil_semantic_mask, local_feature, no_mask_id)

    def to_tensor(self):
        if self.local_feature is not None:
            self.local_feature = self.local_feature.half()
        return (
            F.pil_to_tensor(self.semantic_mask)[0],
            self.local_feature,
        )

    def resize(self, shape: Tuple[int, int]):
        assert len(shape) == 2, f"shape must be 2D, got {shape}"
        desired_aspect_ratio = shape[0] / shape[1]  # W / H
        new_aspect_ratio = self.semantic_mask.width / self.semantic_mask.height
        assert (
            abs(desired_aspect_ratio - new_aspect_ratio) < 1e-6
        ), f"Aspect ratio mismatch: {shape} != {self.semantic_mask.size}"
        self.semantic_mask = self.semantic_mask.resize(shape, Image.NEAREST)

    def to_mask_features(self, device):
        segments, features = self.to_tensor()
        height, width = segments.shape
        n_features, h_dim = features.shape[0], features.shape[1]
        outfeat = torch.zeros(height, width, h_dim, dtype=torch.half, device=device)

        for _i in range(n_features):
            outfeat[segments == _i] = features[_i]
        return outfeat.half()

    def to_dino_features(self, device):
        _, features = self.to_tensor()
        return features.half()


@dataclass
class PointCloudData:
    points_reduced: Optional[Tensor] = None  # N x 3
    features_reduced: Optional[Tensor] = None  # N x H
    weights_reduced: Optional[Tensor] = None  # N
    rgb_reduced: Optional[Tensor] = None  # N x 3 values in [0,1]
    normals: Optional[Tensor] = None  # N x 3

    def __deepcopy__(self, memo):
        return PointCloudData(**deepcopy_tensor_override(self))

    def __post_init__(self):
        """
        Check format of data
        """
        assert len(self.points_reduced.shape) == 2
        assert self.points_reduced.shape[1] == 3

        if self.features_reduced is not None:
            assert len(self.features_reduced.shape) == 2
            assert (
                self.features_reduced.shape[0] == self.points_reduced.shape[0]
            )  # N == N

        if self.weights_reduced is not None:
            assert len(self.weights_reduced.shape) == 1
            assert (
                self.weights_reduced.shape[0] == self.points_reduced.shape[0]
            )  # N == N

        if self.rgb_reduced is not None:
            assert len(self.rgb_reduced.shape) == 2
            assert self.rgb_reduced.shape[0] == self.points_reduced.shape[0]  # N == N
            assert self.rgb_reduced.shape[1] == 3

            assert self.rgb_reduced.dtype.is_floating_point
            assert self.rgb_reduced.max().item() <= 1
            assert self.rgb_reduced.min().item() >= 0
        if self.normals is not None:
            assert self.normals.shape == self.points_reduced.shape


@dataclass
class Object:
    image: Tensor  # C x H x W values in [0,1]
    position: List[float] = None
    crop_id: int = None
    object_class: Text = None

    def __post_init__(self):
        """
        Check format of data
        """
        assert len(self.image.shape) == 3
        assert self.image.shape[0] == 3  # color channel first

        assert self.image.dtype.is_floating_point
        assert self.image.max().item() <= 1
        assert self.image.min().item() >= 0


@dataclass
class BoundingBoxes3D:
    """
    Do we want to use the types.bboxes_3d.py here? That is a very different data structure
    """

    bounds: Tensor  # K x 3 x 2 (mins | maxes)
    class_names: Optional[List[Text]] = None
    # scores: Optional[Tensor] = None # not needed yet but may be needed in future
    # orientation: Optional[Tensor] = None # 3 DoF

    def __post_init__(self):
        """
        Check format of data
        """
        assert len(self.bounds.shape) == 3
        assert self.bounds.shape[1] == 3
        assert self.bounds.shape[2] == 2

        if self.class_names is not None:
            assert len(self.class_names) == len(self.bounds)


@dataclass
class CameraViewsData:
    rgb: Optional[Tensor] = None  # B, C, H, W values in [0,1]
    depth_zbuffer: Optional[Tensor] = None  # [B, H, W] Metric
    cam_to_world: Optional[Tensor] = None  # [B, 4, 4] OpenCV convention
    cam_K: Optional[Tensor] = None  # [B, 3, 3] In pixels
    scene_id: Optional[str] = None  # A UNIQUE identifier for a tour in a scene
    view_id: Union[
        Optional[Tensor], List[str]
    ] = None  # A Tensor of indexes for the views used
    # features: Optional[Tensor] = None
    # instance_image: Optional[Tensor] = None
    # timestamp: Optional[datetime.timedelta] = None

    def __deepcopy__(self, memo):
        return CameraViewsData(**deepcopy_tensor_override(self))

    def __post_init__(self):
        """
        Check format of data
        """
        # Only one of rgb and scene id should be None
        assert (self.rgb is not None) or (self.scene_id is not None)

        if self.rgb is not None:
            assert len(self.rgb.shape) == 4
            assert self.rgb.shape[1] == 3  # color channel first after episode
            assert self.rgb.dtype.is_floating_point
            assert self.rgb.max().item() <= 1
            assert self.rgb.min().item() >= 0

        if self.depth_zbuffer is not None:
            assert self.rgb is not None
            assert len(self.depth_zbuffer.shape) == 3
            assert self.depth_zbuffer.shape[0] == self.rgb.shape[0]  # same episode len
            assert self.rgb.shape[-1] == self.depth_zbuffer.shape[-1]
            assert self.rgb.shape[-2] == self.depth_zbuffer.shape[-2]  # same H x W

        if self.cam_to_world is not None:
            assert self.rgb is not None
            assert len(self.cam_to_world.shape) == 3
            assert self.cam_to_world.shape[0] == self.rgb.shape[0]
            assert self.cam_to_world.shape[-1] == 4
            assert self.cam_to_world.shape[-2] == 4

        if self.cam_K is not None:
            assert self.rgb is not None
            assert len(self.cam_K.shape) == 3
            assert self.cam_K.shape[0] == self.rgb.shape[0]
            assert self.cam_K.shape[-1] == 3
            assert self.cam_K.shape[-2] == 3

        if self.scene_id is not None:
            assert self.view_id is not None
        if self.view_id is not None:
            if self.rgb is not None:
                assert len(self.view_id) == len(self.rgb)
            assert isinstance(self.view_id, list)

    @classmethod
    def from_dict(cls, data_dict: Dict[str, Tensor]):
        """
        Builds the frame_history field in an Observation
        Requires the Dict[str, Any] to have keys with the same name as CameraViewData

        TODO: Move this to the frame hist dataclass?
        """
        rgb = data_dict.get("rgb", [])
        n_frames = len(rgb)
        if n_frames == 0:
            return None

        depth = _get_field(data_dict, "depth_zbuffer", ensure_len=n_frames)
        cam_to_world = _get_field(data_dict, "cam_to_world", ensure_len=n_frames)
        cam_K = _get_field(data_dict, "cam_K", ensure_len=n_frames)
        scene_id = data_dict.get("scene_id", None)
        view_id = data_dict.get("view_id", None)
        return cls(
            rgb=rgb,
            depth_zbuffer=depth,
            cam_to_world=cam_to_world,
            cam_K=cam_K,
            scene_id=scene_id,
            view_id=view_id,
        )

    def rotate_frames_90_degrees_clockwise_about_camera_z(
        self, orig_W: int, orig_H: int, k: int = 1
    ):
        """
        Rotates the frame history 90 degrees clockwise about the camera Z-axis.

        Args:
            orig_W (int): The original width of the frames.
            orig_H (int): The original height of the frames
            k (int): Number of times to rotate the observations 90 degrees clockwise. Default is 1.

        """
        k = k % 4
        if k == 0:
            # if k == 0, do nothing
            return
        # Rotate RGB and depth zbuffer
        if self.rgb is not None:
            self.rgb = torch.rot90(self.rgb, k=k, dims=(2, 3))  # Rotate along H-W plane
        if self.depth_zbuffer is not None:
            self.depth_zbuffer = torch.rot90(self.depth_zbuffer, k=k, dims=(1, 2))

        if self.cam_to_world is not None:
            # Define the rotation matrix for 90 degrees clockwise k times about the z-axis
            rotation_matrix = get_rotation_matrix_z(k)
            # Rotate camera-to-world pose matrix
            self.cam_to_world = torch.matmul(self.cam_to_world, rotation_matrix)

        if self.cam_K is not None:
            self.cam_K = rotate_intrinsics_90_degrees_clockwise_about_camera_z(
                self.cam_K, orig_W, orig_H, k
            )

    @abstractmethod
    def create_empty_camera_view_data(scene_id: str, view_id: Union[Tensor, List[str]]):
        """
        Returns a dummy CameraViewsData object with all rgb, depth, cam_to_world, cam_K set to None
        """
        return CameraViewsData(
            rgb=None,
            depth_zbuffer=None,
            cam_to_world=None,
            cam_K=None,
            scene_id=scene_id,
            view_id=view_id,
        )


@dataclass
class Observations:
    bboxes_3d: Optional[BoundingBoxes3D] = None
    frame_history: Optional[CameraViewsData] = None
    objects: Optional[List[Object]] = None
    pointcloud: Optional[PointCloudData] = None
    latents: Optional[Tensor] = None

    def __deepcopy__(self, memo):
        return Observations(**deepcopy_tensor_override(self))

    @classmethod
    def from_dict(cls, data_dict: Dict[str, Any]):
        """
        Builds an Observation object from a dict, based on available keys in the dict
        """
        if "points" in data_dict and "features" in data_dict:
            if "scene_id" in data_dict:
                return cls(
                    pointcloud=PointCloudData(
                        points_reduced=data_dict["points"],
                        features_reduced=data_dict["features"],
                        rgb_reduced=data_dict["rgb"],
                    ),
                    frame_history=CameraViewsData(
                        scene_id=data_dict["scene_id"], view_id=[]
                    ),
                )
            return cls(
                pointcloud=PointCloudData(
                    points_reduced=data_dict["points"],
                    features_reduced=data_dict["features"],
                    rgb_reduced=data_dict["rgb"],
                )
            )
        if "frame_history" in data_dict or "rgb" in data_dict:
            frame_history = CameraViewsData.from_dict(data_dict)
            return cls(
                frame_history=frame_history,
            )
        return cls()

    def rotate_frames_90_degrees_clockwise_about_camera_z(
        self, orig_W: int, orig_H: int, k: int = 1
    ):
        """
        Rotates the frame history by 90 degrees clockwise about the camera z axis.

        Args:
            orig_W (int): The original width of the frames.
            orig_H (int): The original height of the frames.
            k (int): Number of times to rotate the observations 90 degrees clockwise. Default is 1.
        """
        if self.frame_history is not None:
            self.frame_history.rotate_frames_90_degrees_clockwise_about_camera_z(
                orig_W, orig_H, k=k
            )

    def to_dict(self):
        return {
            "points": self.pointcloud.points_reduced,
            "features": self.pointcloud.features_reduced,
            "rgb": self.pointcloud.rgb_reduced,
            "scene_id": self.frame_history.scene_id,
        }
