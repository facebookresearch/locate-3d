# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import random
from abc import ABC, abstractmethod
from enum import Enum, auto
from numbers import Number
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from natsort import natsorted
from PIL import Image
from torch import Tensor
from torchvision.transforms import functional as F
from tqdm import tqdm

from preprocessing.type_utils import CameraViewsData, Observations

logger = logging.getLogger(__name__)


class Modalities(Enum):
    ALL = auto()
    RGB = auto()
    DEPTH = auto()
    POSE = auto()
    INTRINSICS = auto()


def get_image_from_path(
    image_path: Union[str, Path],
    height: Optional[int] = None,
    width: Optional[int] = None,
    keep_alpha: bool = False,
    resample=Image.BILINEAR,
) -> torch.Tensor:
    """Returns a 3 channel image.
    # Adapted from https://github.com/nerfstudio-project/nerfstudio/blob/main/nerfstudio/data/dataparsers/scannet_dataparser.py
    Args:
        image_idx: The image index in the dataset.
    """
    pil_image = Image.open(image_path)
    assert (height is None) == (width is None)  # Neither or both
    if height is not None and pil_image.size != (width, height):
        pil_image = pil_image.resize((width, height), resample=resample)

    image = F.to_tensor(pil_image)

    if not keep_alpha and image.shape[-1] == 4:
        image = image[:, :, :3]
        # image = image[:, :, :3] * image[:, :, -1:] + self._dataparser_outputs.alpha_color * (1.0 - image[:, :, -1:])
    return image


def get_depth_image_from_path(
    filepath: Path,
    height: Optional[int] = None,
    width: Optional[int] = None,
    scale_factor: float = 1.0,
    interpolation: int = cv2.INTER_NEAREST,
) -> torch.Tensor:
    """Loads, rescales and resizes depth images.
    Filepath points to a 16-bit or 32-bit depth image, or a numpy array `*.npy`.
    # Adapted from https://github.com/nerfstudio-project/nerfstudio/blob/main/nerfstudio/data/dataparsers/scannet_dataparser.py
    Args:
        filepath: Path to depth image.
        height: Target depth image height.
        width: Target depth image width.
        scale_factor: Factor by which to scale depth image.
        interpolation: Depth value interpolation for resizing.

    Returns:
        Depth image torch tensor with shape [height, width].
    """
    assert (height is None) == (width is None)  # Neither or both
    if filepath.suffix == ".npy":
        image = np.load(filepath) * scale_factor
        assert (height is None) == (width is None)  # Neither or both
    else:
        image = cv2.imread(str(filepath.absolute()), cv2.IMREAD_ANYDEPTH)
        image = image.astype(np.float64) * scale_factor
    do_resize = height is not None and image.shape[:2] != (height, width)
    if do_resize:
        image = cv2.resize(image, (width, height), interpolation=interpolation)
    return torch.from_numpy(image[:, :])


def maybe_show_progress(iterable, description, length, show=False):
    if show:
        yield from tqdm(iterable, desc=description, total=length)
    else:
        yield from iterable


def make_lookup_table(
    keys: Tensor,
    values: Tensor,
    key_max: Optional[int] = None,
    missing_key_value: Number = torch.nan,
) -> Tensor:
    """
    Create a lookup table using keys and values tensors.

    This function creates a 1D tensor (lookup table) using keys and values.
    The length of the lookup table is determined by `key_max`. The `keys` tensor
    specifies the indices in the lookup table that will be populated with the corresponding
    values from the `values` tensor. Indices not present in `keys` will be filled with
    `missing_key_value`.

    Parameters:
    -----------
    keys : torch.Tensor
        1D tensor of long integers specifying the indices in the lookup table
        where values should be placed. Must have dtype of torch.long.
    values : torch.Tensor
        1D tensor containing the values to be placed in the lookup table.
        Must have the same length as `keys`.
    key_max : int, optional
        The maximum key value + 1, which determines the length of the lookup table.
        If None, it is set to the maximum value in `keys` + 1. Default is None.
    missing_key_value : Number, optional
        The value to fill in for missing keys in the lookup table. Default is NaN.

    Returns:
    --------
    keys_expanded : torch.Tensor
        The populated lookup table. The dtype will match that of `values`.

    Raises:
    -------
    AssertionError
        If the dtype of the `keys` is not torch.long.

    Example:
    --------
    >>> keys = torch.tensor([1, 3, 5], dtype=torch.long)
    >>> values = torch.tensor([10.0, 30.0, 50.0])
    >>> make_lookup_table(keys, values)
    tensor([nan, 10.0, nan, 30.0, nan, 50.0])
    """
    if key_max is None:
        key_max = keys.max().item() + 1
    assert (
        keys.dtype == torch.long
    ), f"keys must have dtype torch.long -- not {keys.dtype}"
    keys_expanded = torch.full(
        [key_max],
        fill_value=missing_key_value,
        device=values.device,
        dtype=values.dtype,
    )
    keys_expanded.scatter_(dim=0, index=keys, src=values)
    return keys_expanded


class PosedRGBDDataset(ABC):
    """
    PosedRGBDDataset: Abstract class for datasets that contain RGB-D images and camera poses.

    An abstract class that can be used as a base class for scene datasets. This contains code for looping over scene trajectories and loading images per frame and abstract methods for loading poses and intrinsics.
        The camera poses and intrinsics be loaded per frame within the loop or once per scene after the loop. This can be controlled by setting the read_pose_per_frame and read_intrinsics_per_frame flags.
        Additional per-frame (eg. instance segmentation) and global fields (eg. 3D BBOX annotations) can be added by overriding the get_other_per_frame_fields and get_other_global_fields methods.
        Optionally, one can override the get_preloop_prerequisites method to load additional data that is necessary for loading poses and intrinsics before looping over the frames.

    """

    DEPTH_SCALE_FACTOR = 0.001  # to MM
    modalities_enum = Modalities
    # Whether to read intrinsics per frame or all at once per dataset
    read_intrinsics_per_frame = False
    # Whether to read poses per frame or all at once per dataset
    read_pose_per_frame = True
    dataset_name = "posed_rgbd"

    def __init__(
        self,
        root_dir: Union[str, Path],
        split: str = "train",
        keep_only_scenes: Optional[List[str]] = None,
        keep_only_first_k_scenes: Optional[int] = None,
        skip_first_k_scenes: int = None,
        frame_skip: int = 1,
        height: Optional[int] = 480,
        width: Optional[int] = 640,
        modalities: Union[str, Tuple[modalities_enum]] = modalities_enum.ALL,
        show_load_progress: bool = False,
        n_classes: int = 20,
        load_only_first_k_frames: Optional[int] = None,
        skipnan: bool = True,
        skip_frame_loading_for_refexp: bool = False,
        randomize_frame_order: bool = False,
    ):
        """
        Initializes the dataset.

        Sets the root directory of the dataset and retrieves the list of scenes for the given split.

        Args:

        - root_dir: Path to the root directory of the dataset.
        - split: Split of the dataset to load. Can be "train", "val", or "test".
        - keep_only_scenes: List of scene names to keep. If None, all scenes are kept.
        - keep_only_first_k_scenes: Number of scenes to keep starting from the first scene.
        - skip_first_k_scenes: Number of scenes to skip from the beginning.
        - frame_skip: Number of frames to skip between consecutive frames.
        - height: Height of the images to load.
        - width: Width of the images to load.
        - modalities: Modalities to load. Can be a string "all" or a tuple of modalities from the Modalities enum.
        - show_load_progress: Whether to show a progress bar when loading the dataset.
        - n_classes: Number of classes in the dataset.
        - load_only_first_k_frames: Number of frames to load per scene. If None, all frames are loaded.
        - skipnan: Whether to skip frames with NaN poses.
        - skip_frame_loading_for_refexp: Whether to skip loading frames for RefExp training.
        - randomize_frame_order: Whether to randomize the order of frames.
        """
        self.n_classes = n_classes

        self.set_root_dir(root_dir, split=split)

        assert split in ["train", "val", "test"]
        self.split = split

        self.frame_skip = frame_skip

        # Modalities
        if modalities == self.modalities_enum.ALL:
            self.modalities = [
                getattr(self.modalities_enum, v)
                for v in self.modalities_enum.__members__
                if v != "ALL"
            ]
        else:
            self.modalities = modalities

        self.scene_list = self.retrieve_scene_list(self.split)

        if keep_only_scenes is not None:
            self.scene_list = [s for s in self.scene_list if s in keep_only_scenes]
        self.scene_list = natsorted(self.scene_list)
        logger.info(
            f"{self.__class__.__name__}: Keeping next {keep_only_first_k_scenes} scenes starting at idx {skip_first_k_scenes}"
        )
        self.scene_list = self.scene_list[skip_first_k_scenes:][
            :keep_only_first_k_scenes
        ]
        assert len(self.scene_list) > 0

        self.load_only_first_k_frames = load_only_first_k_frames
        self.show_load_progress = show_load_progress
        self.height = height
        self.width = width
        self.skipnan = skipnan
        self.skip_frame_loading_for_refexp = skip_frame_loading_for_refexp
        self.randomize_frame_order = randomize_frame_order

    def set_root_dir(self, root_dir: Union[str, Path], split: str = "train"):
        """
        Sets the root directory of the dataset.

        Sets class attribute root_dir to the given path

        Args:
        - root_dir: Path to the root directory of the dataset.
        - split: Split of the dataset to load. Can be "train", "val", or "test".
        """
        self.root_dir = Path(root_dir)

    def retrieve_scene_list(self, split: str = "train"):
        """
        Retrieves the list of scenes for the given split.

        Override this if the dataset has a different structure.

        Args:
        - split: Split of the dataset to load. Can be "train", "val", or "test".

        Returns:
        - List of scene names for the given split.
        """
        return os.listdir(self.root_dir)

    def find_data(self, scene_name: str) -> dict:
        """
        Reads the data for a given scene and returns a dictionary with paths to necessary files (e.g. images, poses, etc.)
        Called by __getitem__ to load the data for a given scene.
        The dictionary should have the following keys:
        - img_paths: List of paths to RGB images
        - depth_paths: List of paths to depth images
        - pose_paths: List of paths to poses
        - instrinsics_paths: List of paths to intrinsics
        - frame_indexes: List of frame indexes
        It can also have other keys that are specific to the dataset.
        """
        return {
            "img_paths": [],
            "depth_paths": [],
            "pose_paths": [],
            "instrinsics_paths": [] if self.intrinsics_per_frame else None,
            "frame_indexes": [],
        }

    def get_preloop_prerequisites(self, data: dict, scan_name: str) -> dict:
        """
        Load additional data that is necessary for loading poses, intrinsics or any other fields before looping over the frames.
        Note: All data loaded here is included in the data returned by __getitem__.

        Args:
        - data: Dictionary with paths to necessary files for the scene.
        - scan_name: Name of the scene.

        Returns:
        - Dictionary with additional data that is necessary for loading poses, intrinsics or any other fields.
        """
        return {}

    def __len__(self) -> int:
        """
        Returns the number of scenes in the dataset.
        """
        return len(self.scene_list)

    @abstractmethod
    def get_intrinsics(self, data: dict, indices: List[int], **kwargs) -> torch.Tensor:
        """
        Load intrinsics for the given indices.
        Note: if read_intrinsics_per_frame is True, this method is called for each frame. In this case, it can be implemented to work just for len(indices) == 1.

        Args:
        - data: Dictionary with paths to necessary files for the scene.
        - indices: List of indices of frames for which to load intrinsics.
        - kwargs: Additional data that is necessary for loading intrinsics.

        Returns:
        - Tensor with intrinsics for the given indices of shape (len(indices), 3, 3).
        """
        raise NotImplementedError

    @abstractmethod
    def get_poses(self, data: dict, indices: List[int], **kwargs) -> torch.Tensor:
        """
        Load poses for the given indices.
        Note: if read_pose_per_frame is True, this method is called for each frame. In this case, it can be implemented to work just for len(indices) == 1.

        Args:
        - data: Dictionary with paths to necessary files for the scene.
        - indices: List of indices of frames for which to load poses.
        - kwargs: Additional data that is necessary for loading poses.

        Returns:
        - Tensor with poses for the given indices of shape (len(indices), 4, 4).
        """
        raise NotImplementedError

    def get_other_per_frame_fields(self, data: dict, index: int) -> dict:
        """
        Load additional per-frame fields for the given index to be loaded within the loop over frames.
        Note: This method is called for each frame.

        Args:
        - data: Dictionary with paths to necessary files for the scene.
        - index: Index of the frame for which to load additional fields.

        Returns:
        - Dictionary with additional per-frame fields for the given index.
        """
        return {}

    def collate_other_per_frame_fields(self, data: List[dict]) -> dict:
        """
        Collate the list of per-frame fields into a single dictionary.
        Note: This method is called after the loop over frames.

        Args:
        - data: List of dictionaries with per-frame fields.

        Returns:
        - Dictionary with collated per-frame fields.
        """
        return {k: torch.stack([d[k] for d in data]) for k in data[0]}

    def get_other_global_fields(self, data: dict, indices: List[int], **kwargs) -> dict:
        """ """
        return {}

    def build_dict_without_frames(
        self, image_paths, scan_name, **other_fields
    ) -> Dict[str, Any]:
        return {
            **other_fields,
            "scan_name": scan_name,
            "observations": Observations(
                frame_history=CameraViewsData.create_empty_camera_view_data(
                    scene_id=f"{self.dataset_name}-{scan_name}",
                    view_id=image_paths,
                )
            ),
        }

    @staticmethod
    def get_sky_direction_in_frames(observations: Observations) -> str:
        """
        Gets the sky direction in the frames. Expected to be one of "UP", "DOWN", "LEFT", "RIGHT". If not overriden (example for ScanNet), returns "UP".

        """
        return "UP"

    def __getitem__(self, idx: Union[str, int]) -> Dict[str, Any]:
        """
        Loads the data for a given scene.
        Description:  This method loads the data for a given scene. It loops over the frames of the scene and loads the necessary data (e.g. images, poses, intrinsics) for each frame.

        Args:
        - idx: Index of the scene to load. Can be an integer or a string with the name of the scene.

        Returns:
        - Observations object with the loaded data.
        """

        # Get the scene name
        if isinstance(idx, str):
            idx = self.scene_list.index(idx)
        scan_name = self.scene_list[idx]

        # Find the paths at which the data is stored for the scene
        data = self.find_data(scan_name)

        poses, intrinsics, images, depths = [], [], [], []
        image_paths = []
        frame_idxs = []
        other_per_frame_fields = []
        data_idxs = []

        prereq_data = self.get_preloop_prerequisites(data, scan_name)
        prereq_data["scan_name"] = scan_name

        ordered_frame_idxs = list(range(len(data["img_paths"])))
        if self.randomize_frame_order:
            random.shuffle(ordered_frame_idxs)

        # Loop over the frames of the scene
        for i in maybe_show_progress(
            ordered_frame_idxs,
            description=f"Loading scene {scan_name}",
            length=len(data["img_paths"]),
            show=self.show_load_progress,
        ):
            # Skip frames if necessary
            if (
                self.load_only_first_k_frames is not None
                and len(data_idxs) >= self.load_only_first_k_frames
            ):
                break

            if self.skip_frame_loading_for_refexp:
                data_idxs.append(i)
                frame_idxs.append(data["frame_indexes"][i])
                image_paths.append(data["img_paths"][i])
                continue

            # Read poses, intrinsics, images, and depths
            if (
                self.modalities_enum.POSE in self.modalities
                and self.read_pose_per_frame
            ):
                pose = self.get_poses(data, [i], **prereq_data)[0]
                if self.skipnan and torch.any(torch.isnan(pose)):
                    continue
                poses.append(pose)

            if (
                self.modalities_enum.INTRINSICS in self.modalities
                and self.read_intrinsics_per_frame
            ):
                intrinsic = self.get_intrinsics(data, [i], **prereq_data)[0]
                intrinsics.append(intrinsic)

            if self.modalities_enum.RGB in self.modalities:
                image_paths.append(data["img_paths"][i])
                img = get_image_from_path(
                    data["img_paths"][i], height=self.height, width=self.width
                )
                images.append(img)

            if self.modalities_enum.DEPTH in self.modalities:
                depth = get_depth_image_from_path(
                    data["depth_paths"][i],
                    height=self.height,
                    width=self.width,
                    scale_factor=self.DEPTH_SCALE_FACTOR,
                )
                depths.append(depth)

            # Load other per-frame fields
            other_per_frame_fields.append(self.get_other_per_frame_fields(data, i))
            data_idxs.append(i)
            frame_idxs.append(data["frame_indexes"][i])

        # Load other global fields
        other_global_modalities = self.get_other_global_fields(
            data, data_idxs, **prereq_data
        )
        if self.skip_frame_loading_for_refexp:
            return self.build_dict_without_frames(
                image_paths, scan_name, **other_global_modalities
            )

        if self.modalities_enum.RGB in self.modalities:
            images = torch.stack(images)

        if self.modalities_enum.DEPTH in self.modalities:
            depths = torch.stack(depths).float()

        # Collate other per-frame fields
        other_per_frame_fields = self.collate_other_per_frame_fields(
            other_per_frame_fields
        )

        # If poses and intrinsics are not read per frame, load them here. Otherwise, they are stacked.
        if self.modalities_enum.POSE in self.modalities:
            if self.read_pose_per_frame:
                poses = torch.stack(poses).float()
            else:
                poses = self.get_poses(data, data_idxs, **prereq_data)

        if self.modalities_enum.INTRINSICS in self.modalities:
            if self.read_intrinsics_per_frame:
                intrinsics = torch.stack(intrinsics).float()
            else:
                intrinsics = self.get_intrinsics(data, data_idxs, **prereq_data)

        # Return as dict
        data_dict = {
            # Pose
            "cam_to_world": poses,
            "cam_K": intrinsics,
            # Frames
            "rgb": images,
            "depth_zbuffer": depths,
            "image_paths": image_paths,
            "scene_id": f"{self.dataset_name}-{scan_name}",
            "view_id": image_paths,
            # Scene metadata
            "scan_name": scan_name,
            **other_per_frame_fields,
            **other_global_modalities,
        }
        observations = Observations.from_dict(data_dict)
        sky_direction = self.get_sky_direction_in_frames(observations)
        num_rotations = {"UP": 0, "DOWN": 2, "LEFT": 3, "RIGHT": 1}
        observations.rotate_frames_90_degrees_clockwise_about_camera_z(
            images.shape[3], images.shape[2], k=num_rotations[sky_direction]
        )
        # Update rotated frames in data_dict
        data_dict["rgb"] = observations.frame_history.rgb
        data_dict["depth_zbuffer"] = observations.frame_history.depth_zbuffer
        data_dict["cam_to_world"] = observations.frame_history.cam_to_world
        data_dict["observations"] = observations
        return data_dict

    def get_scenes(self):
        return self.scene_list

    def subsample_scenes(self, scenes):
        scene_set = set(self.scene_list)
        new_scene_list = [s for s in scenes if s in scene_set]
        logger.info(
            f"{self.__class__.__name__}: Subsampled scenes from {len(self.scene_list)} to {len(new_scene_list)}"
        )
        self.scene_list = new_scene_list
        return self.scene_list
