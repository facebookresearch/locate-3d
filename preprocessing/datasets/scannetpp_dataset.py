import json
import logging
import math
import os
import pickle
from pathlib import Path
from typing import Union

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import Compose, InterpolationMode, PILToTensor, Resize
from tqdm import tqdm

from preprocessing.datasets.base_dataset import BaseDataset
from preprocessing.type_utils import CameraViewsData, Observations, TrainingSample

INVALID_DEPTH_REPLACEMENT_VALUE = 0.0
IPHONE_SHAPE = (1440, 1920)

logger = logging.getLogger(__name__)


class ScannetppScene_Release:
    def __init__(self, scene_id, data_root=None):
        self._scene_id = scene_id
        self.data_root = Path(data_root)

    @property
    def scene_id(self):
        return self._scene_id

    # dir containing all data for this scene
    @property
    def scene_root_dir(self):
        return self.data_root / self._scene_id

    # ----scan assets
    @property
    def scans_dir(self):
        """
        dir containing all scan-related data
        """
        return self.data_root / self._scene_id / "scans"

    @property
    def pc_dir(self):
        """
        dir containing 1mm point cloud
        """
        return self.scans_dir

    @property
    def scan_pc_path(self):
        """
        path to point cloud
        """
        return self.pc_dir / "pc_aligned.ply"

    @property
    def scan_pc_mask_path(self):
        """
        path to the point cloud mask
        """
        return self.pc_dir / "pc_aligned_mask.txt"

    @property
    def scan_transformed_poses_path(self):
        """
        path containing all scanner poses transformed to aligned coordinates
        in a single file
        """
        return self.pc_dir / "scanner_poses.json"

    @property
    def mesh_dir(self):
        """
        dir containing all the meshes and related data
        put meshes in the same dir as 1mm PCs
        """
        return self.scans_dir

    @property
    def scan_mesh_path(self):
        """
        path to the mesh
        """
        return self.mesh_dir / "mesh_aligned_0.05.ply"

    @property
    def scan_mesh_mask_path(self):
        """
        path to the mesh mask
        """
        return self.mesh_dir / "mesh_aligned_0.05_mask.txt"

    @property
    def scan_mesh_segs_path(self):
        return self.mesh_dir / "segments.json"

    @property
    def scan_anno_json_path(self):
        return self.mesh_dir / "segments_anno.json"

    @property
    def scan_sem_mesh_path(self):
        return self.mesh_dir / "mesh_aligned_0.05_semantic.ply"

    # ----DSLR
    @property
    def dslr_dir(self):
        return self.data_root / self._scene_id / "dslr"

    @property
    def dslr_resized_dir(self):
        return self.dslr_dir / "resized_images"

    @property
    def dslr_resized_mask_dir(self):
        return self.dslr_dir / "resized_anon_masks"

    @property
    def dslr_original_dir(self):
        return self.dslr_dir / "original_images"

    @property
    def dslr_original_mask_dir(self):
        return self.dslr_dir / "original_anon_masks"

    @property
    def dslr_colmap_dir(self):
        return self.dslr_dir / "colmap"

    @property
    def dslr_nerfstudio_transform_path(self):
        return self.dslr_dir / "nerfstudio" / "transforms.json"

    @property
    def dslr_train_test_lists_path(self):
        return self.dslr_dir / "train_test_lists.json"

    # ----iphone
    @property
    def iphone_data_dir(self):
        return self.data_root / self._scene_id / "iphone"

    @property
    def iphone_video_path(self):
        return self.iphone_data_dir / "rgb.mp4"

    @property
    def iphone_rgb_dir(self):
        return self.iphone_data_dir / "rgb"

    @property
    def iphone_video_mask_path(self):
        return self.iphone_data_dir / "rgb_mask.mkv"

    @property
    def iphone_video_mask_dir(self):
        return self.iphone_data_dir / "rgb_masks"

    @property
    def iphone_depth_path(self):
        return self.iphone_data_dir / "depth.bin"

    @property
    def iphone_depth_dir(self):
        return self.iphone_data_dir / "depth"

    @property
    def iphone_pose_intrinsic_imu_path(self):
        return self.iphone_data_dir / "pose_intrinsic_imu.json"

    @property
    def iphone_colmap_dir(self):
        return self.iphone_data_dir / "colmap"

    @property
    def iphone_nerfstudio_transform_path(self):
        return self.iphone_data_dir / "nerfstudio" / "transforms.json"

    @property
    def iphone_exif_path(self):
        return self.iphone_data_dir / "exif.json"


class ScannetPPDataset(BaseDataset):
    def __init__(
        self,
        dataset_path: Union[str, Path] = None,
        cache_path: Union[str, Path] = None,
        frame_skip: int = 100,
        use_render=False,
        split=("train", "val"),
        splits=None,
        frame_chunk_size=None,
        image_scale=1.0 / 3.0,
    ):
        if splits != None:
            split = splits
        self.dataset_path = Path(dataset_path)
        self.cache_path = cache_path
        self.image_scale = image_scale
        new_shape = (
            round(IPHONE_SHAPE[0] * image_scale),
            round(IPHONE_SHAPE[1] * image_scale),
        )
        self.resize_scale = (
            new_shape[0] / IPHONE_SHAPE[0],
            new_shape[1] / IPHONE_SHAPE[1],
        )
        self.image_shape = new_shape
        self.frame_skip = frame_skip
        self.use_render = use_render
        self.frame_chunk_size = frame_chunk_size
        assert frame_chunk_size is None or (
            frame_chunk_size > 0
        ), f"frame_chunk_size must be > 0 but got {frame_chunk_size}"
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

        self.frame_skip = frame_skip

        splits_path = self.dataset_path / "splits"
        with open(splits_path / "nvs_sem_train.txt", "r") as f:
            self.train_ids = [line.strip() for line in f]
        with open(splits_path / "nvs_sem_val.txt", "r") as f:
            self.val_ids = [line.strip() for line in f]
        with open(splits_path / "nvs_test.txt", "r") as f:
            self.test_ids = [line.strip() for line in f]
        with open(splits_path / "sem_test.txt", "r") as f:
            self.test_ids = [line.strip() for line in f]

        self.test_ids = np.unique(self.test_ids).tolist()

        self.all_ids = []
        if "train" in split:
            self.all_ids += self.train_ids
        if "val" in split:
            self.all_ids += self.val_ids
        if "test" in split:
            self.all_ids += self.test_ids

        self.splits = split
        self.all_ids = np.unique(self.all_ids).tolist()
        self._calculate_scene_frame_counts()
        self._chunk_trajectories()

    def __len__(self):
        return len(self.dataset_idx_to_sample_idx_and_traj_chunk)

    def __getitem__(self, idx):
        sample_path_idx, traj_chunk_idx = self.dataset_idx_to_sample_idx_and_traj_chunk[
            idx
        ]
        # This is some class that comes with Scannet++ repo which makes it easier to navigate the directory for a single datapoint
        datum = ScannetppScene_Release(
            scene_id=self.all_ids[sample_path_idx], data_root=self.dataset_path / "data"
        )
        datum2 = ScannetppScene_Release(
            scene_id=self.all_ids[sample_path_idx], data_root=self.dataset_path
        )
        with open(datum.iphone_pose_intrinsic_imu_path, "r") as f:
            rtk_json = json.load(f)
        frame_names = list(rtk_json.keys())
        frame_names = self._subsample_frames(frame_names)

        frame_names = self._get_frame_chunk(frame_names, traj_chunk_idx)

        n_frames = len(frame_names)
        poses = torch.empty(size=(n_frames, 4, 4))
        intrinsics = torch.empty(size=(n_frames, 3, 3))

        rgbs = torch.empty(size=(n_frames, 3, *self.image_shape))
        depths = torch.empty(size=(n_frames, *self.image_shape))

        view_id = []
        for frame_idx, frame_name in enumerate(frame_names):
            if self.use_render:
                rgb_frame_path = (
                    datum2.iphone_data_dir / "render_rgb" / (frame_name + ".jpg")
                )
                rgb = Image.open(rgb_frame_path)
                depth = Image.open(
                    datum2.iphone_data_dir / "render_depth" / (frame_name + ".png")
                )
                mask = None
            else:
                rgb_frame_path = datum.iphone_rgb_dir / (frame_name + ".jpg")
                rgb = Image.open(rgb_frame_path)
                depth = Image.open(datum.iphone_depth_dir / (frame_name + ".png"))
                mask = Image.open(datum.iphone_video_mask_dir / (frame_name + ".png"))

            view_id.append(rgb_frame_path)
            frame = rtk_json[frame_name]
            poses[frame_idx] = torch.tensor(frame["aligned_pose"])
            intrinsics[frame_idx] = torch.tensor(frame["intrinsic"])

            if self.image_scale != 1.0:
                rgb = rgb.resize(self.image_shape, Image.BILINEAR)
                depth = depth.resize(self.image_shape, Image.NEAREST)
                mask = mask.resize(self.image_shape, Image.NEAREST)
                intrinsics[frame_idx][0, 0] *= self.resize_scale[1]
                intrinsics[frame_idx][1, 1] *= self.resize_scale[1]
                intrinsics[frame_idx][0, 2] *= self.resize_scale[0]
                intrinsics[frame_idx][1, 2] *= self.resize_scale[0]
                assert intrinsics[frame_idx][0, 1].abs() < 1e-3
                assert intrinsics[frame_idx][1, 0].abs() < 1e-3
                assert intrinsics[frame_idx][2, 0].abs() < 1e-3
                assert intrinsics[frame_idx][2, 1].abs() < 1e-3
                # print(intrinsics, self.resize_scale)
            rgbs[frame_idx] = self.image_to_tensor(rgb).float() / 255
            depths[frame_idx] = self.depth_to_tensor(depth).float() / 1000
            if mask is not None:
                depths[frame_idx][
                    self.depth_to_tensor(mask)[0] == 0
                ] = INVALID_DEPTH_REPLACEMENT_VALUE

        observations = Observations(
            frame_history=CameraViewsData(
                rgb=rgbs,
                depth_zbuffer=depths,
                cam_to_world=poses,
                cam_K=intrinsics,
                scene_id=f"{self.all_ids[sample_path_idx]}_chunksize_{self.frame_chunk_size}_chunk_{traj_chunk_idx}",
                view_id=view_id,
            )
        )

        return TrainingSample(
            observations=observations,
        )

    def _get_frame_chunk(self, frame_names, chunk_num, chunk_size=None):
        if chunk_size is None:
            chunk_size = self.frame_chunk_size

        if chunk_size is None:
            return frame_names

        return chunk_list(frame_names, chunk_size)[chunk_num]

    def _calculate_scene_frame_counts(self):
        assert not hasattr(
            self, "scene_frame_counts"
        ), "scene_frame_counts already calculated"
        sample_paths_cache_file = self.cache_path
        sample_paths_cache_file += f"_{self.splits}.pkl"
        if os.path.exists(sample_paths_cache_file):
            logger.warning(f"loading sample paths from {sample_paths_cache_file}")
            with open(sample_paths_cache_file, "rb") as f:
                _cache = pickle.load(f)
                self.scene_frame_counts = _cache["scene_frame_counts"]
                return

        self.scene_frame_counts = []
        for sample_path_idx in tqdm(self.all_ids):
            datum = ScannetppScene_Release(
                scene_id=sample_path_idx, data_root=self.dataset_path / "data"
            )
            with open(datum.iphone_pose_intrinsic_imu_path, "r") as f:
                rtk_json = json.load(f)
            frame_names = list(rtk_json.keys())
            self.scene_frame_counts.append(len(frame_names))

        with open(sample_paths_cache_file, "wb") as f:
            pickle.dump(
                {
                    "scene_frame_counts": self.scene_frame_counts,
                },
                f,
            )

    def get_scenes(self):
        return self.all_ids

    def subsample_scenes(self, scenes):
        scene_set = set(scenes)
        all_scene_set = set(self.all_ids)
        assert scene_set.issubset(
            all_scene_set
        ), f"scenes {scene_set - all_scene_set} not in dataset"
        ids_and_counts = [
            (scene, counts)
            for scene, counts in zip(self.all_ids, self.scene_frame_counts)
            if scene in scene_set
        ]
        assert (
            len(ids_and_counts) > 0
        ), f"{len(self.all_ids)}, {len(self.scene_frame_counts)}"
        assert (
            len(ids_and_counts) > 0
        ), f"{len(self.all_ids)}, {len(self.scene_frame_counts)}"
        self.all_ids, self.scene_frame_counts = zip(*sorted(ids_and_counts))
        self._chunk_trajectories()
        logger.info(
            f"{self.__class__.__name__}: Subsampled scenes from {len(all_scene_set)} to {len(scene_set)} ({len(self)} chunks)"
        )
        return self

    def _chunk_trajectories(self):
        # If needed, map the idx used in self[idx] -> (sample_path[sample_idx], traj_chunk_idx)
        self.sample_paths_to_n_traj_chunks = {}
        self.sample_paths_to_n_traj_frames = {}
        self.dataset_idx_to_sample_idx_and_traj_chunk = {}
        counter = 0
        for sample_idx, (sample_path, n_frames) in enumerate(
            zip(self.all_ids, self.scene_frame_counts)
        ):
            n_subsampled_frames = len(self._subsample_frames(list(range(n_frames))))
            n_chunks = (
                math.ceil(n_subsampled_frames / self.frame_chunk_size)
                if self.frame_chunk_size is not None
                else 1
            )
            self.sample_paths_to_n_traj_chunks[sample_path] = n_chunks
            self.sample_paths_to_n_traj_frames[sample_path] = n_subsampled_frames
            for i in range(n_chunks):
                if self.frame_chunk_size is None:
                    i = None
                self.dataset_idx_to_sample_idx_and_traj_chunk[counter] = (sample_idx, i)
                counter += 1
        logger.info(
            f"Finished chunking trajectories, {len(self.all_ids)} -> {len(self)} samples."
        )

    def _subsample_frames(self, _list):
        _list = _list[:: self.frame_skip]
        return _list


def chunk_list(lst, k):
    return [lst[i : i + k] for i in range(0, len(lst), k)]
