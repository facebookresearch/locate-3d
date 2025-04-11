# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
import dataclasses
import logging
import os
import warnings
from collections import namedtuple
from enum import Enum, auto
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from natsort import natsorted
from PIL import Image
from torch import Tensor
from torch_scatter import scatter_mean

from preprocessing.datasets.base_dataset import BaseDataset
from preprocessing.datasets.posed_rgbd_dataset import PosedRGBDDataset, make_lookup_table
from preprocessing.datasets.referit3d_data import ReferIt3dDataConfig, load_referit3d_data
from preprocessing.datasets.scannet_constants import (
    MAX_CLASS_IDX,
    SCANNET_DATASET_CLASS_IDS,
    SCANNET_DATASET_CLASS_LABELS,
    SCANNET_DATASET_COLOR_MAPS,
)
from preprocessing.datasets.scanrefer_data import ScanReferDataConfig, load_scanrefer_data
from preprocessing.types import TrainingSample

logger = logging.getLogger(__name__)

GTPtcData = namedtuple(
    "GTPtcData",
    [
        "points",
        "point_rgbs",
        "point_labels",
        "point_normals",
        "point_segment_ids",
        "segment_centers",
    ],
)


class ScanNetModalities(Enum):
    RGB = auto()
    DEPTH = auto()
    POSE = auto()
    INTRINSICS = auto()
    INSTANCE_2D = auto()
    BBOX_3D = auto()
    GT_PTC = auto()
    ALL = auto()
    # BBOX_3D = auto()


class ScanNetDataset(PosedRGBDDataset):
    """
    This class loads scannet scenes through the __getitem__ function.
    Which modalities are loaded (RGB, depth, bounding boxes, etc) can be set through
    the constructor using the ScanNetModalities enum.
    """

    DEPTH_SCALE_FACTOR = 0.001  # to MM
    DEFAULT_HEIGHT = 968.0
    DEFAULT_WIDTH = 1296.0
    modalities_enum = ScanNetModalities
    read_intrinsics_per_frame = False
    read_pose_per_frame = True
    dataset_name = "scannet"

    def __init__(
        self,
        root_dir: Union[str, Path],
        split: str = "train",
        modalities: Union[str, Tuple[ScanNetModalities]] = ScanNetModalities.ALL,
        n_classes: int = 20,
        referit3d_config: Optional[ReferIt3dDataConfig] = None,
        scanrefer_config: Optional[ScanReferDataConfig] = None,
        **kwargs,
    ):
        """

        frame_skip: Only use every frame_skip frames


        The directory structure after pre-processing should be as below.
        For dowloading and preprocessing scripts, see https://mmdetection3d.readthedocs.io/en/v0.15.0/datasets/scannet_det.html

        scannet
        ├── meta_data
        ├── batch_load_scannet_data.py
        ├── load_scannet_data.py
        ├── scannet_utils.py
        ├── README.md
        ├── scans
        ├── scans_test
        ├── scannet_instance_data
        ├── scannet_2d_instance_data
        │   ├── scene_yyyy_yy
        │   │   ├── labels
        │   │   ├── labels-filt
        │   │   ├── instance
        │   │   ├── instance-filt
        ├── points
        │   ├── xxxxx.bin
        ├── instance_mask
        │   ├── xxxxx.bin
        ├── semantic_mask
        │   ├── xxxxx.bin
        ├── seg_info
        │   ├── train_label_weight.npy
        │   ├── train_resampled_scene_idxs.npy
        │   ├── val_label_weight.npy
        │   ├── val_resampled_scene_idxs.npy
        ├── posed_images
        │   ├── scenexxxx_xx
        │   │   ├── xxxxxx.txt # pose
        │   │   ├── xxxxxx.png # depth
        │   │   ├── xxxxxx.jpg # color
        │   │   ├── intrinsic.txt
        ├── referit3d
        │   ├── nr3d.csv
        │   ├── sr3d.csv
        │   ├── sr3d+.csv
        ├── scanrefer
        │   ├── ScanRefer_filtered_<SPLIT>.csv
        ├── scannet_infos_train.pkl
        ├── scannet_infos_val.pkl
        ├── scannet_infos_test.pkl

        """
        super().__init__(root_dir, split, modalities=modalities, **kwargs)
        assert (
            n_classes in SCANNET_DATASET_COLOR_MAPS
        ), f"{n_classes=} must be in {SCANNET_DATASET_COLOR_MAPS.keys()}"
        self.n_classes = n_classes

        # Set up directories and metadata
        self.posed_dir = self.root_dir / "posed_images"
        self.instance_dir = self.root_dir / "scannet_instance_data"
        self.instance_2d_dir = self.root_dir / "scannet_2d_instance_data"
        self.scan_dir = self.root_dir / "scannet_instance_data"
        self.pointcloud_dir = self.root_dir / f"gt_ptc_{split}"

        # Metainfo
        self.METAINFO = {
            "COLOR_MAP": SCANNET_DATASET_COLOR_MAPS[n_classes],
            "CLASS_NAMES": SCANNET_DATASET_CLASS_LABELS[n_classes],
            "CLASS_IDS": SCANNET_DATASET_CLASS_IDS[n_classes],
        }
        # Load class names
        labels_pd = pd.read_csv(
            self.root_dir / "meta_data" / "scannetv2-labels.combined.tsv",
            sep="\t",
            header=0,
        )
        labels_pd.loc[labels_pd.raw_category == "stick", ["category"]] = "object"
        labels_pd.loc[labels_pd.category == "wardrobe ", ["category"]] = "wardrobe"
        self.ALL_CLASS_IDS_TO_CLASS_NAMES = dict(
            zip(labels_pd["id"], labels_pd["category"])
        )
        self.ALL_CLASS_NAMES_TO_CLASS_IDS = dict(
            zip(labels_pd["category"], labels_pd["id"])
        )
        # self.METAINFO['CLASS_NAMES'] = [self.ALL_CLASS_IDS_TO_CLASS_NAMES[k] for k in self.METAINFO['CLASS_IDS']]
        self.METAINFO["CLASS_IDS"] = [
            self.ALL_CLASS_NAMES_TO_CLASS_IDS[k] for k in self.METAINFO["CLASS_NAMES"]
        ]
        # Create tensor lookup table
        self.class_ids_ten = torch.tensor(self.METAINFO["CLASS_IDS"])
        self.DROP_CLASS_VAL = -1
        self.class_ids_lookup = make_lookup_table(
            self.class_ids_ten,
            self.class_ids_ten,
            missing_key_value=self.DROP_CLASS_VAL,
            key_max=MAX_CLASS_IDX + 1,
        )

        # Referit3d
        self.referit_data: Optional[pd.DateFrame] = None
        if referit3d_config is not None:
            if split != "train":
                warnings.warn(
                    RuntimeWarning("ReferIt3D not evaluated on test set")
                )  # ASK: Why?
            r3d_config_copy = copy.deepcopy(referit3d_config)
            if not os.path.isabs(r3d_config_copy.nr3d_csv_fpath):
                r3d_config_copy.nr3d_csv_fpath = (
                    self.root_dir / r3d_config_copy.nr3d_csv_fpath
                )
            if r3d_config_copy.sr3d_csv_fpath is not None and not os.path.isabs(
                r3d_config_copy.sr3d_csv_fpath
            ):
                r3d_config_copy.sr3d_csv_fpath = (
                    self.root_dir / r3d_config_copy.sr3d_csv_fpath
                )
            self.referit_data = load_referit3d_data(
                scans_split={"train": self.scene_list},
                **dataclasses.asdict(r3d_config_copy),
            )

        # ScanRefer
        self.scanrefer_data: Optional[pd.DateFrame] = None
        if scanrefer_config is not None:
            json_fpath = (
                self.root_dir
                / scanrefer_config.json_dir
                / f"ScanRefer_filtered_{split}.json"
            )
            self.scanrefer_data = load_scanrefer_data(json_fpath)

    def retrieve_scene_list(self, split: str = "train"):
        # Create scene list
        with open(self.root_dir / "meta_data" / f"scannetv2_{split}.txt", "rb") as f:
            scene_list = [line.rstrip().decode() for line in f]
        return scene_list

    def find_data(self, scan_name: str):
        # RGBD + pose
        scene_pose_dir = self.posed_dir / scan_name
        scene_posed_files = [str(s) for s in scene_pose_dir.iterdir()]

        def get_endswith(f_list, endswith):
            return list(natsorted([s for s in f_list if s.endswith(endswith)]))

        frame_idxs = torch.arange(len(get_endswith(scene_posed_files, ".jpg")))[
            :: self.frame_skip
        ]

        # RGB
        img_names = get_endswith(scene_posed_files, ".jpg")[:: self.frame_skip]
        assert len(img_names) > 0, f"Found zero images for scene {scan_name}"

        # Depth
        depth_names = get_endswith(scene_posed_files, ".png")[:: self.frame_skip]
        assert len(depth_names) == len(
            img_names
        ), f"Unequal number of color and depth images for scene {scan_name} ({len(img_names)} != ({len(depth_names)}))"

        # Pose
        pose_names = list(
            natsorted(
                [
                    s
                    for s in scene_posed_files
                    if s.endswith(".txt") and not s.endswith("intrinsic.txt")
                ]
            )
        )[:: self.frame_skip]
        assert len(pose_names) == len(
            img_names
        ), f"Unequal number of color and poses for scene {scan_name} ({len(img_names)} != ({len(pose_names)}))"

        # 2D Instance
        inst2d_names = self.find_instance_2d(scan_name)[:: self.frame_skip]
        assert len(inst2d_names) == len(
            img_names
        ), f"Unequal number of color and poses for scene {scan_name} ({len(img_names)} != ({len(pose_names)}))"

        # img_names = list(
        #     natsorted([s for s in scene_posed_files if s.endswith(".jpg")])
        # )[:: self.frame_skip]
        # depth_names = list(
        #     natsorted([s for s in scene_posed_files if s.endswith(".png")])
        # )[:: self.frame_skip]

        # Camera Intrinsics
        intrinsic_name = self.posed_dir / scan_name / "intrinsic.txt"

        return {
            "img_paths": [scene_pose_dir / f for f in img_names],
            "depth_paths": [scene_pose_dir / f for f in depth_names],
            "pose_paths": [scene_pose_dir / f for f in pose_names],
            "intrinsic_path": intrinsic_name,
            "instance_2d_paths": inst2d_names,
            "bboxs_unaligned_path": self.instance_dir
            / f"{scan_name}_unaligned_bbox.npy",
            "bboxs_aligned_path": self.instance_dir / f"{scan_name}_aligned_bbox.npy",
            "gt_ptc_path": self.pointcloud_dir
            / f"{scan_name}.npy".replace("scene", ""),
            "axis_align_path": self.instance_dir / f"{scan_name}_axis_align_matrix.npy",
            "frame_indexes": frame_idxs,
        }

    def find_instance_2d(self, scan_name: str):
        file_list = list((self.instance_2d_dir / scan_name / "instance-filt").iterdir())
        return natsorted(file_list)

    def get_intrinsics(
        self, data: dict, indices: List[int], **kwargs
    ) -> List[torch.Tensor]:
        # Intrinsics shared across images
        K = torch.from_numpy(np.loadtxt(data["intrinsic_path"]).astype(np.float32))
        K[0] *= float(self.width) / self.DEFAULT_WIDTH  # scale_x
        K[1] *= float(self.height) / self.DEFAULT_HEIGHT  # scale_y
        K = K[:3, :3]
        intrinsics = torch.repeat_interleave(
            K.unsqueeze(0), repeats=len(indices), dim=0
        ).float()
        return intrinsics

    def get_other_global_fields(
        self,
        data: dict,
        indices: List[int],
        axis_align_mat: torch.Tensor,
        scan_name: str,
        **kwargs,
    ) -> dict:
        axis_align_mats = torch.repeat_interleave(
            axis_align_mat.unsqueeze(0), repeats=len(indices), dim=0
        ).float()

        # Load bounding boxes
        boxes_aligned, box_classes, box_obj_ids = None, None, None
        if ScanNetModalities.BBOX_3D in self.modalities:
            boxes_aligned, box_classes, box_obj_ids = load_3d_bboxes(
                data["bboxs_aligned_path"]
            )
            # keep_boxes = (box_classes.unsqueeze(1) == self.class_ids_ten.unsqueeze(0)).any(
            #     dim=1
            # )
            keep_boxes = (
                self.class_ids_lookup[box_classes.long()] != self.DROP_CLASS_VAL
            )
            boxes_aligned = boxes_aligned[keep_boxes]
            box_classes = box_classes[keep_boxes]
            box_obj_ids = box_obj_ids[keep_boxes]
            if len(boxes_aligned) == 0:
                raise RuntimeError(f"No GT boxes for scene {scan_name}")

        # GT ptc data
        gt_ptc_data = None
        if ScanNetModalities.GT_PTC in self.modalities:
            gt_ptc_data = load_gt_pointcloud_data(data["gt_ptc_path"], axis_align_mat)

        # Referring expressions
        column_names = [
            "utterance",
            "instance_type",
            "target_id",
            "stimulus_id",
            "dataset",
        ]
        ref_expr_df = pd.DataFrame(columns=column_names)

        # Referit
        if self.referit_data is not None:
            r3d_expr = self.referit_data[self.referit_data.scan_id == scan_name][
                column_names
            ]
            ref_expr_df = pd.concat([ref_expr_df, r3d_expr])

        # ScanRefer
        if self.scanrefer_data is not None:
            scanrefer_expr = self.scanrefer_data[
                self.scanrefer_data.scan_id == scan_name
            ][column_names]
            ref_expr_df = pd.concat([scanrefer_expr, r3d_expr])

        if len(ref_expr_df) > 0:
            ref_expr_df = filter_ref_exp_by_class(
                ref_expr_df, box_obj_ids, box_classes, drop_class=self.DROP_CLASS_VAL
            )

        # Return as dict
        return {
            # Pose
            "axis_align_mats": axis_align_mats,
            # 3d boxes
            "boxes_aligned": boxes_aligned,
            "box_classes": box_classes,
            "box_target_ids": box_obj_ids,
            # Referring expressions,
            "ref_expr": ref_expr_df,
            "gt_ptc_data": gt_ptc_data,
        }

    def get_preloop_prerequisites(self, data: dict, scan_name: str) -> dict:
        axis_align_mat = torch.from_numpy(np.load(data["axis_align_path"])).float()
        return {"axis_align_mat": axis_align_mat}

    def get_poses(
        self, data: dict, indices: List[int], axis_align_mat, **kwargs
    ) -> List[torch.Tensor]:
        # because self.read_pose_per_frame = True; can also implement this to support len(indices) > 1
        assert len(indices) == 1
        pose = np.loadtxt(data["pose_paths"][indices[0]])
        pose = np.array(pose).reshape(4, 4)
        # pose[:3, 1] *= -1
        # pose[:3, 2] *= -1
        pose = axis_align_mat @ torch.from_numpy(pose.astype(np.float32)).float()
        # add batch dimension and return
        return pose.unsqueeze(0)

    def get_other_per_frame_fields(self, data: dict, index: int) -> dict:
        if (
            ScanNetModalities.INSTANCE_2D in self.modalities
            or self.modalities == ScanNetModalities.ALL
        ):
            instance_2d = get_instance_image_from_path(
                data["instance_2d_paths"][index],
                height=self.height,
                width=self.width,
            )
        return {"instance_2d": instance_2d}

    def __len__(self):
        return len(self.scene_list)


##################################
# Load different modalities
#################################
def load_pose_opengl(path):
    pose = np.loadtxt(path)
    pose = np.array(pose).reshape(4, 4)
    pose[:3, 1] *= -1
    pose[:3, 2] *= -1
    pose = torch.from_numpy(pose).float()
    return pose


def load_cam_intrinsics(path):
    raise NotImplementedError


def load_semantic_masks(path):
    raise NotImplementedError


def load_instance_masks(path):
    raise NotImplementedError


def get_instance_image_from_path(
    image_path: Union[str, Path],
    height: Optional[int] = None,
    width: Optional[int] = None,
    resample=Image.NEAREST,
) -> torch.Tensor:
    """Returns a 1 channel image.
    # Adapted from https://github.com/nerfstudio-project/nerfstudio/blob/main/nerfstudio/data/dataparsers/scannet_dataparser.py
    Args:
        image_idx: The image index in the dataset.
    """
    pil_image = Image.open(image_path)

    assert (height is None) == (width is None)  # Neither or both
    if height is not None:
        pil_image = pil_image.resize((width, height), resample=resample)
    image = np.array(pil_image, dtype="uint8")  # shape is (h, w) or (h, w, 3 or 4)
    assert image.dtype == np.uint8
    image = torch.from_numpy(image)  # .byte()
    # image = image[:, :, :3] * image[:, :, -1:] + self._dataparser_outputs.alpha_color * (1.0 - image[:, :, -1:])
    return image


def load_3d_bboxes(path) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Loads {scene_id}_aligned_bbox.npy or {scene_id}_unaligned_bbox.npy

    Returns:
        bounds: FloatTensor of box bounds (xyz mins and maxes)
        label_id: IntTensor of class IDs
        obj_id: IntTensor of object IDs
    """
    bboxes = np.load(path)
    bbox_coords = torch.from_numpy(bboxes[:, :6]).float()
    labels = torch.from_numpy(bboxes[:, -2]).int()
    obj_ids = torch.from_numpy(bboxes[:, -1]).int()
    centers, lengths = bbox_coords[:, :3], bbox_coords[:, 3:6]
    mins = centers - lengths / 2.0
    maxs = centers + lengths / 2.0
    return torch.stack([mins, maxs], dim=-1), labels, obj_ids


def load_gt_pointcloud_data(filepath, axis_align_matrix) -> GTPtcData:
    points = np.load(filepath)
    coordinates, color, normal, segments, labels = (
        points[:, :3],
        points[:, 3:6],
        points[:, 6:9],
        points[:, 9],
        points[:, 10:12],
    )
    coordinates = align_to_axes(axis_align_matrix, torch.from_numpy(coordinates))[None]
    segments = torch.from_numpy(segments).long()
    segment_xyz = scatter_mean(coordinates.squeeze(), segments, dim=0)
    dataset_dict = {
        "points": coordinates,
        "point_rgbs": torch.from_numpy(color),
        "point_normals": torch.from_numpy(normal),
        "point_labels": torch.from_numpy(labels),
        "point_segment_ids": segments,
        "segment_centers": segment_xyz,
    }
    return GTPtcData(**dataset_dict)


def filter_ref_exp_by_class(
    ref_expr_df: pd.DataFrame,
    box_target_ids: Tensor,
    box_classes: Tensor,
    drop_class: int = -1,
) -> pd.DataFrame:
    """Keeps only referring expressions where referring expression"""
    ref_exp_target_ids = torch.from_numpy(
        ref_expr_df.target_id.to_numpy().astype(np.int64)
    )

    # Make lookuptable of lookuptable[target_ids] -> target_class
    max_key = max(box_target_ids.max(), ref_exp_target_ids.max()) + 1
    ids_to_classes = make_lookup_table(
        box_target_ids.long(), box_classes, missing_key_value=-1, key_max=max_key
    )

    # Keep referring expressions who have targets where class != -1 (i.e. where target is in box_target_ids)
    ref_exp_classes = ids_to_classes[ref_exp_target_ids]
    df = ref_expr_df.copy()
    df["target_class_id"] = ref_exp_classes.cpu().numpy()
    keep_exp = ref_exp_classes != drop_class

    # Bad targets by
    # breakpoint()
    # df['target_id'].loc[np.invert(keep_exp.cpu().numpy())]

    df = df.loc[keep_exp.cpu().numpy()]

    # # Map to class name with something like:
    # df['instance_type2'] = [class_id_to_name[class_idx] for class_idx in df['target_class_id']]
    return df


def align_to_axes(align_matrix: Tensor, point_cloud: Tensor):
    """
    Align the scan to xyz axes using its alignment matrix.
    inputs:
        point_cloud: N X 3
        align_matrix: 4 X 4
    """
    assert point_cloud.shape[-1] == 3, point_cloud.shape
    assert align_matrix.shape == (4, 4), align_matrix.shape

    point_cloud = torch.cat(
        [point_cloud, torch.ones_like(point_cloud[..., :1])], dim=-1
    )
    return torch.matmul(point_cloud, align_matrix.T)[..., :3]


class ScanNetTrainingSampleDataset(ScanNetDataset, BaseDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, idx: int) -> TrainingSample:
        """
        A dataset where each item is a dictionary with the following keys:
         - goal : None
         - action : None
         - Observations : The observation provided to the agent
        """
        raw_data = ScanNetDataset.__getitem__(self, idx)
        return TrainingSample(
            goal=None,
            observations=raw_data["observations"],
            action=None,
            info={"instance_2d": raw_data["instance_2d"]},
        )

    def get_scenes(self):
        return ScanNetDataset.get_scenes(self)

    def subsample_scenes(self, scenes):
        return ScanNetDataset.subsample_scenes(self, scenes)
