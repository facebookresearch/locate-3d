# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.


"""
Generate and cache pointclouds featurized with CLIP and DINO.
"""

import argparse
import os
import sys

# Path to the directory in which this script is located
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

import torch
from omegaconf import OmegaConf
from pathlib import Path

from preprocessing.pointcloud_featurizer import FeatureLifter3D
from locate3d_data.locate3d_dataset import Locate3DDataset


def preprocess_scenes(args, start_idx, end_idx):

    cache_path = args.cache_path

    # Create folders
    (Path(cache_path) / "ScanNet").mkdir(parents=True, exist_ok=True)
    (Path(cache_path) / "ScanNet++").mkdir(parents=True, exist_ok=True)
    (Path(cache_path) / "ARKitScenes").mkdir(parents=True, exist_ok=True)

    l3dd = Locate3DDataset(
        annotations_fpath=args.l3dd_annotations_fpath,
        return_featurized_pointcloud=False,
        scannet_data_dir=args.scannet_data_dir,
        scannetpp_data_dir=args.scannetpp_data_dir,
        arkitscenes_data_dir=args.arkitscenes_data_dir,
    )
    scene_list = sorted(l3dd.list_scenes())

    pointcloud_featurizer_clip_cfg = OmegaConf.load(
        os.path.join(SCRIPT_DIR, "config/clip.yaml")
    )
    pointcloud_featurizer_clip = FeatureLifter3D(pointcloud_featurizer_clip_cfg)
    pointcloud_featurizer_dino_cfg = OmegaConf.load(
        os.path.join(SCRIPT_DIR, "config/dino.yaml")
    )
    pointcloud_featurizer_dino = FeatureLifter3D(pointcloud_featurizer_dino_cfg)

    # Iterate through the dataset and cache the featurized pointclouds
    for idx in range(start_idx, end_idx):
        # Load a sample from the dataset
        scene_dataset = scene_list[idx][0]
        scene_id = scene_list[idx][1]
        frames_used = scene_list[idx][2]

        # Early skip if the scene is already cached
        if frames_used is None:
            cache_file = os.path.join(cache_path, scene_dataset, f"{scene_id}.pt")
        else:
            cache_file = os.path.join(
                cache_path,
                scene_dataset,
                f"{scene_id}_start{frames_used[0]}_end{frames_used[-1]}.pt",
            )
        if os.path.exists(cache_file):
            print(f"Cache file already exists: {cache_file}")
            print(f"Skipping cache creation for scene {scene_id}")
            continue

        print(f"Processing scene {scene_id} ...")
        camera_views = l3dd.get_camera_views(*scene_list[idx])

        # Build CLIP featurized pointcloud
        clip_pcd = pointcloud_featurizer_clip.lift_frames(camera_views)
        torch.manual_seed(0)  # seed the RNG so that the pointclouds are the same
        dino_pcd = pointcloud_featurizer_dino.lift_frames(camera_views)

        # Creating output dictionary
        output_dict = {
            "points": dino_pcd["points_reduced"],
            "rgb": dino_pcd["rgb_reduced"],
            "features_clip": clip_pcd["features_reduced"],
            "features_dino": dino_pcd["features_reduced"],
        }

        # Save the output dictionary to the cache file
        if not os.path.exists(cache_file):
            torch.save(output_dict, cache_file)
            print(f"Saved cache file: {cache_file}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--l3dd_annotations_fpath",
        type=str,
        help="File name of the Locate 3D Dataset to preprocess",
        choices=[
            "locate3d_data/dataset/all.json",
            "locate3d_data/dataset/train_scannet.json",
            "locate3d_data/dataset/val_arkitscenes.json",
            "locate3d_data/dataset/train.json",
            "locate3d_data/dataset/train_scannetpp.json",
            "locate3d_data/dataset/val_scannet.json",
            "locate3d_data/dataset/train_arkitscenes.json",
            "locate3d_data/dataset/val.json",
            "locate3d_data/dataset/val_scannetpp.json",
        ],
        required=True,
    )
    parser.add_argument(
        "--cache_path",
        type=str,
        help="Path to store preprocess cache data",
        default="cache",
    )
    parser.add_argument(
        "--scannet_data_dir",
        type=str,
        help="Path to the scannet dataset directory",
        default=None,
    )
    parser.add_argument(
        "--scannetpp_data_dir",
        type=str,
        help="Path to the scannet dataset directory",
        default=None,
    )
    parser.add_argument(
        "--arkitscenes_data_dir",
        type=str,
        help="Path to the scannet dataset directory",
        default=None,
    )
    parser.add_argument(
        "--start",
        type=int,
        help="Index of first scene to cache",
        default=0,
    )
    parser.add_argument(
        "--end",
        type=int,
        help="Index of last scene to cache",
        default=-1,
    )

    args = parser.parse_args()

    preprocess_scenes(args, start_idx=args.start, end_idx=args.end)
