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
from pytorch3d.ops import knn_points

from preprocessing.pointcloud_featurizer import FeatureLifter3D
from locate3d_data.locate3d_dataset import Locate3DDataset

def main(args, start_idx, end_idx):

    cache_path = args.cache_path

    l3dd = Locate3DDataset(annotations_fpath = 'locate3d_data/dataset/train_scannet.json', scannet_data_dir = '/fsx-cortex/shared/datasets/scannet_ac')
    scene_list = l3dd.list_scenes().sorted()
    
    pointcloud_featurizer_clip_cfg = OmegaConf.load("config/clip.yaml")
    pointcloud_featurizer_clip = FeatureLifter3D(pointcloud_featurizer_clip_cfg)
    pointcloud_featurizer_dino_cfg = OmegaConf.load("config/dino.yaml")
    pointcloud_featurizer_dino = FeatureLifter3D(pointcloud_featurizer_dino_cfg)

    # Iterate through the dataset and cache the featurized pointclouds
    for idx in range(start_idx, end_idx):
        # Load a sample from the dataset
        scene_id = scene_list[idx][1]
        camera_views = l3ddd.get_camera_views(*scene_list[idx])
        
        # Early skip if the scene is already cached
        cache_file = os.path.join(cache_path, f"{scene_id}.pt")
        if os.path.exists(cache_file):
            print(f"Cache file already exists: {cache_file}")
            print("Skipping cache creation.")
            continue
        
        # Build CLIP featurized pointcloud
        clip_ptc = pointcloud_featurizer_clip.lift_frames(camera_views)
        dino_ptc = pointcloud_featurizer_dino.lift_frames(camera_views)

        # Aligning CLIP and DINO features
        _knn = knn_points(obs_dino.pointcloud.points_reduced[None, ...].cuda(), obs_clip.pointcloud.points_reduced[None, ...].cuda(), K=1)
        new_points = obs_clip.pointcloud.points_reduced[_knn.idx[0]].squeeze(1)
        new_rgb = obs_clip.pointcloud.rgb_reduced[_knn.idx[0]].squeeze(1)
        new_clip_features = obs_clip.pointcloud.features_reduced[_knn.idx[0]].squeeze(1)
        new_dino_features = obs_dino.pointcloud.features_reduced

        # Creating output dictionary
        output_dict = {
            "points": new_points,
            "rgb": new_rgb,
            "features_clip": new_clip_features,
            "features_dino": new_dino_features,
        }

        # Save the output dictionary to the cache file
        if not os.path.exists(cache_file):
            torch.save(output_dict, cache_file)
            print(f"Saved cache file: {cache_file}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_name",
        type=str,
        help="Name of the cortex dataset class",
        choices=["ScannetPPDataset", "ARKitScenesDataset", "ScanNetDataset"],
        required=True,
    )
    parser.add_argument(
        "--dataset_split",
        type=str,
        help="Dataset split to cache",
        choices=["train", "val", "test"],
        required=True,
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        help="Path to the dataset directory",
        required=True,
    )
    parser.add_argument(
        "--cache_path",
        type=str,
        help="Path to store the cached dataset",
        required=True,
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
    parser.add_argument(
        "--voxel_size_cm",
        type=int,
        help="Voxel size in centimeters",
        default=5,
    )
    parser.add_argument(
        "--frameskip_fps",
        type=int,
        help="Frameskip in frames per second",
        default=30,
    )

    args = parser.parse_args()

    main(args, start_idx=args.start, end_idx=args.end)
