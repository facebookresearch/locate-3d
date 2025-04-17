"""
This is an example script to demonstrate parallelizing data preprocessing
by using SLURM array jobs.
"""

import argparse
import os
import sys
# Path to the directory in which this script is located
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

import submitit
import torch
from omegaconf import OmegaConf


def main(args, start_idx, end_idx):

    from preprocessing.pointcloud_featurizer import FeatureLifter3D
    from locate3d_data.locate3d_dataset import Locate3DDataset

    cache_path = args.cache_path

    l3dd = Locate3DDataset(
        annotations_fpath = args.l3dd_annotations_fpath,
        scannet_data_dir = args.scannet_data_dir,
        scannetpp_data_dir = args.scannetpp_data_dir,
        arkitscenes_data_dir = args.arkitscenes_data_dir,

    )
    scene_list = sorted(l3dd.list_scenes())
    
    pointcloud_featurizer_clip_cfg = OmegaConf.load(os.path.join(SCRIPT_DIR, "config/clip.yaml"))
    pointcloud_featurizer_clip = FeatureLifter3D(pointcloud_featurizer_clip_cfg)
    pointcloud_featurizer_dino_cfg = OmegaConf.load(os.path.join(SCRIPT_DIR, "config/dino.yaml"))
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
            cache_file = os.path.join(cache_path, scene_dataset, f"{scene_id}_start{frames_used[0]}_end{frames_used[-1]}.pt")
        if os.path.exists(cache_file):
            print(f"Cache file already exists: {cache_file}")
            print(f"Skipping cache creation for scene {scene_id}")
            continue

        print(f"Processing scene {scene_id} ...")
        camera_views = l3dd.get_camera_views(*scene_list[idx])

        # Build CLIP featurized pointcloud
        clip_pcd = pointcloud_featurizer_clip.lift_frames(camera_views)
        torch.manual_seed(0) # seed the RNG so that the pointclouds are the same
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
            'locate3d_data/dataset/all.json',
            'locate3d_data/dataset/train_scannet.json',
            'locate3d_data/dataset/val_arkitscenes.json',
            'locate3d_data/dataset/train.json',
            'locate3d_data/dataset/train_scannetpp.json',
            'locate3d_data/dataset/val_scannet.json',
            'locate3d_data/dataset/train_arkitscenes.json',
            'locate3d_data/dataset/val.json',
            'locate3d_data/dataset/val_scannetpp.json',
        ],
        required=True,
    )
    parser.add_argument(
        "--cache_path",
        type=str,
        help="Path to store preprocess cache data",
        default='cache',
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
    parser.add_argument(
        "--slurm_account",
        type=str,
        help="SLURM account to use",
        default="account",
    )
    parser.add_argument(
        "--slurm_qos",
        type=str,
        help="SLURM quality of service level to use",
        default="normal",
    )
    parser.add_argument(
        "--logdir",
        type=str,
        help="Directory to store SLURM logs",
        default="logs",
    )

    args = parser.parse_args()

    NUM_SCENES = 0
    NUM_SNPP_TRAIN = 6321
    NUM_SNPP_VAL = 1341
    NUM_SN_TRAIN = 1201
    NUM_SN_VAL = 314
    NUM_ARKIT_TRAIN = 4498
    NUM_ARKIT_VAL = 549

    if args.l3dd_annotations_fpath == 'locate3d_data/dataset/all.json':
        NUM_SCENES = NUM_SNPP_TRAIN + NUM_SNPP_VAL + NUM_ARKIT_TRAIN + NUM_ARKIT_VAL
    elif args.l3dd_annotations_fpath == 'locate3d_data/dataset/train_scannet.json':
        NUM_SCENES = NUM_SN_TRAIN
    elif args.l3dd_annotations_fpath == 'locate3d_data/dataset/val_arkitscenes.json':
        NUM_SCENES = NUM_ARKIT_VAL
    elif args.l3dd_annotations_fpath == 'locate3d_data/dataset/train.json':
        NUM_SCENES = NUM_SN_TRAIN + NUM_ARKIT_TRAIN
    elif args.l3dd_annotations_fpath == 'locate3d_data/dataset/train_scannetpp.json':
        NUM_SCENES = NUM_SNPP_TRAIN
    elif args.l3dd_annotations_fpath == 'locate3d_data/dataset/val_scannet.json':
        NUM_SCENES = NUM_SN_VAL
    elif args.l3dd_annotations_fpath == 'locate3d_data/dataset/train_arkitscenes.json':
        NUM_SCENES = NUM_ARKIT_TRAIN
    elif args.l3dd_annotations_fpath == 'locate3d_data/dataset/val.json':
        NUM_SCENES = NUM_SN_VAL + NUM_ARKIT_VAL
    elif args.l3dd_annotations_fpath == 'locate3d_data/dataset/val_scannetpp.json':
        NUM_SCENES = NUM_SNPP_VAL
    else:
        raise ValueError("Invalid annotations file path")
    
    # Time in minutes we have by when we need all scenes cached
    TOTAL_TIME_MINUTES = 6 * 60
    # Time to cache one scene in minutes (20-30 minutes is a reasonable estimate if building everything from scratch)
    TIME_PER_SCENE_MINUTES = 20

    # Number of jobs that we need to run in parallel to cache all scenes in time
    NUM_JOBS = int(NUM_SCENES * TIME_PER_SCENE_MINUTES / TOTAL_TIME_MINUTES)

    # Number of scenes to cache per job
    SCENES_PER_JOB = int(NUM_SCENES // NUM_JOBS)

    # Time to request per job in HH:MM:SS format
    TIME_PER_JOB = f"{int(math.ceil(SCENES_PER_JOB * TIME_PER_SCENE_MINUTES / 60))}:00:00"

    # Start and end indices for each job
    start_inds = list(range(0, NUM_SCENES, SCENES_PER_JOB))
    end_inds = start_inds[1:]
    end_inds[-1] = NUM_SCENES  # last job caches remaining scenes if NUM_SCENES is not divisible by SCENES_PER_JOB
    for start_idx, end_idx in zip(start_inds, end_inds):
        print(f"python run_preprocessing_slurm_array.py {start_idx} {end_idx}")
    print(f"Total number of jobs: {len(start_inds)}. (Suggested: {NUM_JOBS})")
    print(f"Time per job: {TIME_PER_JOB}")

    print(f"Submitting {len(start_inds)} jobs to cache {NUM_SCENES} {args.dataset_} scenes...")
    keypress = input("Press any key to continue or 'q' to cancel.")
    if keypress == "q":
        quit()
    
    # Generate filename for log (use timestamp)
    executor = submitit.AutoExecutor(
        folder=f"{args.logdir}"
    )
    executor.update_parameters(
        slurm_qos=args.slurm_qos,
        slurm_account=args.slurm_account,
        slurm_time=TIME_PER_JOB,
        slurm_gres="gpu:1",
        slurm_mem="64G",
        nodes=1,
        tasks_per_node=1,
        cpus_per_task=12,
        slurm_array_parallelism=len(start_inds),
    )
    jobs = executor.map_array(main, [args for _ in range(len(start_inds))], start_inds, end_inds)
