# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.


"""
This is an example script to demonstrate parallelizing data preprocessing
by using SLURM array jobs. Users will need to adapt the script to their
specific use case and SLURM cluster configuration.
"""

import argparse
import os
import sys

# Path to the directory in which this script is located
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

import submitit
import torch
from omegaconf import OmegaConf


def jobarray_entrypoint(args, start_idx, end_idx):

    # These imports need to be inside the main function to avoid issues
    # with SLURM array jobs
    from preprocessing.pointcloud_featurizer import FeatureLifter3D
    from locate3d_data.locate3d_dataset import Locate3DDataset

    # Import the preprocessing function
    from preprocessing.run_preprocessing import preprocess_scenes

    # Run the preprocessing function
    preprocess_scenes(args, start_idx, end_idx)


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
        required=True,
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

    # Number of scenes to cache
    # This is a placeholder value. You should set this to the actual number of scenes you want to cache.
    NUM_SCENES = 10

    # Number of jobs that we'd like to run
    NUM_JOBS = 10

    # Number of scenes to cache per job
    SCENES_PER_JOB = int(NUM_SCENES // NUM_JOBS)

    # Time to request per job in HH:MM:SS format
    # This is a placeholder value. You should set this to the actual time you expect each job to take.
    TIME_PER_JOB = f"06:00:00"

    # Start and end indices for each job
    start_inds = list(range(0, NUM_SCENES, SCENES_PER_JOB))
    end_inds = start_inds[1:]
    end_inds[-1] = (
        NUM_SCENES  # last job caches remaining scenes if NUM_SCENES is not divisible by SCENES_PER_JOB
    )
    for start_idx, end_idx in zip(start_inds, end_inds):
        print(f"python run_preprocessing_slurm_array.py {start_idx} {end_idx}")
    print(f"Total number of jobs: {len(start_inds)}. (Suggested: {NUM_JOBS})")
    print(f"Time per job: {TIME_PER_JOB}")

    print(f"Submitting {len(start_inds)} jobs to cache {NUM_SCENES} scenes...")
    keypress = input("Press any key to continue or 'q' to cancel.")
    if keypress == "q":
        quit()

    # Generate filename for log (use timestamp)
    executor = submitit.AutoExecutor(folder=f"{args.logdir}")
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
    jobs = executor.map_array(
        jobarray_entrypoint,
        [args for _ in range(len(start_inds))],
        start_inds,
        end_inds,
    )
