# Downloading and preprocessing datasets for Locate 3D

This codebase currently supports preprocessing and inference on pointclouds extracted from ScanNet, ScanNet++, and ARKitScenes datasets.

The script `preprocessing/run_preprocessing.py` is the main entrypoint. However, before running this script, ensure that your datasets and directories are set up as follows. Note that you need to have _at least one_ of these datasets downloaded and processed first.

## Preparing ScanNet dataset

1. Obtain the ScanNet dataset by navigating to [the ScanNet homepage](http://www.scan-net.org/ScanNet/) and filling out the terms of use agreement
2. Extract the frames (rgb, depth, poses) by following the [official ScanNet toolkit instructions](https://github.com/ScanNet/ScanNet)
3. Follow the [dataset preparation instructions from MMDetection3D](https://mmdetection3d.readthedocs.io/en/v0.15.0/datasets/scannet_det.html) to extract additional information such as segmentation files, axis-aligned matrices, bounding boxes 

## Preparing for ScanNet++ dataset

1. Obtain the ScanNet++ dataset by navigating to [the ScanNet++ homepage](https://kaldir.vc.in.tum.de/scannetpp/) and filling out the terms of use agreement
2. Extract the frames (rgb, depth, poses) by following the [official ScanNet++ toolkit instructions](https://github.com/scannetpp/scannetpp)

## Preparing the ARKitScenes dataset

For ARKitScenes, be sure to follow [instructions to download the 3dod data split](https://github.com/apple/ARKitScenes/blob/main/DATA.md).


## Download a checkpoint for SAM-H

We use the SAM-H model to extract masks and compute per-mask CLIP features. Download this model by running the following snippet.
```
cd preprocessing 
mkdir weights
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```


## Running the preprocessing script

With at least one of the above datasets downloaded and prepared, the preprocessing script can be run using
```
python preprocessing/run_preprocessing.py
```

The supported commandline arguments are
```
REQUIRED
--l3dd_annotations_fpath    {locate3d_data/dataset/all.json,locate3d_data/dataset/train_scannet.json,locate3d_data/dataset/val_arkitscenes.json,locate3d_data/dataset/train.json,locate3d_data/dataset/train_scannetpp.json,locate3d_data/dataset/val_scannet.json,locate3d_data/dataset/train_arkitscenes.json,locate3d_data/dataset/val.json,locate3d_data/dataset/val_scannetpp.json} (Locate 3D annotations)

--cache_path CACHE_PATH (directory to store the cached pointclouds in)

OPTIONAL

[--scannet_data_dir SCANNET_DATA_DIR]  (path to scannet directory containing posed images)
[--scannetpp_data_dir SCANNETPP_DATA_DIR] (path to scannet++ directory containing posed images)
[--arkitscenes_data_dir ARKITSCENES_DATA_DIR] (path to arkitscenes directory containing posed images)

[--start START] (index of first scene to be processed; 0 by default)
[--end END] (index of last scene to be processed; -1 by default)
```

## Parallelizing the preprocessing script

For users interested in parallelizing the preprocessing using SLURM array jobs, look at `run_preprocessing_slurm_array.py`
