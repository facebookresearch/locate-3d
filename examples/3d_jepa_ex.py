# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.


from locate3d_data.locate3d_dataset import Locate3DDataset
from models.encoder_3djepa import Encoder3DJEPA
from models.locate_3d import downsample

# Make sure the scenes you want to run are preprocessed and cache
# python -m preprocessing.run_preprocessing --l3dd_annotations_fpath locate3d_data/dataset/train_scannetpp.json --scannet_data_dir $SCANNET_DR --scannetpp_data_dir $SCANNETPP_DIR --end 5

# Set paths to data directories
dataset = Locate3DDataset(
    annotations_fpath="locate3d_data/dataset/val_scannetpp.json",
    return_featurized_pointcloud=True,
    scannet_data_dir="[scannet_data_dir]",
    scannetpp_data_dir="[scannetpp_data_dir]",
    arkitscenes_data_dir="[arkitscenes_data_dir]",
)

# Locate 3D model
model_3djepa = Encoder3DJEPA.from_pretrained("facebook/3d-jepa")

# Run model
data = dataset[219]

# Downsample pointcloud (optional)
data["featurized_sensor_pointcloud"] = downsample(
    data["featurized_sensor_pointcloud"], 30000
)

output = model_3djepa(data["featurized_sensor_pointcloud"])
