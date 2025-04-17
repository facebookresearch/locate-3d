# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.


from models.locate_3d import Locate3D
from omegaconf import OmegaConf
from locate3d_data.locate3d_dataset import Locate3DDataset
from pytest import approx as ptapprox
import functools
import torch

cfg = OmegaConf.load("config/locate_3d.yaml")
model = Locate3D(cfg)
model.load_from_checkpoint(
    "/fsx-cortex-datacache/shared/locate-3d-weights/locate-3d-plus.pt"
)

approx = functools.partial(ptapprox, abs=5e-2)


def test_model_on_scannetpp_scene():
    dataset = Locate3DDataset(
        annotations_fpath="locate3d_data/dataset/val_scannetpp.json",
        return_featurized_pointcloud=True,
        scannet_data_dir="[scannet_dir]",
        scannetpp_data_dir="[scannetpp_dir]",
        arkitscenes_data_dir="[arkit_dir]",
    )

    data = dataset[1587]

    output = model.inference(
        data["featurized_sensor_pointcloud"], data["lang_data"]["text_caption"]
    )

    assert output[0]["tokens_assigned"] == [7, 8]
    assert output[0]["confidence"].item() == approx(0.9363)
    assert torch.allclose(
        output[0]["bbox"],
        torch.tensor([3.0771, 5.0932, 1.5517, 4.8081, 5.5494, 2.3286], device="cuda:0"),
        rtol=5e-2,
    )


def test_model_on_scannet_scene():
    dataset = Locate3DDataset(
        annotations_fpath="locate3d_data/dataset/val_scannet.json",
        return_featurized_pointcloud=True,
        scannet_data_dir="/fsx-cortex/shared/datasets/scannet_ac",
        scannetpp_data_dir="[scannetpp_dir]",
        arkitscenes_data_dir="[arkit_dir]",
    )

    data = dataset[0]

    output = model.inference(
        data["featurized_sensor_pointcloud"], data["lang_data"]["text_caption"]
    )

    assert output[0]["tokens_assigned"] == [5, 6]
    assert output[0]["confidence"].item() == approx(0.9448)
    assert torch.allclose(
        output[0]["bbox"],
        torch.tensor(
            [-0.7044, -2.4510, 0.8885, 1.1859, -2.3210, 1.8263], device="cuda:0"
        ),
        rtol=5e-2,
    )


def test_model_on_arkitscenes_scene():
    dataset = Locate3DDataset(
        annotations_fpath="locate3d_data/dataset/val_arkitscenes.json",
        return_featurized_pointcloud=True,
        scannet_data_dir="[scannet_dir]",
        scannetpp_data_dir="[scannetpp_dir]",
        arkitscenes_data_dir="[arkit_dir]",
    )

    data = dataset[277]

    ## TODO: fix arkit double precision
    data["featurized_sensor_pointcloud"] = {
        key: value.float()
        for key, value in data["featurized_sensor_pointcloud"].items()
    }

    output = model.inference(
        data["featurized_sensor_pointcloud"], data["lang_data"]["text_caption"]
    )

    assert output[0]["tokens_assigned"] == [10, 11]
    assert output[0]["confidence"].item() == approx(0.9705)
    assert torch.allclose(
        output[0]["bbox"],
        torch.tensor(
            [1.8250, 0.2321, -0.0556, 2.7526, 1.0743, 0.8824], device="cuda:0"
        ),
        rtol=1e-1,
        atol=5e-1,
    )
