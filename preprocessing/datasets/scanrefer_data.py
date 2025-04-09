# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Adapted from https://github.com/daveredrum/ScanRefer
# ScanRefer is licensed under a
# Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported License.
from pathlib import Path
from typing import Union

import pandas as pd


class ScanReferDataConfig:
    json_dir: Union[Path, str] = Path("scanrefer")


def load_scanrefer_data(json_fpath: Union[Path, str]):
    df = pd.read_json(json_fpath)

    # Rename to have some columns the same format as referit3d
    df = df.rename(
        columns={
            "scene_id": "scan_id",
            "object_name": "instance_type",
            "token": "tokens",
            "object_id": "target_id",
            "description": "utterance",
        }
    )
    df["stimulus_id"] = (
        df["scan_id"]
        + "_"
        + df["target_id"].astype(str)
        + "_"
        + df["ann_id"].astype(str)
    )
    df["dataset"] = "scanrefer"
    # We could load annotated viewpoints, too.
    # ID key is "<scene_id>-<object_id>_<ann_id>"
    # However, the viewpoints have "position" (3,) and "rotation" (3,).
    # Is rotation euler angles? If so, opencv camera convention?
    # I'm not opening that can of worms until we need it.
    return df
