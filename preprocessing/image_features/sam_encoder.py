# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.


import os
from typing import Optional

import numpy as np
import torch
from segment_anything import SamPredictor, sam_model_registry

from preprocessing.image_features.image_feature_encoder import BaseImageFeatureEncoder


class SAMEncoder(BaseImageFeatureEncoder):
    """Encode image to SAM features (before decoding to masks using prompts)"""

    def __init__(self, version="vit_l", device: Optional[str] = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda":
            device = f"cuda:{torch.cuda.current_device()}"
        self.device = device
        self.version = version
        if version == "vit_h":
            sam = sam_model_registry[version](
                checkpoint=os.path.join(
                    os.environ["ACCEL_CORTEX_PATH"],
                    "cortex/utils/image",
                    "sam_vit_h_4b8939.pth",
                )
            )
        elif version == "vit_l":
            sam = sam_model_registry[version](
                checkpoint=os.path.join(
                    os.environ["ACCEL_CORTEX_PATH"],
                    "cortex/utils/image",
                    "sam_vit_l_0b3195.pth",
                )
            )
        elif version == "vit_b":
            sam = sam_model_registry[version](
                checkpoint=os.path.join(
                    os.environ["ACCEL_CORTEX_PATH"],
                    "cortex/utils/image",
                    "sam_vit_b_01ec64.pth",
                )
            )
        else:
            raise ValueError(f"Unknown version {version}")
        self.model = SamPredictor(sam.to(self.device))

    @torch.no_grad()
    def encode_image(self, image: np.ndarray):
        """Encode this input image to SAM features"""
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()

        self.model.set_image(image, image_format="RGB")
        features = self.model.get_image_embedding()
        self.model.reset_image()
        return features.float()

    def get_feature_dim(self):
        return 256
