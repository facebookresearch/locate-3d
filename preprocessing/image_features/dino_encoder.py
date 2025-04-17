# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.


from typing import Optional

import numpy as np
import torch

from preprocessing.image_features.image_feature_utils import padding_to_patch


class DINOV2Encoder:
    """Encode image to DINO v2 feature"""

    def __init__(self, version="base", device: Optional[str] = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda":
            device = f"cuda:{torch.cuda.current_device()}"
        self.device = device
        self.version = version

        backbone_archs = {
            "small": "vits14_reg",
            "base": "vitb14_reg",
            "large": "vitl14_reg",
            "giant": "vitg14_reg",
        }

        if version not in backbone_archs.keys():
            raise ValueError(f"Unknown version {version}")

        backbone_arch = backbone_archs[version]
        backbone_name = f"dinov2_{backbone_arch}"
        backbone_model = torch.hub.load(
            repo_or_dir="facebookresearch/dinov2", model=backbone_name
        ).to(self.device)

        self.model = backbone_model.eval()
        # NOTE: use imagenet mean and std. Might need double check
        self.mean = torch.tensor([0.485, 0.456, 0.406]).to(self.device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).to(self.device).view(1, 3, 1, 1)

    @torch.no_grad()
    def encode_image(self, image: np.ndarray):
        """Encode this input image to SAM features"""
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).to(self.device)
        elif isinstance(image, torch.Tensor):
            image = image.to(self.device)
        else:
            raise ValueError(f"Unknown image type {type(image)}")

        image = image.float()
        # NOTE: Not sure do we need to resize the position embeddings
        if image.ndim == 3:
            image = image.unsqueeze(0)

        if image.shape[-1] == 3:
            image = image.permute(0, 3, 1, 2)

        image = padding_to_patch(image, patch_size=14)

        patch_H = image.shape[-2] // 14
        patch_W = image.shape[-1] // 14

        image = (image - self.mean) / self.std
        output = self.model.forward_features(image)["x_norm_patchtokens"]
        output = output.view(output.shape[0], -1, patch_H, patch_W)
        return output

    def get_feature_dim(self):
        if self.version == "small":
            return 384
        elif self.version == "base":
            return 768
        elif self.version == "large":
            return 1024
        elif self.version == "giant":
            return 1536
        else:
            raise ValueError(f"Unknown version {self.version}")
