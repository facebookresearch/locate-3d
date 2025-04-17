# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.


import math
from logging import getLogger

import torch
import torch.nn as nn
from packaging import version
from pytorch3d.renderer.implicit.harmonic_embedding import HarmonicEmbedding

from models.point_transformer_v3 import PointTransformerV3
import spconv.pytorch as spconv
from functools import partial
from huggingface_hub import PyTorchModelHubMixin

logger = getLogger()


class Encoder3DJEPA(
    nn.Module,
    PyTorchModelHubMixin,
    license="cc-by-nc-4.0",
):
    """Wrapper to use PTv3 3D Transformers."""

    def __init__(
        self,
        input_feat_dim=512,
        embed_dim=768,
        rgb_proj_dim=None,
        num_rgb_harmonic_functions=16,
        ptv3_args=dict(),
        voxel_size=0.05,
    ):
        self.voxel_size = voxel_size
        self.input_feat_dim = input_feat_dim
        self.embed_dim = embed_dim
        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        super().__init__()
        self.zero_token = nn.Parameter(torch.zeros(input_feat_dim))

        self.rgb_harmonic_embed = HarmonicEmbedding(
            n_harmonic_functions=num_rgb_harmonic_functions
        )
        self.rgb_harmonic_norm = norm_layer(3 * 2 * num_rgb_harmonic_functions + 3)

        self.rgb_projector = nn.Sequential(
            nn.Linear(3 * 2 * num_rgb_harmonic_functions + 3, rgb_proj_dim),
            nn.GELU(),
            norm_layer(rgb_proj_dim),
        )

        self.feature_embed = nn.Linear(input_feat_dim + rgb_proj_dim, embed_dim)

        self.feat_norm = norm_layer(input_feat_dim)

        self.transformer_input_norm = norm_layer(embed_dim)

        self.point_transformer_v3 = PointTransformerV3(**ptv3_args)
        self.num_features = self.point_transformer_v3.out_dim

    def load_weights(self, filename):
        try:
            checkpoint = torch.load(filename, map_location=torch.device("cpu"))
        except Exception as e:
            logger.info(f"Encountered exception when loading checkpoint {e}")

        state_dict = {
            k[len("module.") :] if k.startswith("module.") else k: v
            for k, v in checkpoint["target_encoder"].items()
        }
        state_dict = {
            k[len("backbone.") :] if k.startswith("backbone.") else k: v
            for k, v in state_dict.items()
        }

        self.load_state_dict(state_dict)

    def forward(self, featurized_scene_dict):
        """
        :param x: list obs(featurized pointcloud)
        :param masks: indices of patch tokens to mask (remove)
        """
        batch_size = 1

        features = torch.cat(
            [
                featurized_scene_dict["features_clip"],
                featurized_scene_dict["features_dino"],
            ],
            dim=1,
        ).unsqueeze(0)

        zero_locs = features.abs().sum(dim=2) == 0
        zero_locs = zero_locs.unsqueeze(-1).repeat(1, 1, self.input_feat_dim)
        features = torch.where(zero_locs, self.zero_token, features)

        features = self.feat_norm(features)

        rgb = featurized_scene_dict["rgb"].unsqueeze(0) * 255
        rgb = self.rgb_harmonic_embed(rgb)
        rgb = self.rgb_harmonic_norm(rgb)
        rgb = self.rgb_projector(rgb)

        features = torch.cat([rgb, features], dim=-1)

        x = self.feature_embed(features)

        x = self.transformer_input_norm(x)

        data_dict = {
            "coord": featurized_scene_dict["points"],
            "feat": x.reshape(-1, self.embed_dim),
            "grid_size": self.voxel_size,
            "offset": (torch.tensor(range(batch_size), device=x.device) + 1)
            * x.shape[1],
        }

        out = self.point_transformer_v3(data_dict)

        return {
            "features": out.reshape(*x.shape[:2], -1).squeeze(),
            "points": featurized_scene_dict["points"],
        }
