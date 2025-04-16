import math
from logging import getLogger

import torch
import torch.nn as nn
from packaging import version
from pytorch3d.renderer.implicit.harmonic_embedding import HarmonicEmbedding

from models.point_transformer_v3 import PointTransformerV3
import spconv.pytorch as spconv
logger = getLogger()


class Encoder3DJEPA(nn.Module):
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
        
        super().__init__()
        self.zero_token = nn.Parameter(torch.zeros(input_feat_dim))

        self.rgb_harmonic_embed = HarmonicEmbedding(
            n_harmonic_functions=num_rgb_harmonic_functions
        )
        self.rgb_harmonic_norm = nn.LayerNorm(
            3 * 2 * num_rgb_harmonic_functions + 3
        )

        self.rgb_projector = nn.Sequential(
            nn.Linear(3 * 2 * num_rgb_harmonic_functions + 3, rgb_proj_dim),
            nn.GELU(),
            nn.LayerNorm(rgb_proj_dim),
        )

        self.feature_embed = nn.Linear(input_feat_dim + rgb_proj_dim, embed_dim)

        self.feat_norm = nn.LayerNorm(input_feat_dim)

        self.transformer_input_norm = nn.LayerNorm(embed_dim)
        
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
        
    def forward(self, ptc):
        """
        :param x: list obs(featurized pointcloud)
        :param masks: indices of patch tokens to mask (remove)
        """


        features = ptc["features"]

        zero_locs = features.abs().sum(dim=2) == 0
        zero_locs = zero_locs.unsqueeze(-1).repeat(1, 1, self.input_feat_dim)
        features = torch.where(zero_locs, self.zero_token, features)

        features = self.feat_norm(features)

        rgb = ptc["rgb"] * 255
        rgb = self.rgb_harmonic_embed(rgb)
        rgb = self.rgb_harmonic_norm(rgb)
        rgb = self.rgb_projector(rgb)

        features = torch.cat([rgb, features], dim=-1)

        x = self.feature_embed(features)

        x = self.transformer_input_norm(x)

        data_dict = {
            "coord": ptc['points'].reshape(-1, 3),
            "feat": x.reshape(-1, self.embed_dim),
            "grid_size": self.voxel_size,
            "offset": (torch.tensor(range(x.shape[0]), device=x.device) + 1) * x.shape[1],
        }

        out = self.point_transformer_v3(data_dict)

        return out.reshape(*x.shape[:2], -1)
