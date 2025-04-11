import logging
from typing import Optional, Union

import numpy as np
import torch

from preprocessing.image_features.feature_image_generator_base import (
    AbstractFeatureImageGeneratorWithCache,
)
from preprocessing.image_features.image_feature_encoder import BaseImageFeatureEncoder
from preprocessing.types import SemanticFeatureImage

logger = logging.getLogger(__name__)


class EmbeddingFeatureImageGenerator(AbstractFeatureImageGeneratorWithCache):
    NO_MASK_IDX = -1

    def __init__(
        self,
        image_feature_encoder: BaseImageFeatureEncoder,
        device: Optional[str] = None,
        cache_path: str = "/large_experiments/cortex/shared/pixel_aligned_feature_cache",
    ) -> None:
        """
        Turns an image into pixel-aligned features

        Unlike MaskFeatureImageGenerator, this class does not use masks to generate features.
        """
        super().__init__(device=device, cache_path=cache_path)
        self.image_feature_encoder = image_feature_encoder

    @property
    def image_encoder_name(self):
        return type(self.image_feature_encoder).__name__

    def generate_img_features(
        self,
        img: np.ndarray,
        cache_key: Optional[Union[str, int]] = None,
        compressed: bool = False,
    ) -> torch.Tensor:
        """
        Generate concept fusion features.

        Args:
            img (Image): Original image.

        Returns:
            torch.Tensor: Concept fusion features.
        """

        # image_H and image_W are the height and width of the image respectively
        image_H, image_W = img.shape[-3], img.shape[-2]

        # Embedding will be of shape (1, feat_dim, H // patch_dim, W // patch_dim), where
        # patch_dim (x patch_dim) is the size of the patch used in the vision transformer (e.g. 14 for DINOv2-base),
        # and feat_dim is the dimension of the features (e.g., 768 for DINOv2-base).
        embedding = self.image_feature_encoder.encode_image(img)

        # segments are all zeros because we are not using masks (shape: (1, H // patch_dim, W // patch_dim))
        segments = torch.zeros(image_H, image_W, dtype=torch.int32)

        # # Embedding from (1, feat_dim, H // patch_dim, W // patch_dim) to (H // patch_dim, W // patch_dim, feat_dim)
        # embedding = embedding[0].permute(1, 2, 0).half()
        # Bilinear interpolation to upsample the embedding to the original image size
        # (1, feat_dim, H // patch_dim, W // patch_dim) -> (1, feat_dim, image_H, image_W)
        embedding = torch.nn.functional.interpolate(
            embedding, size=(image_H, image_W), mode="bilinear", align_corners=False
        )
        # Normalize the embedding such that features have unit norm
        embedding = torch.nn.functional.normalize(embedding, p=2, dim=-3)
        # Drop the batch dimension
        embedding = embedding[0]
        # Permute from (feat_dim, image_H, image_W) to (image_H, image_W, feat_dim)
        embedding = embedding.permute(1, 2, 0)

        # Cache the data
        cache_data = SemanticFeatureImage.from_tensor(
            segments, embedding, no_mask_id=self.NO_MASK_IDX
        )
        self.to_cache(cache_key, cache_data)
        if compressed:
            return cache_data
        return embedding

    def generate_features(
        self,
        image: torch.Tensor,
        frame_path: Optional[int] = None,
        compressed: bool = False,
    ):
        """
        Takes a float image as input computes the 2D encoder features and cache them.
        """
        if self.image_feature_encoder is None:
            return None
        cache_key = frame_path
        try:
            cache_data = self.from_cache(cache_key)
        except EOFError as e:
            logger.error(
                f"This cache is corrupted, please maunally delete :{self.cache_path}/{self.image_encoder_name}/{cache_key}.pth"
            )
            raise e

        if cache_data is None:
            logger.info(f"cache miss: {cache_key} in {self.cache_path}")

        if cache_data is not None:
            if compressed:
                return cache_data
            return cache_data.to_dino_features(cache_data)
        uint_img = (image.cpu().numpy() * 255).astype(np.uint8)

        outfeat = self.generate_img_features(uint_img, cache_key, compressed)
        return outfeat
