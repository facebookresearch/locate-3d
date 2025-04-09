import logging
import os
from pathlib import Path
from typing import Optional, Union

import torch

from ..types import SemanticFeatureImage

logger = logging.getLogger(__name__)


class AbstractFeatureImageGeneratorWithCache:
    NO_MASK_IDX = -1

    def __init__(
        self,
        device: Optional[str] = None,
        cache_path: str = "/large_experiments/cortex/shared/pixel_aligned_feature_cache",
    ) -> None:
        """
        The base class for generating feature images from images (and text).
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.cache_path = cache_path

        if os.path.exists(cache_path):
            self._chown_group_id = os.stat(cache_path).st_gid
        else:
            self._chown_group_id = None

        self.feat_dim = None

    @property
    def image_encoder_name(self):
        raise NotImplementedError("image_encoder_name property must be implemented")

    def cache_exists(self, cache_key: Union[str, int]) -> bool:
        pth_path = f"{self.cache_path}/{self.image_encoder_name}/{cache_key}.pth"
        cache_exists = os.path.exists(pth_path)
        if not cache_exists:
            logger.warn("No image embedding cache exists")
        return cache_exists

    def from_cache(self, cache_key: Optional[Union[str, int]]):
        if cache_key is None:
            return None
        pth_path = f"{self.cache_path}/{self.image_encoder_name}/{cache_key}.pth"
        old_pth_path = pth_path
        if not os.path.exists(pth_path):
            # If cannot find the `.pth` file, try to find the `.pkl` file (old version)
            pth_path = f"{self.cache_path}/{self.image_encoder_name}/{cache_key}.pkl"

        if os.path.exists(pth_path):
            with open(pth_path, "rb") as f:
                try:
                    return torch.load(f, map_location=self.device)
                except Exception as e:
                    # Various unrelated CUDA errors will cause random cache files to be deleted
                    logger.error(
                        f"Error loading from cache: {pth_path}, error raised was {e}"
                    )
                    if "CUDA error" in str(e):
                        raise e
                    try:
                        os.remove(pth_path)
                    except Exception:
                        pass
                return None
        else:
            logger.warn(f"{pth_path=} and {old_pth_path=} not found")
        return None

    def to_cache(
        self, cache_key: Union[str, int], semantic_features: SemanticFeatureImage
    ):
        if cache_key is None:
            return
        pth_path = f"{self.cache_path}/{self.image_encoder_name}/{cache_key}.pth"
        parent_path = Path(pth_path).parent
        must_give_ownership = not os.path.exists(parent_path)
        os.makedirs(parent_path, exist_ok=True)
        if must_give_ownership and self._chown_group_id is not None:
            try:
                os.chown(parent_path, -1, self._chown_group_id)
            except PermissionError:  # noqa: SIM105
                pass
        if not os.path.exists(pth_path):
            try:
                # in case of concurent writing
                with open(pth_path, "wb") as f:
                    torch.save(semantic_features, f)
            except Exception as e:
                raise e

    def clear(self):
        self.feat_dim = None

    def _cache_data_to_mask_features(self, cache_data: SemanticFeatureImage):
        return cache_data.to_mask_features(self.device)
