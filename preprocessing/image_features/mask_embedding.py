import logging
from typing import List, Optional

import numpy as np
import torch
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

from preprocessing.image_features.feature_image_generator_base import AbstractFeatureImageGeneratorWithCache
from preprocessing.image_features.image_text_encoder import BaseImageTextEncoder
from preprocessing.type_utils import SemanticFeatureImage

logger = logging.getLogger(__name__)


class FeatureImageGenerator:
    def generate_features(
        self,
        image: torch.Tensor,
        frame_path: Optional[int] = None,
        compressed: bool = False,
    ):
        raise NotImplementedError


def get_sam_model(model_path: str, device: str, version="vit_t"):
    checkpoint_path = model_path
    model = sam_model_registry[version](checkpoint=checkpoint_path)
    model.to(device)
    return model


class MaskEmbeddingFeatureImageGenerator(AbstractFeatureImageGeneratorWithCache):
    NO_MASK_IDX = -1

    def __init__(
        self,
        mask_generator: SamAutomaticMaskGenerator,
        image_text_encoder: BaseImageTextEncoder = None,
        device: Optional[str] = None,
        cache_path: str = None,
    ) -> None:
        """
        Turns an image into pixel-aligned features
        Uses MaskCLIP
         - generate_features() : takes an image and returns a feature image using various masks and the provided 2D encoder
        """
        super().__init__(device=device, cache_path=cache_path)
        self.mask_generator = mask_generator
        self.image_text_encoder = image_text_encoder
        self.cosine_similarity = torch.nn.CosineSimilarity(dim=-1)

    @property
    def image_encoder_name(self):
        return type(self.image_text_encoder).__name__

    def generate_mask(self, img: np.ndarray):
        assert not ((img / 255) == 0).all() and not (img > 255).any()
        try:
            masks = self.mask_generator.generate(img)
        except IndexError:
            masks = []
        # remove masks with zero area
        masks = list(filter(lambda x: x["bbox"][2] * x["bbox"][3] != 0, masks))

        return masks

    def generate_global_features(
        self,
        img: np.ndarray,
    ):
        # CLIP features global
        global_feat = None
        with torch.cuda.amp.autocast():
            global_feat = self.image_text_encoder.encode_image(img)
            global_feat /= global_feat.norm(dim=-1, keepdim=True)

        global_feat = torch.nn.functional.normalize(
            global_feat, dim=-1
        )  # --> (1, 1024)
        global_feat = global_feat.half().to(self.device)

        if self.feat_dim is None:
            self.feat_dim = global_feat.shape[-1]

        return global_feat

    def generate_local_features(
        self,
        img: np.ndarray,
        masks: List[dict],
        global_feat: torch.Tensor,
        cache_key: Optional[str] = None,
        compressed: bool = False,
    ) -> torch.Tensor:
        """
        Generate concept fusion features.

        Args:
            img (Image): Original image.
            masks (list[dict]): List of segmentation masks.
            global_feat (torch.Tensor): CLIP features global.

        Returns:
            torch.Tensor: Concept fusion features.
        """
        load_image_height, load_image_width = img.shape[0], img.shape[1]
        # CLIP features per ROI
        feat_per_roi = []
        roi_nonzero_inds = []
        similarity_scores = []
        outfeat = torch.zeros(
            load_image_height,
            load_image_width,
            self.feat_dim,
            dtype=torch.half,
            device=self.device,
        )
        if len(masks) == 0:
            segments = (
                torch.ones(load_image_height, load_image_width, dtype=torch.int32)
                * self.NO_MASK_IDX
            )
            cat_features = torch.zeros(0, self.feat_dim)

            cache_data = SemanticFeatureImage.from_tensor(
                segments, cat_features, no_mask_id=self.NO_MASK_IDX
            )
            self.to_cache(cache_key, cache_data)
            if compressed:
                return cache_data
            return outfeat

        for mask in masks:
            _x, _y, _w, _h = tuple(mask["bbox"])  # xywh bounding box

            # make sure _x, _y, _w, _h are ints
            _x, _y, _w, _h = int(_x), int(_y), int(_w), int(_h)

            nonzero_inds = torch.argwhere(torch.from_numpy(mask["segmentation"]))

            # Note: Image is (H, W, 3). In SAM output, y coords are along height, x along width
            img_roi = img[_y : _y + _h, _x : _x + _w, :]

            roifeat = self.image_text_encoder.encode_image(img_roi)
            roifeat = torch.nn.functional.normalize(roifeat, dim=-1)
            feat_per_roi.append(roifeat)
            roi_nonzero_inds.append(nonzero_inds)
            _sim = self.cosine_similarity(global_feat, roifeat)
            similarity_scores.append(_sim)

        similarity_scores = torch.cat(similarity_scores)
        softmax_scores = torch.nn.functional.softmax(similarity_scores, dim=0)
        WITH_OVERLAPPING_MASKS = False
        if WITH_OVERLAPPING_MASKS:
            for maskidx in range(len(masks)):
                _weighted_feat = (
                    softmax_scores[maskidx] * global_feat
                    + (1 - softmax_scores[maskidx]) * feat_per_roi[maskidx]
                )
                _weighted_feat = torch.nn.functional.normalize(_weighted_feat, dim=-1)
                outfeat[
                    roi_nonzero_inds[maskidx][:, 0], roi_nonzero_inds[maskidx][:, 1]
                ] += (_weighted_feat[0].detach().half())
                outfeat[
                    roi_nonzero_inds[maskidx][:, 0], roi_nonzero_inds[maskidx][:, 1]
                ] = torch.nn.functional.normalize(
                    outfeat[
                        roi_nonzero_inds[maskidx][:, 0], roi_nonzero_inds[maskidx][:, 1]
                    ].float(),
                    dim=-1,
                ).half()
        else:
            features = []
            segments = (
                torch.ones(load_image_height, load_image_width, dtype=torch.int32)
                * self.NO_MASK_IDX
            )
            for maskidx in range(len(masks)):
                _weighted_feat = (
                    softmax_scores[maskidx] * global_feat
                    + (1 - softmax_scores[maskidx]) * feat_per_roi[maskidx]
                )
                _weighted_feat = torch.nn.functional.normalize(_weighted_feat, dim=-1)
                outfeat[
                    roi_nonzero_inds[maskidx][:, 0], roi_nonzero_inds[maskidx][:, 1]
                ] = (_weighted_feat[0].detach().half())
                features.append(_weighted_feat.detach().half())
                segments[
                    roi_nonzero_inds[maskidx][:, 0], roi_nonzero_inds[maskidx][:, 1]
                ] = maskidx
            cat_features = torch.cat(features, dim=0)
            cache_data = SemanticFeatureImage.from_tensor(
                segments, cat_features, no_mask_id=self.NO_MASK_IDX
            )
            self.to_cache(cache_key, cache_data)

        if compressed:
            return cache_data
        outfeat = outfeat.unsqueeze(
            0
        ).float()  # interpolate is not implemented for float yet in pytorch
        outfeat = outfeat.permute(0, 3, 1, 2)  # 1, H, W, feat_dim -> 1, feat_dim, H, W
        outfeat = torch.nn.functional.interpolate(
            outfeat, [load_image_height, load_image_width], mode="nearest"
        )
        outfeat = outfeat.permute(0, 2, 3, 1)  # 1, feat_dim, H, W --> 1, H, W, feat_dim
        outfeat = torch.nn.functional.normalize(outfeat, dim=-1)
        outfeat = outfeat[0].half()  # --> H, W, feat_dim
        return outfeat

    def generate_features(
        self,
        image: torch.Tensor,
        frame_path: Optional[int] = None,
        compressed: bool = False,
        resize_cache_to_image_shape: bool = False,
    ):
        """
        Takes a float image as input, extracts masks, computes the 2D encoder features on the masks
        and on the whole image and returns the new "image" where RGB is replaced with the encoder features.
        """
        if self.image_text_encoder is None:
            return None
        cache_key = frame_path
        try:
            assert len(image.shape) == 3 and image.shape[-1] in [1, 3, 4], image.shape
            cache_data = self.from_cache(cache_key)
        except EOFError as e:
            logger.error(
                f"This cache is corrupted, please maunally delete :{self.cache_path}/{self.image_encoder_name}/{cache_key}.pth"
            )
            raise e

        if cache_data is None:
            logger.info(f"cache miss: {cache_key} in {self.cache_path}")

        if cache_data is not None:
            if resize_cache_to_image_shape:
                cache_data.resize(
                    (image.shape[1], image.shape[0])
                )  # PIL W x H format, but Torch uses H x W: https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.resize
            else:
                if cache_data.semantic_mask.size != image.size:
                    logger.error(
                        f"not resizing and cache data shape {cache_data.semantic_mask.size} does not match image shape {image.size}"
                    )
                    # cache_data = None
            if compressed:
                return cache_data
            return self._cache_data_to_mask_features(cache_data)
        uint_img = (image.cpu().numpy() * 255).astype(np.uint8)
        masks = self.generate_mask(uint_img)

        # CLIP features global
        global_feat = self.generate_global_features(uint_img)

        # CLIP features per ROI
        outfeat = self.generate_local_features(
            uint_img, masks, global_feat, cache_key, compressed
        )
        return outfeat
        # return self.generate_mask_features(image, scene_id=scene_id, frame_number=frame_number)
