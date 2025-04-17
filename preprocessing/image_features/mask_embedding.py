# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.


import logging
from typing import List, Optional

import numpy as np
import torch
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

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


class MaskEmbeddingFeatureImageGenerator:
    NO_MASK_IDX = -1

    def __init__(
        self,
        mask_generator: SamAutomaticMaskGenerator,
        image_text_encoder=None,
        device: Optional[str] = None,
    ) -> None:
        """
        Turns an image into pixel-aligned features
        Uses MaskCLIP
         - generate_features() : takes an image and returns a feature image using various masks and the provided 2D encoder
        """

        self.mask_generator = mask_generator
        self.image_text_encoder = image_text_encoder
        self.cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.feat_dim = None

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
    ):
        """
        Takes a float image as input, extracts masks, computes the 2D encoder features on the masks
        and on the whole image and returns the new "image" where RGB is replaced with the encoder features.
        """
        if self.image_text_encoder is None:
            return None

        uint_img = (image.cpu().numpy() * 255).astype(np.uint8)
        masks = self.generate_mask(uint_img)

        # CLIP features global
        global_feat = self.generate_global_features(uint_img)

        # CLIP features per ROI
        outfeat = self.generate_local_features(uint_img, masks, global_feat)
        return outfeat
