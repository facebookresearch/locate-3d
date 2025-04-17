# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.


import os
from collections import Counter

import hydra
import numpy as np
import open3d as o3d
import torch
import torch.nn as nn

from models.encoder_3djepa import Encoder3DJEPA
from models.locate_3d_decoder import Locate3DDecoder

from huggingface_hub import PyTorchModelHubMixin


def get_text_from_token_indices(tokenizer, text, indices):
    """
    Extract text from specific token indices given original text

    Args:
        tokenizer: The HuggingFace tokenizer
        text: Original text string (e.g. "select the door that is...")
        indices: List of indices to extract (on tokenizer space)

    Returns:
        String containing the text from the specified tokens
    """
    # First encode the text to get the tokens
    encoded = tokenizer(text, return_offsets_mapping=True)

    # Get the character spans for our desired indices
    offset_mapping = encoded["offset_mapping"]
    selected_spans = []
    for idx in indices:
        if idx < len(offset_mapping):
            selected_spans.append(offset_mapping[idx])

    # Extract the text spans and join them
    text_pieces = [text[start:end] for start, end in selected_spans]
    return "".join(text_pieces)


def load_state_dict(model, state_dict):
    """
    Loads a state_dict into a model, with the flexibility to handle
    different prefixes like 'module.' that PyTorch DDP might add.
    """
    # Remove 'module.' prefix if it exists
    cleaned_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            cleaned_state_dict[k[7:]] = v  # Remove first 7 characters ('module.')
        else:
            cleaned_state_dict[k] = v

    model.load_state_dict(cleaned_state_dict, strict=True)
    return model


def downsample(pointcloud_dict, limit_points):
    if len(pointcloud_dict["points"]) < limit_points:
        return pointcloud_dict

    indices = torch.randperm(
        len(pointcloud_dict["points"]), device=pointcloud_dict["points"].device
    )[:limit_points]
    return {k: v[indices] for k, v in pointcloud_dict.items()}


class Locate3D(
    nn.Module,
    PyTorchModelHubMixin,
    license="cc-by-nc-4.0",
):

    def __init__(self, cfg):
        """
        Initialize the Locate3D model.

        Args:
            cfg: Configuration object containing model settings.
        """
        super().__init__()
        self.cfg = cfg
        self.encoder = self.__init_encoder()
        self.decoder = self.__init_decoder()
        self.freeze_encoder = False

    def __init_encoder(self):
        """Initialize and return the encoder module."""
        return Encoder3DJEPA(**self.cfg["encoder"]).cuda()

    def __init_decoder(self):
        """Initialize and return the decoder module."""
        return Locate3DDecoder(**self.cfg["decoder"]).cuda()

    def train(self, mode: bool = True):
        """Set the model to training mode."""
        super().train(mode)
        if not mode:
            self.encoder.eval()
            self.decoder.eval()
        else:
            self.decoder.train()
            if not self.freeze_encoder:
                self.encoder.train()
            else:
                self.encoder.eval()
        torch.cuda.empty_cache()

    def load_from_checkpoint(self, checkpoint_path: str):
        """Load model state from a checkpoint."""
        checkpoint = torch.load(checkpoint_path)
        load_state_dict(self, checkpoint["model_state_dict"])

    def forward(self, featurized_scene_dict, query):
        """
        Forward pass of the model.

        Args:
            sample_batch: Batch of input samples.

        Returns:
            Decoder output for the processed batch.
        """
        # downsample

        encoded = self.encoder(featurized_scene_dict)
        return self.decoder(encoded, query)

    @torch.inference_mode()
    def inference(self, featurized_scene_dict, query):
        """
        Perform inference on a single sample.

        Args:
            sample: Input sample.

        Returns:
            Processed prediction.
        """
        prediction = self(featurized_scene_dict, query)
        return self._post_process_sigmoid_loss_prediction(query, prediction)

    def _post_process_sigmoid_loss_prediction(self, query, prediction):
        """
        Post-process the model prediction. -- This is for models
        trained with the sigmoid loss.

        Args:
            sample: Training sample.
            prediction: Model prediction.
        """

        assert len(prediction["pred_logits"]) == 1, "Batched inference not supported"
        masks, tokens = torch.where(prediction["pred_logits"][0].sigmoid() > 0.5)
        correspondence = {}
        for mask, token in zip(masks, tokens):
            if mask.item() not in correspondence:
                correspondence[mask.item()] = []
            correspondence[mask.item()].append(token.item())

        instances = []
        for mask_idx in correspondence.keys():
            mask = prediction["pred_masks"][0][mask_idx]
            token_indices = correspondence[mask_idx]
            confidence = (
                prediction["pred_logits"][0][mask_idx][token_indices].sigmoid().mean()
            )
            mask = mask.sigmoid().detach()
            bbox = prediction["pred_boxes"][0][mask_idx]
            instances.append(
                {
                    "tokens_assigned": token_indices,
                    "text": get_text_from_token_indices(
                        self.decoder.tokenizer, query, token_indices
                    ),
                    "mask": mask,
                    "bbox": bbox,
                    "confidence": confidence,
                }
            )

        return instances
