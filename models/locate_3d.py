import os
from collections import Counter

import hydra
import numpy as np
import open3d as o3d
import torch
import torch.nn as nn

from models.encoder_3djepa import Encoder3DJEPA
from models.locate_3d_decoder import Locate3DDecoder


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

    model_dict = model.state_dict()
    # Filter out unexpected keys
    filtered_dict = {k: v for k, v in cleaned_state_dict.items() if k in model_dict}
    # Update model state dict
    model_dict.update(filtered_dict)
    # Load the filtered state dict
    model.load_state_dict(model_dict, strict=False)
    return model


class Locate3D(nn.Module):

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
        return Encoder3DJEPA(**self.cfg.encoder).cuda()

    def __init_decoder(self):
        """Initialize and return the decoder module."""
        return Locate3DDecoder(**self.cfg.decoder).cuda()

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

    def forward(self, sample_batch):
        """
        Forward pass of the model.

        Args:
            sample_batch: Batch of input samples.

        Returns:
            Decoder output for the processed batch.
        """
        post_encoded_batch = []
        encoded_batch, _ = self.encoder(sample_batch)
        for element in encoded_batch:
            for transform in self.post_encode_transforms:
                element, _ = transform.transform_sample(element)
            post_encoded_batch.append(element)
        return self.decoder(post_encoded_batch)

    @torch.inference_mode()
    def inference(self, sample):
        """ 
        Perform inference on a single sample.

        Args:
            sample: Input sample.
        
        Returns:
            Processed prediction.
        """

        if isinstance(sample, list):
            assert len(list) == 0, "No batched inference supported"
            sample = sample[0]

        assert isinstance(sample, TrainingSample), "Must be a TrainingSample"

        sample_is_encoded = hasattr(sample, "encoded")
        if sample_is_encoded:
            prediction = self.decoder([sample])
        else:
            encoded_sample, _ = self.encoder([sample])
            for transform in self.post_encode_transforms:
                encoded_sample, _ = transform.transform_sample(encoded_sample)
            prediction = self.decoder(encoded_sample)
            encoded_sample[0].encoded = True

        return self._post_process_sigmoid_loss_prediction(sample, prediction)

    def _post_process_sigmoid_loss_prediction(self, sample, prediction):
        '''
        Post-process the model prediction. -- This is for models
        trained with the sigmoid loss.

        Args:
            sample: Training sample.
            prediction: Model prediction.
        '''

        assert len(prediction["pred_logits"]) == 1, "Batched inference not supported"
        masks, tokens = torch.where(prediction['pred_logits'][0].sigmoid() > 0.5)
        correspondence = {}
        for mask, token in zip(masks, tokens):
            if mask.item() not in correspondence:
                correspondence[mask.item()] = []
            correspondence[mask.item()].append(token.item())
        
        instances = []
        for mask_idx in correspondence.keys():
            mask = prediction['pred_masks'][0][mask_idx]
            token_indices = correspondence[mask_idx]
            confidence = prediction['pred_logits'][0][mask_idx][token_indices].sigmoid().mean()
            mask = mask.sigmoid().detach()
            bbox = prediction['pred_boxes'][0][mask_idx]
            instances.append(
                {
                    "tokens_assigned": token_indices,
                    "text": get_text_from_token_indices(self.decoder.tokenizer, sample.goal, token_indices),
                    "mask": mask,
                    "bbox": bbox,
                    'confidence': confidence
                }
            )
        
        return instances
