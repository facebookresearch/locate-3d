# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.


from typing import List, Optional, Union

import clip
import numpy as np
import torch
from PIL import Image


class ClipEncoder:
    """Simple wrapper for encoding different things as text."""

    def __init__(self, version="ViT-B/32", device: Optional[str] = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda":
            device = f"cuda:{torch.cuda.current_device()}"
        self.device = device
        version = version.replace("_", "/")
        self.version = version
        self.model, self.preprocess = clip.load(self.version, device=self.device)

    def encode_image(self, image: np.ndarray):
        """Encode this input image to a CLIP vector"""
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy() * 255
        image = image.astype(np.uint8)
        pil_image = Image.fromarray(image)
        processed_image = self.preprocess(pil_image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_features = self.model.encode_image(processed_image)
        return image_features.float()

    def encode_text(self, text: Union[str, List[str]], truncate: bool = True):
        """Return clip vector for text"""
        if not isinstance(text, list):
            text = [text]
        text = clip.tokenize(text, truncate=truncate).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(text)
        return text_features.float()

    def text_feature_dim(self):
        return 512 if "ViT-B" in self.version else 768

    def __name__(self):
        return f"CLIP-{self.version}".replace("/", "_")
