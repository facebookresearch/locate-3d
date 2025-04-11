from typing import Union

import numpy as np
from torch import Tensor


class BaseImageTextEncoder:
    """
    Encodes images, encodes text, and allows comparisons between the two encoding.
    """

    def encode_image(self, image: Union[np.ndarray, Tensor]):
        raise NotImplementedError

    def encode_text(self, text: str):
        raise NotImplementedError
