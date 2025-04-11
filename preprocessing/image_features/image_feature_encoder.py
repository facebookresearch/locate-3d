from typing import Union

import numpy as np
from torch import Tensor


class BaseImageFeatureEncoder:
    """
    Encodes images into pixel-aligned features.
    """

    def encode_image(self, image: Union[np.ndarray, Tensor]):
        raise NotImplementedError
