from PIL import Image
from typing import Any, Dict, List, Optional, Tuple, Union
from torchvision.transforms import functional as F
from pathlib import Path
import torch
import cv2
import numpy as np

def get_image_from_path(
    image_path: Union[str, Path],
    height: Optional[int] = None,
    width: Optional[int] = None,
    keep_alpha: bool = False,
    resample=Image.BILINEAR,
) -> torch.Tensor:

    pil_image = Image.open(image_path)
    assert (height is None) == (width is None)  # Neither or both
    if height is not None and pil_image.size != (width, height):
        pil_image = pil_image.resize((width, height), resample=resample)

    image = F.to_tensor(pil_image)

    if not keep_alpha and image.shape[-1] == 4:
        image = image[:, :, :3]

    return image

def get_depth_image_from_path(
    filepath: Path,
    height: Optional[int] = None,
    width: Optional[int] = None,
    scale_factor: float = 1.0,
    interpolation: int = cv2.INTER_NEAREST,
) -> torch.Tensor:
    """Loads, rescales and resizes depth images.
    Filepath points to a 16-bit or 32-bit depth image, or a numpy array `*.npy`.
    # Adapted from https://github.com/nerfstudio-project/nerfstudio/blob/main/nerfstudio/data/dataparsers/scannet_dataparser.py
    Args:
        filepath: Path to depth image.
        height: Target depth image height.
        width: Target depth image width.
        scale_factor: Factor by which to scale depth image.
        interpolation: Depth value interpolation for resizing.

    Returns:
        Depth image torch tensor with shape [height, width].
    """
    assert (height is None) == (width is None)  # Neither or both
    if filepath.suffix == ".npy":
        image = np.load(filepath) * scale_factor
        assert (height is None) == (width is None)  # Neither or both
    else:
        image = cv2.imread(str(filepath.absolute()), cv2.IMREAD_ANYDEPTH)
        image = image.astype(np.float64) * scale_factor
    do_resize = height is not None and image.shape[:2] != (height, width)
    if do_resize:
        image = cv2.resize(image, (width, height), interpolation=interpolation)
    return torch.from_numpy(image[:, :])
