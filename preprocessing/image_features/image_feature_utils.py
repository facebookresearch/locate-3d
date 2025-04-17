# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.


import torch


def padding_to_patch(image, patch_size):
    _, _, h, w = image.size()
    pad_h = (patch_size - h % patch_size) % patch_size
    pad_w = (patch_size - w % patch_size) % patch_size
    padding = (0, pad_w, 0, pad_h)
    return torch.nn.functional.pad(image, padding, mode="constant", value=0)
