# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.

"""
This file contains a torch implementation and helpers of a
"voxelized pointcloud" that stores features, centroids, and counts in a sparse voxel grid
"""

# from https://github.com/facebookresearch/home-robot/blob/a32a23d10f7b4474833ae66e995af666e4e80718/src/home_robot/home_robot/utils/voxel.py#L21

from collections import namedtuple
from typing import Optional, Tuple, Union

import torch
from torch import Tensor
from torch_geometric.nn.pool.consecutive import consecutive_cluster
from torch_geometric.nn.pool.voxel_grid import voxel_grid
from torch_geometric.utils import scatter


class VoxelizedPointcloud:
    _INTERNAL_TENSORS = [
        "_points",
        "_features",
        "_weights",
        "_rgb",
        "dim_mins",
        "dim_maxs",
        "_mins",
        "_maxs",
    ]

    _INIT_ARGS = ["voxel_size", "dim_mins", "dim_maxs", "feature_pool_method"]

    def __init__(
        self,
        voxel_size: float = 0.05,
        dim_mins: Optional[Tensor] = None,
        dim_maxs: Optional[Tensor] = None,
        feature_pool_method: str = "mean",
        limit_voxels: Optional[int] = None,
    ):
        """

        Args:
            voxel_size (Tensor): float, voxel size in each dim
            dim_mins (Tensor): 3, tensor of minimum coords possible in voxel grid
            dim_maxs (Tensor): 3, tensor of maximum coords possible in voxel grid
            feature_pool_method (str, optional): How to pool features within a voxel. One of 'mean', 'max', 'sum'. Defaults to 'mean'.
            limit_voxels (int, optional): Maximum number of voxels to store. Defaults to None.
        """

        assert (dim_mins is None) == (dim_maxs is None)
        self.dim_mins = dim_mins
        self.dim_maxs = dim_maxs
        self.voxel_size = voxel_size
        self.feature_pool_method = feature_pool_method
        self.limit_voxels = limit_voxels
        assert self.feature_pool_method in [
            "mean",
            "max",
            "sum",
        ], f"Unknown feature pool method {feature_pool_method}"

        self.reset()

    def reset(self):
        """Resets internal tensors"""
        self._points, self._features, self._weights, self._rgb = None, None, None, None
        self._mins = self.dim_mins
        self._maxs = self.dim_maxs

    def add(
        self,
        points: Tensor,
        features: Optional[Tensor],
        rgb: Optional[Tensor],
        weights: Optional[Tensor] = None,
    ):
        """Add a feature pointcloud to the voxel grid.

        Args:
            points (Tensor): N x 3 points to add to the voxel grid
            features (Tensor): N x D features associated with each point.
                Reduction method can be set with feature_reduciton_method in init
            rgb (Tensor): N x 3 colors s associated with each point.
            weights (Optional[Tensor], optional): Weights for each point.
                Can be detection confidence, distance to camera, etc.
                Defaults to None.
        """
        if weights is None:
            weights = torch.ones_like(points[..., 0])

        # Update voxel grid bounds
        # This isn't strictly necessary since the functions below can infer the bounds
        # But we might want to do this anyway to enforce that bounds are a multiple of self.voxel_size
        # And to enforce that the added points are within user-defined boundaries, if those were specified.
        pos_mins, _ = points.min(dim=0)
        pos_maxs, _ = points.max(dim=0)
        if self.dim_mins is not None:
            assert torch.all(
                self.dim_mins <= pos_mins
            ), "Got points outside of user-defined 3D bounds"
        if self.dim_maxs is not None:
            assert torch.all(
                pos_maxs <= self.dim_maxs
            ), "Got points outside of user-defined 3D bounds"

        if self._mins is None:
            self._mins, self._maxs = pos_mins, pos_maxs
            # recompute_voxels = True
        else:
            assert (
                self._maxs is not None
            ), "How did self._mins get set without self._maxs?"
            # recompute_voxels = torch.any(pos_mins < self._mins) or torch.any(self._maxs < pos_maxs)
            self._mins = torch.min(self._mins, pos_mins)
            self._maxs = torch.max(self._maxs, pos_maxs)

        if self._points is None:
            assert (
                self._features is None
            ), "How did self._points get unset while _features is set?"
            # assert self._rgbs is None, "How did self._points get unset while _rgbs is set?"
            assert (
                self._weights is None
            ), "How did self._points get unset while _weights is set?"
            all_points, all_features, all_weights, all_rgb = (
                points,
                features,
                weights,
                rgb,
            )
        else:
            assert (self._features is None) == (features is None)
            all_points = torch.cat([self._points, points], dim=0)
            all_weights = torch.cat([self._weights, weights], dim=0)
            all_features = (
                torch.cat([self._features, features], dim=0)
                if (features is not None)
                else None
            )
            all_rgb = torch.cat([self._rgb, rgb], dim=0) if (rgb is not None) else None
        # Future optimization:
        # If there are no new voxels, then we could save a bit of compute time
        # by only recomputing the voxel/cluster for the new points
        # e.g. if recompute_voxels:
        #   raise NotImplementedError
        cluster_voxel_idx, cluster_consecutive_idx, _ = voxelize(
            all_points, voxel_size=self.voxel_size, start=self._mins, end=self._maxs
        )
        self._points, self._features, self._weights, self._rgb = reduce_pointcloud(
            cluster_consecutive_idx,
            pos=all_points,
            features=all_features,
            weights=all_weights,
            rgbs=all_rgb,
            feature_reduce=self.feature_pool_method,
        )
        if self.limit_voxels is not None and len(self._points) > self.limit_voxels:
            # randomly drop voxels if they exceed the limit
            self.downsample_pointcloud(self.limit_voxels)
        return

    def get_idxs(self, points: Tensor) -> Tensor:
        """Returns voxel index (long tensor) for each point in points

        Args:
            points (Tensor): N x 3

        Returns:
            cluster_voxel_idx (Tensor): The voxel grid index (long tensor) for each point in points
            cluster_consecutive_idx (Tensor): Voxel grid reindexed to be consecutive (packed)
        """
        (
            cluster_voxel_idx,
            cluster_consecutive_idx,
            _,
        ) = voxelize(points, self.voxel_size, start=self._mins, end=self._maxs)
        return cluster_voxel_idx, cluster_consecutive_idx

    def get_voxel_idx(self, points: Tensor) -> Tensor:
        """Returns voxel index (long tensor) for each point in points

        Args:
            points (Tensor): N x 3

        Returns:
            Tensor: voxel index (long tensor) for each point in points
        """
        (
            cluster_voxel_idx,
            _,
        ) = self.get_idxs(points)
        return cluster_voxel_idx

    def get_consecutive_cluster_idx(self, points: Tensor) -> Tensor:
        """Returns voxel index (long tensor) for each point in points

        Args:
            points (Tensor): N x 3

        Returns:
            Tensor: voxel index (long tensor) for each point in points
        """
        (
            _,
            cluster_consecutive_idx,
        ) = self.get_idxs(points)
        return cluster_consecutive_idx

    def get_pointcloud(self) -> Tuple[Tensor]:
        """Returns pointcloud (1 point per occupied voxel)

        Returns:
            points (Tensor): N x 3
            features (Tensor): N x D
            weights (Tensor): N
        """
        return self._points, self._features, self._weights, self._rgb

    def clone(self):
        """
        Deep copy of object. All internal tensors are cloned individually.

        Returns:
            new VoxelizedPointcloud object.
        """
        other = self.__class__({k: getattr(self, k) for k in self._INIT_ARGS})
        for k in self._INTERNAL_TENSORS:
            v = getattr(self, k)
            if torch.is_tensor(v):
                setattr(other, k, v.clone())
        return other

    def to(self, device: Union[str, torch.device]):
        """

        Args:
          device: Device (as str or torch.device) for the new tensor.

        Returns:
          self
        """
        other = self.clone()
        for k in self._INTERNAL_TENSORS:
            v = getattr(self, k)
            if torch.is_tensor(v):
                setattr(other, k, v.to(device))
        return other

    def cpu(self):
        return self.to("cpu")

    def cuda(self):
        return self.to("cuda")

    def detach(self):
        """
        Detach object. All internal tensors are detached individually.

        Returns:
            new VoxelizedPointcloud object.
        """
        other = self.__class__({k: getattr(self, k) for k in self._INIT_ARGS})
        for k in self._INTERNAL_TENSORS:
            v = getattr(self, k)
            if torch.is_tensor(v):
                setattr(other, k, v.detach())
        return other

    def downsample_pointcloud(self, limit_points: int):
        """
        Downsample a pointcloud to a fixed number of points

        Args:
            limit_points: int, number of points to limit to

        Returns:
            None
        """
        (
            self._points,
            self._weights,
            self._features,
            self._rgb,
            _,
        ) = downsample_pointcloud(
            points=self._points,
            weights=self._weights,
            features=self._features,
            rgbs=self._rgb,
            limit_points=limit_points,
        )


def reduce_pointcloud(
    voxel_cluster: Tensor,
    pos: Tensor,
    features: Tensor,
    weights: Optional[Tensor] = None,
    rgbs: Optional[Tensor] = None,
    feature_reduce: str = "mean",
) -> Tuple[Tensor]:
    """Pools values within each voxel

    Args:
        voxel_cluster (LongTensor): [N] IDs of each point
        pos (Tensor): [N, 3] position of each point
        features (Tensor): [N, D] features at each point
        weights (Optional[Tensor], optional): [N,] weights of each point. Defaults to None.
        rgbs (Optional[Tensor], optional): [N, 3] colors of each point. Defaults to None.
        feature_reduce (str, optional): Feature reduction method. Defaults to 'mean'.

    Raises:
        NotImplementedError: if unknown reduction method

    Returns:
        pos_cluster (Tensor): weighted average position within each voxel
        feature_cluster (Tensor): aggregated feature of each voxel
        weights_cluster (Tensor): aggregated weights of each voxel
        rgb_cluster (Tensor): colors of each voxel
    """
    if weights is None:
        weights = torch.ones_like(pos[..., 0])
    weights_cluster = scatter(weights, voxel_cluster, dim=0, reduce="sum")

    pos_cluster = scatter_weighted_mean(
        pos, weights, voxel_cluster, weights_cluster, dim=0
    )

    if rgbs is not None:
        rgb_cluster = scatter_weighted_mean(
            rgbs, weights, voxel_cluster, weights_cluster, dim=0
        )
    else:
        rgb_cluster = None

    if features is None:
        return pos_cluster, None, weights_cluster, rgb_cluster

    if feature_reduce == "mean":
        feature_cluster = scatter_weighted_mean(
            features, weights, voxel_cluster, weights_cluster, dim=0
        )
    elif feature_reduce == "max":
        feature_cluster = scatter(features, voxel_cluster, dim=0, reduce="max")
    elif feature_reduce == "sum":
        feature_cluster = scatter(features, voxel_cluster, dim=0, reduce="sum")
    else:
        raise NotImplementedError(f"Unknown feature reduction method {feature_reduce}")

    return pos_cluster, feature_cluster, weights_cluster, rgb_cluster


def downsample_pointcloud(
    points: Tensor,
    weights: Optional[Tensor] = None,
    features: Optional[Tensor] = None,
    rgbs: Optional[Tensor] = None,
    normals: Optional[Tensor] = None,
    limit_points: Optional[int] = None,
):
    """
    Downsample a pointcloud to a fixed number of points
    """
    if limit_points is None:
        return points, weights, features, rgbs, normals
    indices = torch.randperm(len(points), device=points.device)[:limit_points]
    points = points[indices]
    if weights is not None:
        weights = weights[indices]
    if rgbs is not None:
        rgbs = rgbs[indices]
    if features is not None:
        features = features[indices]
    if normals is not None:
        normals = normals[indices]
    return points, weights, features, rgbs, normals


def scatter_weighted_mean(
    features: Tensor,
    weights: Tensor,
    cluster: Tensor,
    weights_cluster: Tensor,
    dim: int,
) -> Tensor:
    """_summary_

    Args:
        features (Tensor): [N, D] features at each point
        weights (Optional[Tensor], optional): [N,] weights of each point. Defaults to None.
        cluster (LongTensor): [N] IDs of each point (clusters.max() should be <= N, or you'll OOM)
        weights_cluster (Tensor): [N,] aggregated weights of each cluster, used to normalize
        dim (int): Dimension along which to do the reduction -- should be 0

    Returns:
        Tensor: Agggregated features, weighted by weights and normalized by weights_cluster
    """
    assert dim == 0, "Dim != 0 not yet implemented"
    feature_cluster = scatter(
        features * weights[:, None], cluster, dim=dim, reduce="sum"
    )
    feature_cluster = feature_cluster / weights_cluster[:, None]
    return feature_cluster


#######################
# Voxel cluster
#######################
def voxelize(
    pos: Tensor,
    voxel_size: float,
    batch: Optional[Tensor] = None,
    start: Optional[Union[float, Tensor]] = None,
    end: Optional[Union[float, Tensor]] = None,
) -> Tuple[Tensor]:
    """Returns voxel indices and packed (consecutive) indices for points

    Args:
        pos (Tensor): [N, 3] locations
        voxel_size (float): Size (resolution) of each voxel in the grid
        batch (Optional[Tensor], optional): Batch index of each point in pos. Defaults to None.
        start (Optional[Union[float, Tensor]], optional): Mins along each coordinate for the voxel grid.
            Defaults to None, in which case the starts are inferred from min values in pos.
        end (Optional[Union[float, Tensor]], optional):  Maxes along each coordinate for the voxel grid.
            Defaults to None, in which case the starts are inferred from max values in pos.
    Returns:
        voxel_idx (LongTensor): Idx of each point's voxel coordinate. E.g. [0, 0, 4, 3, 3, 4]
        cluster_consecutive_idx (LongTensor): Packed idx -- contiguous in cluster ID. E.g. [0, 0, 2, 1, 1, 2]
        batch_sample: See https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/pool/max_pool.html

    Example:
        voxel_size = (4.0, 4.0, 1.0)
        voxel_indices = compute_voxel_indices(locs_torch, voxel_size)

        #
        arr_max = voxel_indices.max(dim=0).values
        hashed_keys, arr_max = ravel_hash_vec(voxel_indices)
        voxel_idx, voxels_idx_cons, _ = voxelize(locs_torch, torch.tensor(voxel_size))
        assert torch.all(voxel_idx == hashed_keys)

        reconstructed_indices = unravel_hash_vec(hashed_keys, arr_max)

        print("Hashed Keys:", hashed_keys)
        print("Reconstructed Indices:", reconstructed_indices)
        assert torch.all(voxel_indices == reconstructed_indices)

    """
    voxel_cluster = voxel_grid(
        pos=pos, batch=batch, size=voxel_size, start=start, end=end
    )
    cluster_consecutive_idx, perm = consecutive_cluster(voxel_cluster)
    batch_sample = batch[perm] if batch is not None else None
    cluster_idx = voxel_cluster
    return cluster_idx, cluster_consecutive_idx, batch_sample


def compute_voxel_indices(points, voxel_size):
    """
    Compute voxel indices for a given point cloud and voxel sizes.

    Parameters:
        points (torch.Tensor): A tensor of shape (N, 3) containing XYZ coordinates of the points.
        voxel_size (torch.Tensor): (3,) The size of each voxel along each dimension.

    Returns:
        torch.Tensor: A tensor of shape (N, 3) containing voxel indices for each point.
    """
    if not isinstance(voxel_size, torch.Tensor):
        voxel_size = torch.tensor(voxel_size)

    # Ensure points tensor is in floating point format for division
    points = points.float()

    # Compute the normalized coordinates of points within the range [0, 1)
    normalized_points = (
        points - points.min(0).values
    )  # / (points.max(0).values - points.min(0).values)
    # Calculate the voxel indices along each dimension
    voxel_indices = (normalized_points / voxel_size.unsqueeze(0)).floor().int()

    return voxel_indices


HashVecReturn = namedtuple("HashVecReturnValue", ["keys", "arr_max"])


def ravel_hash_vec(voxel_indices: Tensor):
    """
    Convert voxel indices into a hashed vector using a raveling approach.

    Parameters:
        voxel_indices (torch.LongTensor): A tensor of shape (N, 3) containing voxel indices for each point.

    Returns:
        torch.Tensor: A tensor of shape (N,) containing hashed coordinates.

    Example:
        arr_max = voxel_indices.max(dim=0).values
        hashed_keys, arr_max = ravel_hash_vec(voxel_indices)
        reconstructed_indices = unravel_hash_vec(hashed_keys, arr_max)

        print("Hashed Keys:", hashed_keys)
        print("Reconstructed Indices:", reconstructed_indices)
        assert torch.all(voxel_indices == reconstructed_indices)
    """
    assert voxel_indices.ndim == 2, "Input array must be 2-dimensional"
    assert voxel_indices.dtype in [torch.int32, torch.int64], voxel_indices.dtype
    arr = voxel_indices.clone().detach().to(torch.int64)
    arr -= arr.min(0).values
    arr_max = arr.max(0).values + 1

    keys = torch.zeros(arr.shape[0], dtype=torch.int64, device=arr.device)
    # C-style indexing
    for j in reversed(range(arr.shape[1])):
        keys *= arr_max[j]
        keys += arr[:, j]
    return HashVecReturn(keys, arr_max)

    # Fortran style indexing
    for j in reversed(range(arr.shape[1] - 1)):
        keys += arr[:, j]
        keys *= arr_max[j + 1]
    keys += arr[:, -1]
    return HashVecReturn(keys, arr_max)


def unravel_hash_vec(keys: Tensor, arr_max: Tensor):
    """
    Convert hashed keys back into voxel indices using the known maximum ranges.

    Parameters:
        keys (torch.Tensor): A tensor of shape (N,) containing hashed coordinates.
        arr_max (torch.Tensor): A tensor of shape (3,) with the maximum values for each dimension used during hashing.

    Returns:
        torch.Tensor: A tensor of shape (N, 3) containing the reconstructed voxel indices.

    Example:
        arr_max = voxel_indices.max(dim=0).values
        hashed_keys, arr_max = ravel_hash_vec(voxel_indices)
        reconstructed_indices = unravel_hash_vec(hashed_keys, arr_max)

        print("Hashed Keys:", hashed_keys)
        print("Reconstructed Indices:", reconstructed_indices)
        assert torch.all(voxel_indices == reconstructed_indices)
    """
    assert keys.ndim == 1, "Keys array must be 1-dimensional"
    indices = []
    product = 1
    for i in range(arr_max.shape[0]):
        indices.append((keys // product) % arr_max[i])
        product *= arr_max[i]
    return torch.stack(indices, dim=1)
