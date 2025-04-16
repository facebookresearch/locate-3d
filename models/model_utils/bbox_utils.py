import torch

@torch.jit.script
def box_cxcyczwhd_to_xyzxyz_jit(x):
    centers = x[..., :3]
    dims = torch.clamp(x[..., 3:], min=1e-6)
    return torch.cat([centers - 0.5 * dims, centers + 0.5 * dims], dim=-1)
