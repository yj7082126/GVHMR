import numpy as np
import torch


def axis_rotate_to_matrix(angle, axis="x", with_trans=False, use_deg=False):
    """Get rotation matrix for rotating around one axis
    Args:
        angle: (N, 1)
    Returns:
        R: (N, 3, 3)
    """
    if use_deg:
        angle = torch.tensor([np.deg2rad(angle)], dtype=torch.float)
    else:
        angle = torch.tensor([angle], dtype=torch.float)

    c = torch.cos(angle)
    s = torch.sin(angle)
    z = torch.zeros_like(angle)
    o = torch.ones_like(angle)
    if axis == "x":
        R = torch.stack([o, z, z, z, c, -s, z, s, c], dim=1).view(-1, 3, 3)
    elif axis == "y":
        R = torch.stack([c, z, s, z, o, z, -s, z, c], dim=1).view(-1, 3, 3)
    else:
        assert axis == "z"
        R = torch.stack([c, -s, z, s, c, z, z, z, o], dim=1).view(-1, 3, 3)
    if with_trans:
        R_h = torch.eye(4, dtype=torch.float32).unsqueeze(0).repeat(R.shape[0], 1, 1)
        R_h[:, :3, :3] = R
        return R_h
    return R