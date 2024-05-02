# Util functions for 3D Gaussians 

import numpy as np
import torch

from gsplat._torch_impl import quat_to_rotmat


def rot2quat(rotmat):
    """
    Rotation matrix to quaternion conversion

    Args:
        rotmat: rotation matrix of shape (B, 3, 3)

    Returns:
        quat: quaternion of shape (B, 4)
    """
    from scipy.spatial.transform import Rotation as R
    rot = rotmat.cpu().numpy()
    # Create a rotation object from the rotation matrices
    rotation = R.from_matrix(rot)
    # Convert to quaternions, initially in xyzw order
    quat_xyzw = rotation.as_quat()
    # Rearrange to wxyz order
    quat_wxyz = np.hstack((quat_xyzw[:, -1:], quat_xyzw[:, :-1]))
    quat_wxyz = torch.tensor(quat_wxyz).to(rotmat)
    return quat_wxyz


def transform_gaussians(pose, means, quats):
    """
    Transform the means and quats of 3D Gaussians by a 6D pose

    Args:
        pose (4, 4): pose
        means (N, 3): Gaussian means
        quats (N, 4): Gaussian quaternions

    Returns:
        means_new (N, 3): transformed Gaussian means
        quats_new (N, 4): transformed Gaussian quaternions
    """
    assert pose.shape == (4, 4)
    assert means.shape[0] == quats.shape[0]
    pose = pose.to(means.device)
    means_new = (pose[:3, :3] @ means.T + pose[:3, 3:]).T
    # Rotate the Gaussians

    quats = quats / quats.norm(dim=-1, keepdim=True)
    rots = quat_to_rotmat(quats)
    obj_rots_new = pose[:3, :3] @ rots
    quats_new = rot2quat(obj_rots_new)
    return means_new, quats_new


def get_gaussian_endpts(means, scales, quats, n_sigma=1.0):
    """
    Get 1-sigma points along the principal axes of 3D Gaussians

    Args:
        means (N, 3): Gaussian means
        scales (N, 3): Gaussian log(scales)
        quats (N, 4): Gaussian quaternions
        n_sigma (float): number of Gaussians standard deviations to consider

    Returns:
        endpts (N, 6, 3): Gaussian endpoints
    """
    assert means.shape[0] == scales.shape[0] == quats.shape[0]
    rots = quat_to_rotmat(quats)
    # Get the endpoints
    scale_diag = torch.diag_embed(scales.exp()) * n_sigma
    scaled_axis = torch.matmul(rots, scale_diag)
    endpts = torch.zeros(means.shape[0], 6, 3, device=means.device)
    endpts[:, :3] = means.unsqueeze(1) - scaled_axis
    endpts[:, 3:] = means.unsqueeze(1) + scaled_axis
    return endpts