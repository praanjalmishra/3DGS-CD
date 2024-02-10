# Projection functions for NeRFacto2
import torch
from nerfstudio.utils.img_utils import in_image


def undistort_points(points, intrinsics, dist_coeffs):
    """
    Undistort 2D image points

    Args:
        points (NxMx2 Tensor): 2D points (in pixel coords)
        intrinsics (Nx3x3 Tensor): Camera intrinsics
        dist_coeffs (Nx4 Tensor): Distortion coefficients [k1, k2, p1, p2]
    
    Returns:
        points_und (NxMx2 Tensor): Undistorted 2D points
    """
    assert intrinsics.shape[-2:] == (3, 3)
    assert dist_coeffs.shape[-1] == 4
    # Extract cam parameters
    f_x = intrinsics[:, 0, 0].unsqueeze(-1)
    f_y = intrinsics[:, 1, 1].unsqueeze(-1)
    c_x = intrinsics[:, 0, 2].unsqueeze(-1)
    c_y = intrinsics[:, 1, 2].unsqueeze(-1)
    k1 = dist_coeffs[:, 0].unsqueeze(-1)
    k2 = dist_coeffs[:, 1].unsqueeze(-1)
    p1 = dist_coeffs[:, 2].unsqueeze(-1)
    p2 = dist_coeffs[:, 3].unsqueeze(-1)
    # Normalize coordinates
    x_d = (points[..., 0] - c_x) / f_x
    y_d = (points[..., 1] - c_y) / f_y
    # Compute radial distortion
    r2 = x_d**2 + y_d**2
    radial_factor = 1 + k1 * r2 + k2 * r2**2
    # Compute tangential distortion
    x_tangential = 2 * p1 * x_d * y_d + p2 * (r2 + 2 * x_d**2)
    y_tangential = 2 * p2 * x_d * y_d + p1 * (r2 + 2 * y_d**2)
    # Compute undistorted normalized coordinates by dividing out the distortion terms
    x = (x_d - x_tangential) / radial_factor
    y = (y_d - y_tangential) / radial_factor
    # Convert undistorted normalized coordinates to pixel coordinates
    x_pixel = x * f_x + c_x
    y_pixel = y * f_y + c_y
    return torch.stack([x_pixel, y_pixel], dim=-1)

def distort_points(points, intrinsics, dist_coeffs):
    """
    Distort 2D image points

    Args:
        points (NxMx2 Tensor): 2D points (in pixel coords)
        intrinsics (Nx3x3 Tensor): Camera intrinsics
        dist_coeffs (Nx4 Tensor): Distortion coefficients [k1, k2, p1, p2]

    Returns:
        points_dist (NxMx2 Tensor): Distorted 2D points
    """
    # Apply distortion
    k1 = dist_coeffs[:, 0].unsqueeze(-1)
    k2 = dist_coeffs[:, 1].unsqueeze(-1)
    p1 = dist_coeffs[:, 2].unsqueeze(-1)
    p2 = dist_coeffs[:, 3].unsqueeze(-1)
    fx = intrinsics[:, 0, 0].unsqueeze(-1)
    fy = intrinsics[:, 1, 1].unsqueeze(-1)
    cx = intrinsics[:, 0, 2].unsqueeze(-1)
    cy = intrinsics[:, 1, 2].unsqueeze(-1)
    # Normalize coordinates
    x_norm = (points[..., 0] - cx) / fx
    y_norm = (points[..., 1] - cy) / fy
    r2 = x_norm ** 2 + y_norm ** 2
    d = 1.0 + r2 * (k1 + r2 * k2)
    x_distorted = d*x_norm + 2*p1*x_norm*y_norm + p2*(r2+2*x_norm**2)
    y_distorted = d*y_norm + 2*p2*x_norm*y_norm + p1*(r2+2*y_norm**2)
    x_distorted = x_distorted * fx + cx
    y_distorted = y_distorted * fy + cy
    points2d = torch.stack((x_distorted, y_distorted), dim=-1)
    return points2d

def project_points(points, poses, intrinsics, dist_coeffs, H, W):
    """
    Project 3D points to 2D using the camera poses and intrinsics.

    Args:
        points (..., 3): 3D points
        poses (N, 4, 4): Camera-to-world poses
        intrinsics (N, 3, 3): Camera intrinsics
        dist_coeffs (N, 4): Distortion coefficients
        H, W (int): Image height and width

    Returns:
        points2d (N, ..., 2): 2D points
        valid_pts (N, ...): Points that can project into image
    """
    device = points.device
    points = points.view(-1, 3)
    # 3D point projection
    poses = poses.inverse() # c2w -> w2c
    # [R|T] @ [X|1]
    points_cam = torch.einsum(
        'bij,aj->bai', poses[:, :3, :3].to(device), points
    )
    points_cam = points_cam + \
        poses[:, :3, 3].to(device).unsqueeze(1).expand_as(points_cam)
    is_behind_camera = points_cam[..., 2] < 0 # behind-cam pts
    points_cam = points_cam / (points_cam[..., -1:] + 1e-7)
    # K @ X^c
    points2d = torch.einsum(
        'bji,bai->baj', intrinsics.to(device), points_cam
    )
    points2d = points2d[..., :2] / (points2d[..., -1:] + 1e-7)
    is_in_image = in_image(points2d, H, W)
    points2d = distort_points(points2d, intrinsics, dist_coeffs)
    points2d = points2d.view(poses.shape[0], *points.shape[:-1], 2)
    return points2d, is_in_image & ~is_behind_camera


def proj_check_3D_points(
    points, poses, intrinsics, dist_coeffs, masks,
    cutoff=0.95, chunk_size=3e5
):
    """
    Check if 3D points are inside object by projecting them to 2D masks

    Args:
        points (..., 3): 3D points
        poses (N, 4, 4): Camera-to-world poses
        intrinsics (N, 3, 3): Camera intrinsics
        dist_coeffs (N, 4): Distortion coefficients
        masks (N, 1, H, W): 2D object masks
        cutoff (float): percent cutoff to consider a point as in the object
        chunk_size (int): size of each chunk to process at a time

    Returns:
        masks_percentage (...): # proj checks passed / # 2D masks
    """
    device = points.device

    original_shape = points.shape[:-1]
    points_reshaped = points.view(-1, 3)
    if len(masks.shape) == 4:
        assert masks.shape[1] == 1
        masks = masks.squeeze(1)

    num_chunks = (points_reshaped.shape[0] + chunk_size - 1) // chunk_size
    chunks = torch.chunk(points_reshaped, int(num_chunks), dim=0)
    masks_in_object = []
    for chunk in chunks:
        points2d, valid_mask = project_points(
            chunk, poses, intrinsics, dist_coeffs,
            masks.shape[-2], masks.shape[-1]
        )
        # NOTE: Only consider points that can project in the image bounds
        valid_mask_counts = valid_mask.sum(dim=0)
        # Clamp points to be within valid region
        points2d[..., 0] = points2d[..., 0].clamp(0, masks.shape[-1] - 1)
        points2d[..., 1] = points2d[..., 1].clamp(0, masks.shape[-2] - 1)
        points2d = points2d.round().long()
        # Get the x and y coordinates
        x_coords = points2d[..., 0]
        y_coords = points2d[..., 1]
        # Create indices for batch dimension
        batch_indices = torch.arange(masks.shape[0]).view(-1, 1).to(device)
        # Use advanced indexing to get the mask values
        mask_values = masks[batch_indices, y_coords, x_coords]
        # Apply the valid mask
        mask_values[~valid_mask] = 0
        # Calculate mask counts
        mask_counts = mask_values.sum(dim=0).float()
        mask_percentage = mask_counts / (valid_mask_counts.float() + 1e-7)
        mask_in_chunk = (mask_percentage >= cutoff).to(device)
        masks_in_object.append(mask_in_chunk)
    return torch.cat(masks_in_object, dim=0).view(*original_shape)