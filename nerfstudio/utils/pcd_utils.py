# Point cloud processing functions for for NeRFacto2
import cv2
import torch
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist


def compute_point_cloud(depths, poses, Ks, masks):
    """
    Compute the point cloud from depths maps, camera poses and intrinsics

    Args:
        depths (N, 1, H, W): Depth maps
        poses (N, 4, 4): Camera poses
        Ks (N, 3, 3): Camera intrinsics for each image
        masks (N, 1, H, W): Binary masks

    Returns:
        point_cloud (torch.Tensor): Point cloud
    """
    N, _, H, W = depths.shape
    # Create a batched meshgrid
    u = torch.linspace(
        0, W - 1, W, device=depths.device
    ).repeat(H, 1).expand(N, H, W)
    v = torch.linspace(
        0, H - 1, H, device=depths.device
    ).repeat(W, 1).t().expand(N, H, W)
    # Normalize (u, v) coordinates and scale by depth map
    X = (u - Ks[:, 0, 2].view(N, 1, 1)) \
        / Ks[:, 0, 0].view(N, 1, 1) * depths.squeeze(1)
    Y = (v - Ks[:, 1, 2].view(N, 1, 1)) \
        / Ks[:, 1, 1].view(N, 1, 1) * depths.squeeze(1)
    Z = depths.squeeze(1)
    # Stack to create (N, H, W, 3) post cloud map
    point_cloud = torch.stack((X, Y, Z), dim=-1)
    # Apply 4x4 camera pose matrix to map to world coordinates
    ones = torch.ones(N, H, W, 1, device=depths.device)
    point_cloud_hom = torch.cat((point_cloud, ones), dim=-1)
    point_cloud_world = torch.einsum(
        'bij,bklj->bkli', poses, point_cloud_hom
    )
    # Reshape to (NxHxW, 4) for easy min/max computation
    point_cloud_world_reshaped = point_cloud_world.reshape(-1, 4)
    # Remove zero points using the mask
    non_zero_mask = masks.view(-1).bool()
    point_cloud_in_mask = point_cloud_world_reshaped[non_zero_mask, :3]
    return point_cloud_in_mask


def point_cloud_filtering(point_cloud, percentile_threshold=0.90):
    """
    Simple percentile-based point cloud outlier filtering
    TODO: Improve the outlier filtering strategy
    
    Args:
        point_cloud (Nx3): input point cloud
    
    Returns:
        point_cloud_inliers (Mx3): inlier point cloud
    """
    # Compute the centroid of the point cloud
    # TODO: make this averaging operation more robust to outliers
    centroid = torch.mean(point_cloud, dim=-2)
    # Compute distances of all points to the centroid
    distances = torch.norm(point_cloud - centroid, dim=-1)
    # Filter out the paints that are beyond the 95% percentile
    threshold_distance = torch.quantile(distances, percentile_threshold)
    inlier_mask = distances < threshold_distance
    point_cloud_inliers = point_cloud[inlier_mask]
    return point_cloud_inliers


def nn_distance(point_cloud1, point_cloud2):
    """
    Efficiently compute the approximate NN distance between two point clouds

    Args:
        point_cloud1 (N, 3): First point cloud
        point_cloud2 (M, 3): Second point cloud
    
    Returns:
        distances (float): Nearest neighbor distance
    """
    assert point_cloud1.shape[-1] == 3
    assert point_cloud2.shape[-1] == 3
    # import time
    # start = time.time()
    index_params = dict(algorithm = 1, trees = 1)  # KDTree
    search_params = dict(checks = 2) 
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    flann.add([point_cloud1.cpu().numpy()])
    flann.train()
    # Find the nearest neighbors
    matches = flann.match(point_cloud2.cpu().numpy())
    smallest_distance = min(match.distance for match in matches)
    # print(f"KDTree query time: {time.time() - start}")
    return smallest_distance


def pcd_size(pcd):
    """
    Efficiently compute the maximum distance between two points in a pcd

    Args:
        pcd (Nx3): point cloud

    Returns:
        max_dist (float): Maximum distance between two points in a pcd
    """
    # Max-distance points must lie on the convex hull
    hull = ConvexHull(pcd.cpu().numpy())
    hullpoints = pcd.cpu().numpy()[hull.vertices, :]
    hdist = cdist(hullpoints, hullpoints, metric='euclidean')
    return hdist.max()


def transform_point_cloud(point_cloud, transform_matrix): 
    """
    Transforms a point cloud using a given transformation matrix.
    
    Args:
        point_cloud (..., 3): point cloud.
        transform_matrix (4, 4): 4x4 transformation matrix.

    Returns:
        transformed_point_cloud (..., 3): Transformed point cloud of shape.
    """
    assert point_cloud.shape[-1] == 3
    assert transform_matrix.shape == (4, 4)
    # Get the shape of the input point cloud
    *leading_dims, _ = point_cloud.shape
    # Convert point cloud to homogeneous coordinates
    ones = torch.ones((*leading_dims, 1)).to(point_cloud)
    point_cloud_homo = torch.cat((point_cloud, ones), dim=-1)
    #Transform the point cloud using the transformation matrix
    transformed_point_cloud_homo = torch.einsum(
        "ij,...j->...i", transform_matrix, point_cloud_homo
    )
    # Extract the 3D coordinates of the transformed point cloud 
    transformed_point_cloud = transformed_point_cloud_homo[..., :3]
    return transformed_point_cloud


def compute_3D_bbox(point_cloud):
    """
    Computes the 3D bounding box
    from depth maps, camera pases and intrinsics.

    Args:
        point cloud (Nx3 Tensor): point cloud.

    Returns:
        bbox_min (3-Tuple): Minimum corner of the bounding box.
        bbox_max (3-Tuple): Maximum corner of the bounding box.
    """
    assert point_cloud.shape[-1] == 3
    bbox_min = torch.min(point_cloud, dim=0)[0].cpu().numpy()
    bbox_max = torch.max(point_cloud, dim=0)[0].cpu().numpy()
    return tuple(bbox_min), tuple(bbox_max)


def expand_3D_bbox(bbox3d, percentage=.8):
    """
    Expand 3D bounding box by a certain percentage

    Args:
        bbox3d ((bbox_min, bbox_max)): 3D bounding box
        percentage (float): Percentage to expand

    Return:
        bbox3d_new ((bbox_min, bbox_max)): Expanded 3D bounding box
    """
    bbox_min, bbox_max = bbox3d
    xmin, ymin, zmin = bbox_min
    xmax, ymax, zmax = bbox_max
    # Calculate the center of the bounding box
    x_center = (xmin + xmax) / 2
    y_center = (ymin + ymax) / 2
    z_center = (zmin + zmax) / 2
    # Calculate half-widths
    x_half = x_center - xmin
    y_half = y_center - ymin
    z_half = z_center - zmin
    # Calculate expansion amounts
    x_expand = x_half * percentage
    y_expand = y_half * percentage
    z_expand = z_half * percentage
    # Calculate new bounding box corners
    xmin_new = x_center - (x_half + x_expand)
    xmax_new = x_center + (x_half + x_expand)
    ymin_new = y_center - (y_half + y_expand)
    ymax_new = y_center + (y_half + y_expand)
    zmin_new = z_center - (z_half + z_expand)
    zmax_new = z_center + (z_half + z_expand)
    return (xmin_new, ymin_new, zmin_new), (xmax_new, ymax_new, zmax_new)