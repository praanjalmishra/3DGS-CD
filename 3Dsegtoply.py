import torch
import trimesh
import numpy as np
from pathlib import Path
import argparse


def extract_voxel_coords(obj_data):
    """
    Converts a 3D voxel occupancy grid (DxHxW) into (N, 3) xyz coords
    in world space (min + spacing * index)
    """
    voxel_grid = obj_data["voxel"]  # shape: (D, H, W)
    bbox_min = obj_data["bbox_min"]  # shape: (3,)
    bbox_max = obj_data["bbox_max"]  # shape: (3,)

    # Assume voxel grid is binary occupancy
    occupied_indices = voxel_grid.nonzero(as_tuple=False)  # (N, 3)

    dims = voxel_grid.shape
    spacing = (bbox_max - bbox_min) / torch.tensor(dims).to(bbox_min)

    # Convert grid indices to world coordinates
    coords = bbox_min + spacing * occupied_indices  # (N, 3)
    return coords.cpu().numpy()


def convert_to_ply(obj3d_path, output_dir):
    obj = torch.load(obj3d_path)
    points_np = extract_voxel_coords(obj)

    # Create point cloud
    pcd = trimesh.points.PointCloud(points_np)

    # Save to PLY
    ply_path = output_dir / (obj3d_path.stem + ".ply")
    pcd.export(ply_path)
    print(f"✅ Saved: {ply_path}")


def main(input_dir, output_dir):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    obj3d_files = sorted(input_dir.glob("obj3Dseg*.pt"))
    if not obj3d_files:
        print("❌ No obj3Dseg*.pt files found!")
        return

    for obj_file in obj3d_files:
        convert_to_ply(obj_file, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert obj3Dseg voxel grid to .ply point cloud")
    parser.add_argument("--input", "-i", required=True, help="Directory with obj3Dseg*.pt files")
    parser.add_argument("--output", "-o", required=True, help="Directory to save .ply files")

    args = parser.parse_args()
    main(args.input, args.output)
