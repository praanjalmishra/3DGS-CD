import matplotlib.pyplot as plt
import numpy as np
import os

from matplotlib import patches
from mpl_toolkits.mplot3d import Axes3D
from lightglue import viz2d
from PIL import Image
from tqdm import tqdm


def debug_point_prompts(images, points, debug_dir):
    """
    Debugging point prompts for the SAM model

    Args:
        images (Nx3xHxW): Images
        points (NxKx2): 2D points
        debug_dir (str): Directory to save the images
    """
    assert images.shape[0] == points.shape[0]
    assert points.shape[-1] == 2
    assert os.path.isdir(debug_dir)
    target_images = images.permute(0, 2, 3, 1).cpu().numpy()
    target_points_np = points.cpu().numpy()
    for i, img in enumerate(target_images):
        plt.imshow(img)
        plt.scatter(
            target_points_np[i, :, 0], target_points_np[i, :, 1],
            c='r', marker='o'
        )
        plt.title(f"Target View {i}")
        plt.axis("off")
        plt.savefig(f"{debug_dir}/debug_{i}.png")
        plt.close()


def debug_bbox_prompts(images, bboxes, debug_dir):
    """
    Debugging bounding box prompts for the SAM model

    Args:
        images (Nx3xHxW): Images
        bboxes (Nx4): Bounding boxes
        debug_dir (str): Directory to save the images
    """
    assert images.shape[0] == bboxes.shape[0]
    assert os.path.isdir(debug_dir)
    target_images = images.permute(0, 2, 3, 1).cpu().numpy()
    bboxes_np = bboxes.cpu().numpy()
    for i, img in enumerate(target_images):
        plt.imshow(img)
        bbox = bboxes_np[i]
        rect = patches.Rectangle(
            (bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1],
            linewidth=1, edgecolor='r', facecolor='none'
        )
        plt.gca().add_patch(rect)
        plt.title(f"Target View {i}")
        plt.axis("off")
        plt.savefig(f"{debug_dir}/debug_{i}.png")
        plt.close()


def debug_point_cloud(point_cloud, debug_dir):
    """
    Save point cloud as a 3D plot

    Args:
        point_cloud (Nx3): Point cloud
        debug_dir (str): Directory to save the images
    """
    assert len(point_cloud.shape) == 2
    assert point_cloud.shape[-1] == 3
    assert os.path.isdir(debug_dir)
    # visualize the point cloud
    plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(
        point_cloud.cpu()[:, 0], point_cloud.cpu()[:, 1],
        point_cloud.cpu()[:, 2], c='r', marker='o'
    )
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.savefig(f"{debug_dir}/point_cloud.png")
    plt.close()


def debug_matches(imgs1, imgs2, kps1, kps2, matches, debug_dir):
    assert os.path.isdir(debug_dir)
    for i, (img1, img2, kp1, kp2, match) in \
        enumerate(zip(imgs1, imgs2, kps1, kps2, matches)):
        mkp1, mkp2 = kp1[match[..., 0]], kp2[match[..., 1]]
        viz2d.plot_images([img1.cpu(), img2.cpu()])
        viz2d.plot_matches(mkp1, mkp2, color="lime", lw=0.2)
        viz2d.save_plot(f"{debug_dir}/debug_{i}.png")
        plt.close()


def debug_image_pairs(imgs1, imgs2, debug_dir):
    """
    Compare two set of images side by side
    """
    assert len(imgs1.shape) == len(imgs2.shape) == 4
    assert imgs1.shape[1] == imgs2.shape[1] == 3
    assert os.path.isdir(debug_dir)
    for idx, (img1, img2) in enumerate(zip(imgs1, imgs2)):
        viz2d.plot_images([img1.cpu(), img2.cpu()])
        viz2d.save_plot(f"{debug_dir}/debug_{idx}.png")
        plt.close()


def debug_images(imgs, debug_dir):
    """
    Save torch images as images

    Args:
        imgs (Nx3xHxW): Images
        debug_dir (str): Directory to save the images
    """
    assert os.path.isdir(debug_dir)
    assert len(imgs.shape) == 4 and imgs.shape[1] == 3
    for i, img in enumerate(imgs):
        img = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        Image.fromarray(img).save(f"{debug_dir}/img_{i}.png")


def debug_masks(masks, debug_dir):
    """
    Save torch masks as images

    Args:
        masks (Nx1xHxW): Binary masks
        debug_dir (str): Directory to save the images
    """
    assert os.path.isdir(debug_dir)
    assert len(masks.shape) == 4 and masks.shape[1] == 1
    for i, mask in enumerate(tqdm(masks, desc="Save masks")):
        mask = (mask.squeeze() * 255).byte().cpu().numpy()
        Image.fromarray(mask).save(f"{debug_dir}/mask_{i}.png")


def debug_depths(depths, debug_dir):
    """
    Save torch depths as images

    Args:
        depths (Nx1xHxW): Depth maps
        debug_dir (str): Directory to save the images
    """
    assert os.path.isdir(debug_dir)
    assert len(depths.shape) == 4 and depths.shape[1] == 1
    for i, depth in enumerate(tqdm(depths, desc="Save depths")):
        depth = depth / depth.max()
        depth = (depth.squeeze() * 255).byte().cpu().numpy()
        Image.fromarray(depth).save(f"{debug_dir}/depth_{i}.png")