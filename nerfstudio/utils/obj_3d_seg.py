import itertools
import torch
import matplotlib.pyplot as plt
import numpy as np
import pycolmap

from lightglue.utils import rbd
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from nerfstudio.utils.img_utils import points2D_to_point_masks
from nerfstudio.utils.proj_utils import project_points
from pyquaternion import Quaternion
from scipy.ndimage import binary_dilation, generate_binary_structure
from skimage.measure import marching_cubes


class Object3DSeg:
    """
    Object 3D segmentation represented as binary voxel grid
    """
    def __init__(
        self, bbox_min, bbox_max, voxel, pose_change,
        tight_bbox=None, dilate=31
    ):
        """
        Args:
            bbox_min (3-tuple): Min corner of the object bounding box
            bbox_max (3-tuple): Max corner of the object bounding box
            voxel (N, M, K tensor): Binary voxel grid representing obj 3D seg
            pose_change (4x4 tensor): 6DoF object pose change
            tight_bbox (3-tuple-tuple): Tight bounding box for the object
            dilate (int): #Iters to dilate the 3D mask to get move-out region
        """
        assert voxel.dtype == torch.bool
        assert all([min_ < max_ for min_, max_ in zip(bbox_min, bbox_max)])
        assert pose_change.shape == (4, 4)
        self.bbox_min = torch.tensor(bbox_min).to(voxel.device).float()
        self.bbox_max = torch.tensor(bbox_max).to(voxel.device).float()
        self.dims = self.bbox_max - self.bbox_min
        self.voxel = voxel
        self.voxel_original = voxel.clone()
        self.tight_bbox = tight_bbox
        self.pose_change = pose_change
        # NOTE: We must dilate the object 3D segmentation to account for
        #       densities exceeding the object boundaries
        self.dilate = dilate
        self.voxel_dilated = self.dilate_uniform(dilate)
        # self.visualize("/home/ziqi/Desktop/test")
        # self.voxel = self.dilate_uniform(1)
        self.voxel = self.dilate_top(7)

    def get_bbox(self):
        """
        Get the **loose** 3D bbox for the object
        """
        return self.bbox_min, self.bbox_max
    
    def get_tight_bbox(self):
        """
        Get the **tight** 3D bbox that perfectly enclose the object
        """
        if self.tight_bbox is not None:
            return self.tight_bbox
        else:
            raise NotImplementedError("Need tight bbox computation")
        
    def get_pose_change(self):
        """
        Get the 6DoF pose change of the object
        """
        return self.pose_change
    
    def set_pose_change(self, pose_change):
        """
        Set the 6DoF pose change of the object

        Args:
            pose_change (4x4 tensor): 6DoF pose change
        """
        self.pose_change = pose_change

    def get_all_corners(self):
        """
        Get all corner of the loose bbox

        Returns:
            corners (8x3 tensor): 8 corners of the bbox
        """
        combined = torch.stack([self.bbox_min, self.bbox_max], dim=1)
        corners = torch.tensor(list(itertools.product(*combined)))
        return corners.to(self.voxel.device)

    def dilate_uniform(self, kernel_size=1):
        """
        Uniformly dilate the binary voxel grid

        Args:
            kernel_size (int): Size of the dilation kernel
        
        Returns:
            dilated_voxel (N, M, K tensor): Dilated voxel grid
        """
        voxel = self.voxel.cpu().numpy()
        voxel = binary_dilation(
            voxel, iterations=kernel_size,
            structure=generate_binary_structure(3, 3)
        )
        voxel = torch.from_numpy(voxel).to(self.voxel.device)
        return voxel

    def dilate_top(self, kernel_size=3):
        """
        Non-uniformly dilate the binary voxel grid, more on top, none at bottom

        Args:
            kernel_size (int): Size of the dilation kernel
        
        Returns:
            dilated_voxel (N, M, K tensor): Dilated voxel grid
        """
        voxel = self.voxel.cpu().numpy()
        struct = generate_binary_structure(3, 1)
        struct[..., 0:-1] = False
        struct[..., -1] = True
        struct[1, 1, 1] = True
        voxel = binary_dilation(
            voxel, iterations=kernel_size, structure=struct
        )
        voxel = torch.from_numpy(voxel).to(self.voxel.device)
        return voxel

    def query(self, points, dilate=False):
        """
        Query the object 3D seg at points to determine whether they are inside

        Args:
            points (..., 3): 3D query points
            dilate (bool): Whether to use dilated voxel grid for query

        Returns:
            inside (...,): Whether the points are inside the object
        """
        assert points.shape[-1] == 3
        in_bbox = (
            (points >= self.bbox_min) & (points <= self.bbox_max)
        ).all(dim=-1)
        points_in_bbox = points[in_bbox]
        if points_in_bbox.numel() == 0:
            return torch.zeros_like(in_bbox).bool()
        # Normalize the points to the bounding box
        points_in_bbox = (points_in_bbox - self.bbox_min) / self.dims * 2 - 1
        # Convert to voxel indices
        grid = points_in_bbox.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        grid = torch.flip(grid, dims=(-1,)) # grid_sample 3D uses zyx order
        # Trilinear interpolation to extract occupancy values
        voxel = self.voxel_dilated if dilate else self.voxel
        occupancy = torch.nn.functional.grid_sample(
            voxel.float().unsqueeze(0).unsqueeze(0), grid.float(),
            align_corners=True
        ).squeeze()
        # Occupancy check
        inside = torch.zeros_like(in_bbox).to(points.device)
        inside[in_bbox] = occupancy > 0.5
        return inside
    
    def save(self, output_dir):
        """
        Save the input of object 3D segmentation

        Args:
            output_dir (str): Directory to save the 3D object mask
        """
        data_to_save = {
            'bbox_min': self.bbox_min.cpu(), 'bbox_max': self.bbox_max.cpu(),
            'voxel': self.voxel_original.cpu(),
            'pose_change': self.pose_change.cpu(),
            'dilate': self.dilate
        }
        if self.tight_bbox is not None:
            data_to_save['tight_bbox'] = self.tight_bbox
        else:
            data_to_save['tight_bbox'] = None
        torch.save(data_to_save, f"{output_dir}/obj3Dseg.pt")

    @classmethod
    def read_from_file(cls, file_path, device='cuda'):
        """
        Create an Object3DSeg instance from a file

        Args:
            file_path (str): Path to mask_3D.pt file
        
        Returns:
            Object3DSeg: An instance of the Object3DSeg class
        """
        # Load the data from the file
        data = torch.load(file_path)
        # Create a new Object3DSeg instance with the loaded data
        if data['tight_bbox'] is not None:
            tight_bbox = data['tight_bbox']
        else:
            tight_bbox = None
        obj = cls(
            bbox_min=data['bbox_min'].tolist(),
            bbox_max=data['bbox_max'].tolist(),
            voxel=data['voxel'].to(device),
            tight_bbox=tight_bbox,
            pose_change=data['pose_change'].to(device),
            dilate=data['dilate']
        )
        print(f"Succesfully loaded the scene change estimates")
        print(f"Pose change:\n {obj.pose_change}")
        return obj

    def set_appearance_weights(self, app_weights):
        """
        Set the appearance latent for the object

        Args:
            app_latent (2, N, N, N, C tensor): Appearance weights
        """
        assert app_weights.shape[0] == 2
        assert len(app_weights.shape) == 5
        self.appearance_weights = app_weights
    
    def Fourier_reconstruct(self, weights, coords):
        """
        Use the following Fourier decomposition to reconstruct a 4D signal

        f(x, y, z) = \Sum_{i=1}^N \Sum_{j=1}^N \Sum_{k=1}^N
        a_{ijk} * cos[2\pi (i*x + j*y + k*z)] +
        b_{ijk} * sin[2\pi (i*x + j*y + k*z)]
        NOTE: This is for one channel in the C dimension)

        Args:
            weights (2, N, N, N, C): Weights for the cos and sin modes
            coords (..., 3): Coordinates to reconstruct the signal at

        Returns:
            signal (..., C): Reconstructed signal
        """
        assert coords.shape[-1] == 3
        device = coords.device
        # Only reconstruct for in bbox points
        in_bbox = (
            (coords >= self.bbox_min) & (coords <= self.bbox_max)
        ).all(dim=-1, keepdim=True)
        # Normalize the coordinates to the bounding box
        coords = (coords - self.bbox_min) / self.dims
        x, y, z = coords[..., 0], coords[..., 1], coords[..., 2]
        # Get the Fourier modes
        i = torch.arange(weights.size(1)).to(coords)
        j = torch.arange(weights.size(2)).to(coords)
        k = torch.arange(weights.size(3)).to(coords)
        i, j, k = torch.meshgrid(i, j, k, indexing='ij')
        weights = weights.to(coords)
        # Reconstruct the signal
        # TODO: Do this in chunks so the memo doesn't explode
        ixjykz = torch.einsum("ijk,...->...ijk", i, x) + \
            torch.einsum("ijk,...->...ijk", j, y) + \
            torch.einsum("ijk,...->...ijk", k, z)
        signal = torch.einsum(
            "ijkc,...ijk->...c", weights[0], torch.cos(2 * torch.pi * ixjykz)
        ) + torch.einsum(
            "ijkc,...ijk->...c", weights[1], torch.sin(2 * torch.pi * ixjykz)
        )
        # Zero out the signal for points outside the bbox
        signal = torch.where(
            in_bbox, signal, torch.zeros_like(signal).to(device)
        )
        return signal
    
    def visualize(self, output_dir):
        voxel = self.voxel_dilated.cpu().numpy()
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        xmin, ymin, zmin = self.bbox_min.cpu().numpy()
        xmax, ymax, zmax = self.bbox_max.cpu().numpy()
        # Define the 12 edges of the bounding box
        edges = [
            [(xmin, ymin, zmin), (xmax, ymin, zmin)],
            [(xmax, ymin, zmin), (xmax, ymax, zmin)],
            [(xmax, ymax, zmin), (xmin, ymax, zmin)],
            [(xmin, ymax, zmin), (xmin, ymin, zmin)],
            [(xmin, ymin, zmax), (xmax, ymin, zmax)],
            [(xmax, ymin, zmax), (xmax, ymax, zmax)],
            [(xmax, ymax, zmax), (xmin, ymax, zmax)],
            [(xmin, ymax, zmax), (xmin, ymin, zmax)],
            [(xmin, ymin, zmin), (xmin, ymin, zmax)],
            [(xmax, ymin, zmin), (xmax, ymin, zmax)],
            [(xmax, ymax, zmin), (xmax, ymax, zmax)],
            [(xmin, ymax, zmin), (xmin, ymax, zmax)]
        ]
        for edge in edges:
            ax.plot3D(*zip(*edge), color='r')
        # Create a mesh grid for the voxel positions
        x = np.linspace(xmin, xmax, voxel.shape[0]+1)
        y = np.linspace(ymin, ymax, voxel.shape[1]+1)
        z = np.linspace(zmin, zmax, voxel.shape[2]+1)
        x, y, z = np.meshgrid(x, y, z, indexing='ij')
        verts, faces, _, _ = marching_cubes(voxel, 0.5)
        x_scale = (xmax - xmin) / voxel.shape[0]
        y_scale = (ymax - ymin) / voxel.shape[1]
        z_scale = (zmax - zmin) / voxel.shape[2]
        verts[:, 0] = verts[:, 0] * x_scale + xmin
        verts[:, 1] = verts[:, 1] * y_scale + ymin
        verts[:, 2] = verts[:, 2] * z_scale + zmin
        mesh = Poly3DCollection(verts[faces])
        mesh.set_edgecolor('b')
        ax.add_collection3d(mesh)
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        plt.savefig(f"{output_dir}/occ_grid.png")


class Obj3DFeats:
    """
    Object's multiview SuperPoint features
    """
    def __init__(self, feats=[], pts3D=[]):
        """
        Args:
            feats (dict): SuperPoint feature dict
        """
        self.feats = feats
        self.pts3D = pts3D
        assert len(feats) == len(pts3D)
        assert all(
            [feats['keypoints'].shape[1] == pts.shape[0] 
            for feats, pts in zip(feats, pts3D)]
        )
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    
    def add_feats(self, feats, pts3D):
        """
        Add features to the object

        Args:
            feats (dict): SuperPoint feature dict from a view
            pts3D (Nx3 tensor): 3D points corresponding to the features
        """
        assert feats['keypoints'].shape[1] == pts3D.shape[0]
        self.feats.append(feats)
        self.pts3D.append(pts3D)

    def match(self, feats, matcher=None):
        """
        Match features with the object features

        Args:
            feats (dict): SuperPoint feature dict for 2D image points
            matcher (LightGlue.Matcher): Feature matcher
        
        Returns:
            matched3D (Nx3): Matched 3D points
            matched2D (Nx2): Matched 2D points
        """
        assert len(self.feats) > 0, "No features to match"
        # TODO: have to take matcher as input
        if matcher is None:
            from lightglue import LightGlue            
            matcher = LightGlue(features='superpoint').eval().to(self.device)
        matched_pts3D, matches = [], []
        for obj_feats, obj_pts3D in zip(self.feats, self.pts3D): 
            mm = matcher({'image0': obj_feats, 'image1': feats})
            _, _, mm = [rbd(x) for x in [obj_feats, feats, mm]]
            matched3D = obj_pts3D[mm["matches"][:, 0]]
            matches.append(mm["matches"][:, 1])
            matched_pts3D.append(matched3D)
        matches = torch.cat(matches, dim=0)
        matched_pts3D = torch.cat(matched_pts3D, dim=0)
        # TODO: handle duplicate matches by comparing 
        matched_pts2D = feats["keypoints"][0][matches]
        return matched_pts2D, matched_pts3D
    
    def PnP(self, feats, K, H, W, matcher=None, verbose=False):
        """
        Solve PnP problem to estimate cam-to-obj pose

        Args:
            feats (dict): SuperPoint feature dict
            K (3x3): Camera intrinsics
            matcher (LightGlue.Matcher): Feature matcher
        
        Returns:
            pose (4x4 tensor): Camera-to-obj pose
            num_inliers (int): Number of inliers
            num_matches (int): Number of matches
        """

        assert K.shape == (3, 3)
        matched_pts2D, matched_pts3D = self.match(feats, matcher)
        matched_pts2D = matched_pts2D.cpu().numpy()
        matched_pts3D = matched_pts3D.cpu().numpy()
        if matched_pts3D.shape[0] < 4:
            print("Warn: Not enough points for PnP")
            return None, 0.0, 0
        pycolmap_cam = pycolmap.Camera(
            model='OPENCV', width=W, height=H,
            params=[K[0, 0], K[1, 1], K[0, -1], K[1, -1], 0, 0, 0, 0]
        )
        ret = pycolmap.absolute_pose_estimation(
            matched_pts2D, matched_pts3D, pycolmap_cam,
            estimation_options={'ransac': {'max_error': 12.0}}, 
            refinement_options={'print_summary': verbose}
        )
        if not ret['success']:
            print("Warn: PnP failed!!")
            return None, 0.0, 0
        R_mat = torch.tensor(Quaternion(*ret['qvec']).rotation_matrix)
        tvec = torch.tensor(ret['tvec'])
        pose = torch.eye(4, device=K.device)
        pose[:3, :3], pose[:3, 3] = R_mat, tvec
        pose = pose.inverse()
        if verbose:
            print(
                f"Number of inliers: {ret['num_inliers']}/{len(matched_pts2D)}"
            )
            print(f"pose est:\n {pose}")
        return pose, ret["num_inliers"], len(matched_pts2D)