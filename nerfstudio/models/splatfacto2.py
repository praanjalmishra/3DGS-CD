# ruff: noqa: E741
# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
NeRF implementation that combines many recent advancements.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Type, Union

import numpy as np
import torch
from gsplat._torch_impl import quat_to_rotmat
from gsplat.project_gaussians import project_gaussians
from gsplat.rasterize import rasterize_gaussians
from gsplat.sh import num_sh_bases, spherical_harmonics
from pytorch_msssim import SSIM
from torch.nn import Parameter
from typing_extensions import Literal

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.cameras.camera_optimizers import CameraOptimizer, CameraOptimizerConfig
from nerfstudio.data.scene_box import OrientedBox
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes, TrainingCallbackLocation
from nerfstudio.engine.optimizers import Optimizers

# need following import for background color override
from nerfstudio.model_components import renderers
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils.colors import get_color
from nerfstudio.utils.gauss_utils import (
    get_gaussian_endpts, transform_gaussians, rot2quat, sample_gaussians,
    fit_gaussian_batch
)
from nerfstudio.utils.rich_utils import CONSOLE
from nerfstudio.utils.obj_3d_seg import Object3DSeg
from nerfstudio.utils.io import read_transforms


def cube_mask(
    pos, corner1=(1.5 / 11.0, 1.5 / 11.0, -0.5 / 11.0),
    corner2=(-1.5 / 11.0, -1.5 / 11.0, 2.5 / 11.0)
):
    '''
    3D cubic mask function from 2 opposite cube corners
    Args:
        pos: (..., 3)
        corner1: one corner of the cube
        corner2: another corner of the cube

    Returns:
        mask: Boolean tensor of shape (...), where True means in the cube
    '''
    assert len(corner1) == len(corner2) == 3
    if not type(corner1) == type(corner2) == torch.Tensor:
        corner1 = torch.from_numpy(np.array(corner1))
        corner2 = torch.from_numpy(np.array(corner2))
    corner1 = corner1.to(pos.device)
    corner2 = corner2.to(pos.device)
    min_corner = torch.min(corner1, corner2)
    max_corner = torch.max(corner1, corner2)
    mask = (pos >= min_corner) & (pos <= max_corner)
    return mask.all(dim=-1)

def generate_random_points_in_mask(f, corner1, corner2, num_points):
    xmin, xmax = min(corner1[0], corner2[0]), max(corner1[0], corner2[0])
    ymin, ymax = min(corner1[1], corner2[1]), max(corner1[1], corner2[1])
    zmin, zmax = min(corner1[2], corner2[2]), max(corner1[2], corner2[2])
    points = []
    while len(points) < num_points:
        # Randomly sample points within the bounds
        x = torch.rand(num_points) * (xmax - xmin) + xmin
        y = torch.rand(num_points) * (ymax - ymin) + ymin
        z = torch.zeros_like(x) # + 0.05 * torch.rand_like(x)
        point = torch.stack((x, y, z), dim=1)
        # Evaluate the mask function
        mask = f(point)
        # Collect valid points
        valid_points = point[mask]
        points.extend(valid_points.tolist())
        # If we have enough points, stop
        if len(points) >= num_points:
            points = points[:num_points]
            break
    return torch.tensor(points)

def get_points_color(points, images, poses, Ks, H, W):
    from nerfstudio.utils.proj_utils import project_points
    dist_coeffs = torch.zeros(Ks.shape[0], 4).to(points)
    pts2d, valid = project_points(points, poses, Ks, dist_coeffs, H, W)
    pts2d = pts2d.long() # N, M, 2
    colors = []
    for i in range(images.shape[0]):
        img = images[i].permute(1, 2, 0)
        color = img[pts2d[i, :, 1], pts2d[i, :, 0]]
        colors.append(color)
    return torch.stack(colors).mean(dim=0)


def k_nearest_sklearn(x: torch.Tensor, k: int):
    """
        Find k-nearest neighbors using sklearn's NearestNeighbors.
    x: The data tensor of shape [num_samples, num_features]
    k: The number of neighbors to retrieve
    """
    # Convert tensor to numpy array
    x_np = x.cpu().numpy()

    # Build the nearest neighbors model
    from sklearn.neighbors import NearestNeighbors

    nn_model = NearestNeighbors(
        n_neighbors=k + 1, algorithm="auto", metric="euclidean"
    ).fit(x_np)

    # Find the k-nearest neighbors
    distances, indices = nn_model.kneighbors(x_np)

    # Exclude the point itself from the result and return
    return distances[:, 1:].astype(np.float32), indices[:, 1:].astype(np.float32)


def random_quat_tensor(N):
    """
    Defines a random quaternion tensor of shape (N, 4)
    """
    u = torch.rand(N)
    v = torch.rand(N)
    w = torch.rand(N)
    return torch.stack(
        [
            torch.sqrt(1 - u) * torch.sin(2 * math.pi * v),
            torch.sqrt(1 - u) * torch.cos(2 * math.pi * v),
            torch.sqrt(u) * torch.sin(2 * math.pi * w),
            torch.sqrt(u) * torch.cos(2 * math.pi * w),
        ],
        dim=-1,
    )


def RGB2SH(rgb):
    """
    Converts from RGB values [0,1] to the 0th spherical harmonic coefficient
    """
    C0 = 0.28209479177387814
    return (rgb - 0.5) / C0


def SH2RGB(sh):
    """
    Converts from the 0th spherical harmonic coefficient to RGB values [0,1]
    """
    C0 = 0.28209479177387814
    return sh * C0 + 0.5



@dataclass
class Splatfacto2ModelConfig(ModelConfig):
    """Splatfacto Model Config, nerfstudio's implementation of Gaussian Splatting"""

    _target: Type = field(default_factory=lambda: Splatfacto2Model)
    warmup_length: int = 500
    """period of steps where refinement is turned off"""
    refine_every: int = 100
    """period of steps where gaussians are culled and densified"""
    resolution_schedule: int = 3000
    """training starts at 1/d resolution, every n steps this is doubled"""
    background_color: Literal["random", "black", "white"] = "random"
    """Whether to randomize the background color."""
    num_downscales: int = 2
    """at the beginning, resolution is 1/2^d, where d is this number"""
    cull_alpha_thresh: float = 0.1
    """threshold of opacity for culling gaussians. One can set it to a lower value (e.g. 0.005) for higher quality."""
    cull_scale_thresh: float = 0.5
    """threshold of scale for culling huge gaussians"""
    continue_cull_post_densification: bool = True
    """If True, continue to cull gaussians post refinement"""
    reset_alpha_every: int = 30
    """Every this many refinement steps, reset the alpha"""
    densify_grad_thresh: float = 0.0002
    """threshold of positional gradient norm for densifying gaussians"""
    densify_size_thresh: float = 0.01
    """below this size, gaussians are *duplicated*, otherwise split"""
    n_split_samples: int = 2
    """number of samples to split gaussians into"""
    sh_degree_interval: int = 1000
    """every n intervals turn on another sh degree"""
    cull_screen_size: float = 0.15
    """if a gaussian is more than this percent of screen space, cull it"""
    split_screen_size: float = 0.05
    """if a gaussian is more than this percent of screen space, split it"""
    stop_screen_size_at: int = 4000
    """stop culling/splitting at this step WRT screen size of gaussians"""
    random_init: bool = False
    """whether to initialize the positions uniformly randomly (not SFM points)"""
    num_random: int = 50000
    """Number of gaussians to initialize if random init is used"""
    random_scale: float = 10.0
    "Size of the cube to initialize random gaussians within"
    ssim_lambda: float = 0.2
    """weight of ssim loss"""
    stop_split_at: int = 15000
    """stop splitting at this step"""
    sh_degree: int = 3
    """maximum degree of spherical harmonics to use"""
    use_scale_regularization: bool = False
    """If enabled, a scale regularization introduced in PhysGauss (https://xpandora.github.io/PhysGaussian/) is used for reducing huge spikey gaussians."""
    max_gauss_ratio: float = 10.0
    """threshold of ratio of gaussian max to min scale before applying regularization
    loss from the PhysGaussian paper
    """
    output_depth_during_training: bool = False
    """If True, output depth during training. Otherwise, only output depth during evaluation."""
    rasterize_mode: Literal["classic", "antialiased"] = "classic"
    """
    Classic mode of rendering will use the EWA volume splatting with a [0.3, 0.3] screen space blurring kernel. This
    approach is however not suitable to render tiny gaussians at higher or lower resolution than the captured, which
    results "aliasing-like" artifacts. The antialiased mode overcomes this limitation by calculating compensation factors
    and apply them to the opacities of gaussians to preserve the total integrated density of splats.

    However, PLY exported with antialiased rasterize mode is not compatible with classic mode. Thus many web viewers that
    were implemented for classic mode can not render antialiased mode PLY properly without modifications.
    """
    camera_optimizer: CameraOptimizerConfig = field(default_factory=lambda: CameraOptimizerConfig(mode="off"))
    """Config of the camera optimizer to use"""


class Splatfacto2Model(Model):
    """Nerfstudio's implementation of Gaussian Splatting

    Args:
        config: Splatfacto configuration to instantiate model
    """

    config: Splatfacto2ModelConfig

    def __init__(
        self,
        *args,
        seed_points: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ):
        self.seed_points = seed_points
        super().__init__(*args, **kwargs)

    def populate_modules(self):
        # initialize the gaussian params with empty tensors
        means = torch.empty((0, 3)).float().cuda()
        scales = torch.empty((0, 3)).float().cuda()
        quats = torch.empty((0, 4)).float().cuda()
        dim_sh = num_sh_bases(self.config.sh_degree)
        features_dc = torch.empty((0, 3)).float().cuda()
        features_rest = torch.empty((0, dim_sh-1, 3)).float().cuda()
        opacities = torch.empty((0, 1)).float().cuda()
        self.gauss_params = torch.nn.ParameterDict(
            {
                "means": means, "scales": scales, "quats": quats,
                "features_dc": features_dc, "features_rest": features_rest,
                "opacities": opacities,
            }
        )
        # initialize the camera optimizer
        if hasattr(self.config, "camera_optimizer"):
            self.camera_optimizer: CameraOptimizer = self.config.camera_optimizer.setup(
                num_cameras=self.num_train_data, device="cpu"
            )

        # metrics
        from torchmetrics.image import PeakSignalNoiseRatio
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = SSIM(data_range=1.0, size_average=True, channel=3)
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)
        self.step = 0

        self.crop_box: Optional[OrientedBox] = None
        if self.config.background_color == "random":
            self.background_color = torch.tensor(
                [0.1490, 0.1647, 0.2157]
            )  # This color is the same as the default background color in Viser. This would only affect the background color when rendering.
        else:
            self.background_color = get_color(self.config.background_color)

        self.xys_grad_norm = None
        self.max_2Dsize = None

    @property
    def colors(self):
        if self.config.sh_degree > 0:
            return SH2RGB(self.features_dc)
        else:
            return torch.sigmoid(self.features_dc)

    @property
    def shs_0(self):
        return self.features_dc

    @property
    def shs_rest(self):
        return self.features_rest

    @property
    def num_points(self):
        return self.means.shape[0]

    @property
    def means(self):
        return self.gauss_params["means"]

    @property
    def scales(self):
        return self.gauss_params["scales"]

    @property
    def quats(self):
        return self.gauss_params["quats"]

    @property
    def features_dc(self):
        return self.gauss_params["features_dc"]

    @property
    def features_rest(self):
        return self.gauss_params["features_rest"]

    @property
    def opacities(self):
        return self.gauss_params["opacities"]

    def load_state_dict(self, dict, **kwargs):  # type: ignore
        # resize the parameters to match the new number of points
        self.step = 30000
        param_names = [
            "means", "scales", "quats",
            "features_dc", "features_rest", "opacities"
        ]
        # For backwards compatibility, we remap the names of parameters from
        # means->gauss_params.means since old checkpoints have that format
        if "means" in dict:
            for p in param_names:
                dict[f"gauss_params.{p}"] = dict[p]
        if self.training:
            # Read object 3D segmentation mask
            assert hasattr(self, "obj_3d_seg")
            # self.obj_3d_seg = Object3DSeg.read_from_file(
            #     self.config.obj_mask_file, device="cuda"
            # )
            # corner1 = (1.8 / 11.0, 1.8 / 11.0, 2.02 / 11.0)
            # corner2 = (-1.8 / 11.0, -1.8 / 11.0, -2.02 / 11.0)
            # obj_mask = cube_mask(
            #     dict["gauss_params.means"], corner1=corner1, corner2=corner2
            # )

            obj_mask = self.obj_3d_seg.query(
               dict["gauss_params.means"].cuda()
            ).cpu()
            if dict["gauss_params.quats"][obj_mask].shape[0] > 0:
                (
                    dict["gauss_params.means"][obj_mask], 
                    dict["gauss_params.quats"][obj_mask]
                ) = transform_gaussians(
                    self.obj_3d_seg.pose_change.cpu(),
                    dict["gauss_params.means"][obj_mask],
                    dict["gauss_params.quats"][obj_mask]
                )

            # for p in param_names:
            #     dict[f"gauss_params.{p}"] = dict[f"gauss_params.{p}"][~obj_mask]

            # corner1 = (2.5 / 11.0, 2.5 / 11.0, 2.02 / 11.0)
            # corner2 = (-2.5 / 11.0, -2.5 / 11.0, -2.02 / 11.0)
            # obj_mask_dilate = cube_mask(
            #     dict["gauss_params.means"], corner1=corner1, corner2=corner2
            # )
            obj_mask_dilate = self.obj_3d_seg.query(
                dict["gauss_params.means"].cuda(), dilate=True
            ).cpu()
            # Duplicate the gaussians in the dilated object 3D mask
            dict_new = {}
            for p in param_names:
                dict_new[f"gauss_params.{p}"] = \
                    dict[f"gauss_params.{p}"][obj_mask_dilate]

            # Load pre-trained 3DGS and fix the gaussian params
            self.gauss_params_fixed = {}
            for name, param in self.gauss_params.items():
                self.gauss_params_fixed[name] = \
                    dict[f"gauss_params.{name}"].to(self.device)
            # Load the Gaussians to finetune
            newp = dict_new["gauss_params.means"].shape[0]
            for name, param in self.gauss_params.items():
                old_shape = param.shape
                new_shape = (newp,) + old_shape[1:]
                self.gauss_params[name] = torch.nn.Parameter(
                    torch.zeros(new_shape, device=self.device)
                )
            super().load_state_dict(dict_new, **kwargs)
        else:
            newp = dict["gauss_params.means"].shape[0]
            for name, param in self.gauss_params.items():
                old_shape = param.shape
                new_shape = (newp,) + old_shape[1:]
                self.gauss_params[name] = torch.nn.Parameter(
                    torch.zeros(new_shape, device=self.device)
                )
            super().load_state_dict(dict, **kwargs)

    def state_dict(self, **kwargs):
        # Also save the fixed Gaussian params when save checkpoints
        dict = super().state_dict(**kwargs)
        param_names = [
            "means", "scales", "quats", "features_dc", "features_rest",
            "opacities"
        ]
        for p in param_names:
            dict[f"_model.gauss_params.{p}"] = torch.cat(
                [dict[f"_model.gauss_params.{p}"], self.gauss_params_fixed[p]],
                dim=0
            )
        return dict

    def remove_gaussians_in_mask(self, mask, optimizers):
        """
        Remove Gaussians within the 3D mask

        Args:
            mask (..., 3 --> ...): 3D mask function, True means out of object
            optimizers: Optimizers
        """
        gauss_mask = mask(self.means)
        for name, param in self.gauss_params.items():
            self.gauss_params[name] = torch.nn.Parameter(param[~gauss_mask])
        self.remove_from_all_optim(optimizers, gauss_mask)

    def cut_gaussians_in_mask(self, mask, optimizers):
        """
        Cut Gaussians to smaller ones within the 3D mask

        Args:
            mask (..., 3 --> ...): 3D mask function, True means out of object
            optimizers: Optimizers
        """
        gauss_mask = mask(self.means) # gaussians to cut
        # Sample points from the out-of-mask gaussians
        samples = sample_gaussians(
            self.means[gauss_mask], self.scales.exp()[gauss_mask],
            self.quats[gauss_mask], 500
        )
        # how many sampled points are in mask
        in_mask = ~mask(samples)
        # We only cut the gaussians that have enough points in the mask
        enough_in_mask = in_mask.sum(dim=-1) > 50
        # Recontruct gaussians from the in-mask sampled points
        means_rec, cov_rec = fit_gaussian_batch(
            samples[enough_in_mask], in_mask[enough_in_mask]
        )
        # Extract the eigenvalues and eigenvectors from cov3d
        eigenvalues, eigenvectors = torch.linalg.eigh(cov_rec, UPLO='L')
        # Ensure positive semidefiniteness
        eigenvalues = torch.clamp(eigenvalues, min=1e-15)
        # Get the scales and quats
        scales_rec = torch.log(torch.sqrt(eigenvalues))
        quats_rec = rot2quat(eigenvectors)
        # Get the rest of the features
        # TODO: recompute the opacities
        opac_rec = self.opacities[gauss_mask][enough_in_mask]
        feat_dc_rec = self.features_dc[gauss_mask][enough_in_mask]
        feat_rest_rec = self.features_rest[gauss_mask][enough_in_mask]
        # Cutted gaussians
        gaussians_rec = {
            "means": means_rec, "scales": scales_rec, "quats": quats_rec,
            "features_dc": feat_dc_rec, "features_rest": feat_rest_rec,
            "opacities": opac_rec
        }
        # Remove out-of-mask gaussians
        for name, param in self.gauss_params.items():
            self.gauss_params[name] = torch.nn.Parameter(param[~gauss_mask])
        self.remove_from_all_optim(optimizers, gauss_mask)
        # Add the cutted gaussians
        for name, param in self.gauss_params.items():
            self.gauss_params[name] = torch.nn.Parameter(
                torch.cat([param.detach(), gaussians_rec[name]], dim=0)
            )
        # Initiliaze the optimizer states for cutted gaussians
        param_groups = self.get_gaussian_param_groups()
        for group, gauss_param in param_groups.items():
            optimizer = optimizers.optimizers[group]
            param = optimizer.param_groups[0]["params"][0]
            param_state = optimizer.state[param]
            num_cutted = means_rec.shape[0]
            exp_avg = param_state["exp_avg"]
            exp_avg_sq = param_state["exp_avg_sq"]
            param_state["exp_avg"] = torch.cat(
                [
                    exp_avg,
                    torch.zeros(num_cutted, *exp_avg.shape[1:]).to(exp_avg)
                ],
                dim=0,
            )
            param_state["exp_avg_sq"] = torch.cat(
                [
                    exp_avg_sq,
                    torch.zeros(num_cutted, *exp_avg_sq.shape[1:]).to(exp_avg)
                ],
                dim=0,
            )
            del optimizer.state[param]
            optimizer.state[gauss_param[0]] = param_state
            optimizer.param_groups[0]["params"] = gauss_param
            del param

    def remove_from_optim(self, optimizer, deleted_mask, new_params):
        """removes the deleted_mask from the optimizer provided"""
        assert len(new_params) == 1
        # assert isinstance(optimizer, torch.optim.Adam), "Only works with Adam"

        param = optimizer.param_groups[0]["params"][0]
        param_state = optimizer.state[param]
        del optimizer.state[param]

        # Modify the state directly without deleting and reassigning.
        if "exp_avg" in param_state:
            param_state["exp_avg"] = param_state["exp_avg"][~deleted_mask]
            param_state["exp_avg_sq"] = param_state["exp_avg_sq"][~deleted_mask]

        # Update the parameter in the optimizer's param group.
        del optimizer.param_groups[0]["params"][0]
        del optimizer.param_groups[0]["params"]
        optimizer.param_groups[0]["params"] = new_params
        optimizer.state[new_params[0]] = param_state

    def remove_from_all_optim(self, optimizers, deleted_mask):
        param_groups = self.get_gaussian_param_groups()
        for group, param in param_groups.items():
            self.remove_from_optim(optimizers.optimizers[group], deleted_mask, param)
        torch.cuda.empty_cache()

    def dup_in_optim(self, optimizer, dup_mask, new_params, n=2):
        """adds the parameters to the optimizer"""
        param = optimizer.param_groups[0]["params"][0]
        param_state = optimizer.state[param]
        if "exp_avg" in param_state:
            repeat_dims = (n,) + tuple(1 for _ in range(param_state["exp_avg"].dim() - 1))
            param_state["exp_avg"] = torch.cat(
                [
                    param_state["exp_avg"],
                    torch.zeros_like(param_state["exp_avg"][dup_mask.squeeze()]).repeat(*repeat_dims),
                ],
                dim=0,
            )
            param_state["exp_avg_sq"] = torch.cat(
                [
                    param_state["exp_avg_sq"],
                    torch.zeros_like(param_state["exp_avg_sq"][dup_mask.squeeze()]).repeat(*repeat_dims),
                ],
                dim=0,
            )
        del optimizer.state[param]
        optimizer.state[new_params[0]] = param_state
        optimizer.param_groups[0]["params"] = new_params
        del param

    def dup_in_all_optim(self, optimizers, dup_mask, n):
        param_groups = self.get_gaussian_param_groups()
        for group, param in param_groups.items():
            self.dup_in_optim(optimizers.optimizers[group], dup_mask, param, n)

    def after_train(self, optimizers, step: int):
        assert step == self.step

        # to save some training time, we no longer need to update those stats post refinement
        if self.step >= self.config.stop_split_at:
            return
        with torch.no_grad():
            # keep track of a moving average of grad norms
            radii = self.radii[:self.num_points]
            visible_mask = (radii > 0).flatten()
            assert self.xys.grad is not None
            grads = self.xys.grad[:self.num_points].detach().norm(dim=-1)
            # print(f"grad norm min {grads.min().item()} max {grads.max().item()} mean {grads.mean().item()} size {grads.shape}")
            if self.xys_grad_norm is None:
                self.xys_grad_norm = grads
                self.vis_counts = torch.ones_like(self.xys_grad_norm)
            else:
                assert self.vis_counts is not None
                self.vis_counts[visible_mask] = self.vis_counts[visible_mask] + 1
                self.xys_grad_norm[visible_mask] = grads[visible_mask] + self.xys_grad_norm[visible_mask]

            # update the max screen size, as a ratio of number of pixels
            if self.max_2Dsize is None:
                self.max_2Dsize = torch.zeros_like(radii, dtype=torch.float32)
            newradii = radii.detach()[visible_mask]
            self.max_2Dsize[visible_mask] = torch.maximum(
                self.max_2Dsize[visible_mask],
                newradii / float(max(self.last_size[0], self.last_size[1])),
            )
            if self.step % 100 == 0:
                avg_grad_norm = (self.xys_grad_norm / self.vis_counts) * 0.5 * max(self.last_size[0], self.last_size[1])
                high_grads = (avg_grad_norm > self.config.densify_grad_thresh).squeeze()
                splits = (self.scales.exp().max(dim=-1).values > self.config.densify_size_thresh).squeeze()
                if self.step < self.config.stop_screen_size_at:
                    splits |= (self.max_2Dsize > self.config.split_screen_size).squeeze()
                splits &= high_grads
                nsamps = self.config.n_split_samples
                split_params = self.split_gaussians(splits, nsamps)
                dups = (self.scales.exp().max(dim=-1).values <= self.config.densify_size_thresh).squeeze()
                dups &= high_grads
                dup_params = self.dup_gaussians(dups)
                for name, param in self.gauss_params.items():
                    self.gauss_params[name] = torch.nn.Parameter(
                        torch.cat([param.detach(), split_params[name], dup_params[name]], dim=0)
                    )
                # append zeros to the max_2Dsize tensor
                self.max_2Dsize = torch.cat(
                    [
                        self.max_2Dsize,
                        torch.zeros_like(split_params["scales"][:, 0]),
                        torch.zeros_like(dup_params["scales"][:, 0]),
                    ],
                    dim=0,
                )

                split_idcs = torch.where(splits)[0]
                self.dup_in_all_optim(optimizers, split_idcs, nsamps)

                dup_idcs = torch.where(dups)[0]
                self.dup_in_all_optim(optimizers, dup_idcs, 1)

                # After a guassian is split into two new gaussians, the original one should also be pruned.
                splits_mask = torch.cat(
                    (
                        splits,
                        torch.zeros(
                            nsamps * splits.sum() + dups.sum(),
                            device=self.device,
                            dtype=torch.bool,
                        ),
                    )
                )

                deleted_mask = self.cull_gaussians(splits_mask)
                self.remove_from_all_optim(optimizers, deleted_mask)
                self.xys_grad_norm = None
                self.vis_counts = None
                self.max_2Dsize = None
            if self.step >= 30100 and self.step % 100 == 0:
                self.remove_gaussians_in_mask(
                    lambda x: ~self.obj_3d_seg.query(x), optimizers
                )
                self.xys_grad_norm = None
                self.vis_counts = None
                self.max_2Dsize = None
            if self.step % 501 == 0:
                reset_value = self.config.cull_alpha_thresh * 2.0
                self.opacities.data = torch.clamp(
                    self.opacities.data,
                    max=torch.logit(torch.tensor(reset_value, device=self.device)).item(),
                )
                # reset the exp of optimizer
                optim = optimizers.optimizers["opacities"]
                param = optim.param_groups[0]["params"][0]
                param_state = optim.state[param]
                param_state["exp_avg"] = torch.zeros_like(param_state["exp_avg"])
                param_state["exp_avg_sq"] = torch.zeros_like(param_state["exp_avg_sq"])
                self.xys_grad_norm = None
                self.vis_counts = None
                self.max_2Dsize = None


    def set_crop(self, crop_box: Optional[OrientedBox]):
        self.crop_box = crop_box

    def set_background(self, background_color: torch.Tensor):
        assert background_color.shape == (3,)
        self.background_color = background_color

    def cull_gaussians(self, extra_cull_mask: Optional[torch.Tensor] = None):
        """
        This function deletes gaussians with under a certain opacity threshold
        extra_cull_mask: a mask indicates extra gaussians to cull besides existing culling criterion
        """
        n_bef = self.num_points
        # cull transparent ones
        culls = (torch.sigmoid(self.opacities) < self.config.cull_alpha_thresh).squeeze()
        below_alpha_count = torch.sum(culls).item()
        toobigs_count = 0
        if extra_cull_mask is not None:
            culls = culls | extra_cull_mask
        if self.step > self.config.refine_every * self.config.reset_alpha_every:
            # cull huge ones
            toobigs = (torch.exp(self.scales).max(dim=-1).values > self.config.cull_scale_thresh).squeeze()
            if self.step < self.config.stop_screen_size_at:
                # cull big screen space
                assert self.max_2Dsize is not None
                toobigs = toobigs | (self.max_2Dsize > self.config.cull_screen_size).squeeze()
            culls = culls | toobigs
            toobigs_count = torch.sum(toobigs).item()
        for name, param in self.gauss_params.items():
            self.gauss_params[name] = torch.nn.Parameter(param[~culls])

        CONSOLE.log(
            f"Culled {n_bef - self.num_points} gaussians "
            f"({below_alpha_count} below alpha thresh, {toobigs_count} too bigs, {self.num_points} remaining)"
        )

        return culls

    def split_gaussians(self, split_mask, samps):
        """
        This function splits gaussians that are too large
        """
        n_splits = split_mask.sum().item()
        CONSOLE.log(f"Splitting {split_mask.sum().item()/self.num_points} gaussians: {n_splits}/{self.num_points}")
        centered_samples = torch.randn((samps * n_splits, 3), device=self.device)  # Nx3 of axis-aligned scales
        scaled_samples = (
            torch.exp(self.scales[split_mask].repeat(samps, 1)) * centered_samples
        )  # how these scales are rotated
        quats = self.quats[split_mask] / self.quats[split_mask].norm(dim=-1, keepdim=True)  # normalize them first
        rots = quat_to_rotmat(quats.repeat(samps, 1))  # how these scales are rotated
        rotated_samples = torch.bmm(rots, scaled_samples[..., None]).squeeze()
        new_means = rotated_samples + self.means[split_mask].repeat(samps, 1)
        # step 2, sample new colors
        new_features_dc = self.features_dc[split_mask].repeat(samps, 1)
        new_features_rest = self.features_rest[split_mask].repeat(samps, 1, 1)
        # step 3, sample new opacities
        new_opacities = self.opacities[split_mask].repeat(samps, 1)
        # step 4, sample new scales
        size_fac = 1.6
        new_scales = torch.log(torch.exp(self.scales[split_mask]) / size_fac).repeat(samps, 1)
        self.scales[split_mask] = torch.log(torch.exp(self.scales[split_mask]) / size_fac)
        # step 5, sample new quats
        new_quats = self.quats[split_mask].repeat(samps, 1)
        out = {
            "means": new_means,
            "features_dc": new_features_dc,
            "features_rest": new_features_rest,
            "opacities": new_opacities,
            "scales": new_scales,
            "quats": new_quats,
        }
        for name, param in self.gauss_params.items():
            if name not in out:
                out[name] = param[split_mask].repeat(samps, 1)
        return out

    def dup_gaussians(self, dup_mask):
        """
        This function duplicates gaussians that are too small
        """
        n_dups = dup_mask.sum().item()
        CONSOLE.log(f"Duplicating {dup_mask.sum().item()/self.num_points} gaussians: {n_dups}/{self.num_points}")
        new_dups = {}
        for name, param in self.gauss_params.items():
            new_dups[name] = param[dup_mask]
        return new_dups

    def fit_gaussian_batch(self, points, masks):
        """
        Fit 3D Gaussians to a batch of points with valid masks

        Args:
            points (BxNx3): 3D points in a batch
            masks (BxN): mask in a batch

        Returns:
            means (Bx3): Gaussian means
            covariances (Bx3x3): Gaussian covariances
        """
        assert len(masks) > 0 and masks.sum(dim=-1).min() > 0
        masks_f = masks.float()
        masked_points = points * masks_f.unsqueeze(-1)
        sum_masked_points = torch.sum(masked_points, dim=1)
        count_masked_points = torch.sum(masks_f, dim=1, keepdim=True)
        means = sum_masked_points / count_masked_points
        # Correct the mean shape and compute demeaned points
        demeaned = masked_points - means.unsqueeze(1)
        # Correct broadcasting for outer products
        outer_product = torch.einsum('bni,bnj->bnij', demeaned, demeaned)
        weighted_outer_product = outer_product * masks_f.unsqueeze(-1).unsqueeze(-1)
        # Ensure proper division by broadcasting count_masked_points correctly
        covariances = torch.sum(weighted_outer_product, dim=1) \
            / (count_masked_points.unsqueeze(-1) - 1)
        return means, covariances

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        cbs = []
        cbs.append(TrainingCallback([TrainingCallbackLocation.BEFORE_TRAIN_ITERATION], self.step_cb))
        # The order of these matters
        cbs.append(
            TrainingCallback(
                [TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                self.after_train,
                args=[training_callback_attributes.optimizers],
            )
        )
        return cbs

    def step_cb(self, step):
        self.step = step

    def get_gaussian_param_groups(self) -> Dict[str, List[Parameter]]:
        # Here we explicitly use the means, scales as parameters so that the user can override this function and
        # specify more if they want to add more optimizable params to gaussians.
        return {
            name: [self.gauss_params[name]]
            for name in ["means", "scales", "quats", "features_dc", "features_rest", "opacities"]
        }

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        """Obtain the parameter groups for the optimizers

        Returns:
            Mapping of different parameter groups
        """
        gps = self.get_gaussian_param_groups()
        self.camera_optimizer.get_param_groups(param_groups=gps)
        return gps

    def _get_downscale_factor(self):
        if self.training:
            return 2 ** max(
                (self.config.num_downscales - self.step // self.config.resolution_schedule),
                0,
            )
        else:
            return 1

    def _downscale_if_required(self, image):
        d = self._get_downscale_factor()
        if d > 1:
            newsize = [image.shape[0] // d, image.shape[1] // d]

            # torchvision can be slow to import, so we do it lazily.
            import torchvision.transforms.functional as TF

            return TF.resize(image.permute(2, 0, 1), newsize, antialias=None).permute(1, 2, 0)
        return image

    def get_outputs(self, camera: Cameras) -> Dict[str, Union[torch.Tensor, List]]:
        """Takes in a Ray Bundle and returns a dictionary of outputs.

        Args:
            ray_bundle: Input bundle of rays. This raybundle should have all the
            needed information to compute the outputs.

        Returns:
            Outputs of model. (ie. rendered colors)
        """
        if not isinstance(camera, Cameras):
            print("Called get_outputs with not a camera")
            return {}
        assert camera.shape[0] == 1, "Only one camera at a time"

        # get the background color
        if self.training:
            optimized_camera_to_world = self.camera_optimizer.apply_to_camera(camera)
            if self.config.background_color == "random":
                background = torch.rand(3, device=self.device)
            elif self.config.background_color == "white":
                background = torch.ones(3, device=self.device)
            elif self.config.background_color == "black":
                background = torch.zeros(3, device=self.device)
            else:
                background = self.background_color.to(self.device)
        else:
            optimized_camera_to_world = camera.camera_to_worlds
            if renderers.BACKGROUND_COLOR_OVERRIDE is not None:
                background = renderers.BACKGROUND_COLOR_OVERRIDE.to(self.device)
            else:
                background = self.background_color.to(self.device)

        if self.crop_box is not None and not self.training:
            crop_ids = self.crop_box.within(self.means).squeeze()
            if crop_ids.sum() == 0:
                rgb = background.repeat(int(camera.height.item()), int(camera.width.item()), 1)
                depth = background.new_ones(*rgb.shape[:2], 1) * 10
                accumulation = background.new_zeros(*rgb.shape[:2], 1)
                return {"rgb": rgb, "depth": depth, "accumulation": accumulation, "background": background}
        else:
            crop_ids = None
        camera_downscale = self._get_downscale_factor()
        camera.rescale_output_resolution(1 / camera_downscale)
        # shift the camera to center of scene looking at center
        R = optimized_camera_to_world[0, :3, :3]  # 3 x 3
        T = optimized_camera_to_world[0, :3, 3:4]  # 3 x 1
        # flip the z and y axes to align with gsplat conventions
        R_edit = torch.diag(torch.tensor([1, -1, -1], device=self.device, dtype=R.dtype))
        R = R @ R_edit
        # analytic matrix inverse to get world2camera matrix
        R_inv = R.T
        T_inv = -R_inv @ T
        viewmat = torch.eye(4, device=R.device, dtype=R.dtype)
        viewmat[:3, :3] = R_inv
        viewmat[:3, 3:4] = T_inv
        # calculate the FOV of the camera given fx and fy, width and height
        cx = camera.cx.item()
        cy = camera.cy.item()
        W, H = int(camera.width.item()), int(camera.height.item())
        self.last_size = (H, W)

        if crop_ids is not None:
            opacities_crop = self.opacities[crop_ids]
            means_crop = self.means[crop_ids]
            features_dc_crop = self.features_dc[crop_ids]
            features_rest_crop = self.features_rest[crop_ids]
            scales_crop = self.scales[crop_ids]
            quats_crop = self.quats[crop_ids]
        else:
            opacities_crop = self.opacities
            means_crop = self.means
            features_dc_crop = self.features_dc
            features_rest_crop = self.features_rest
            scales_crop = self.scales
            quats_crop = self.quats
        
        if hasattr(self, "gauss_params_fixed"):
            opacities_crop = torch.cat(
                [opacities_crop, self.gauss_params_fixed["opacities"]], dim=0
            )
            means_crop = torch.cat(
                [means_crop, self.gauss_params_fixed["means"]], dim=0
            )
            features_dc_crop = torch.cat(
                [
                    features_dc_crop,
                    self.gauss_params_fixed["features_dc"]
                ], dim=0
            )
            features_rest_crop = torch.cat(
                [
                    features_rest_crop,
                    self.gauss_params_fixed["features_rest"]
                ], dim=0
            )
            scales_crop = torch.cat(
                [scales_crop, self.gauss_params_fixed["scales"]], dim=0)
            quats_crop = torch.cat(
                [quats_crop, self.gauss_params_fixed["quats"]], dim=0
            )
        else:
            if self.training:
                return {}                

        colors_crop = torch.cat((features_dc_crop[:, None, :], features_rest_crop), dim=1)
        BLOCK_WIDTH = 16  # this controls the tile size of rasterization, 16 is a good default
        self.xys, depths, self.radii, conics, comp, num_tiles_hit, cov3d = project_gaussians(  # type: ignore
            means_crop,
            torch.exp(scales_crop),
            1,
            quats_crop / quats_crop.norm(dim=-1, keepdim=True),
            viewmat.squeeze()[:3, :],
            camera.fx.item(),
            camera.fy.item(),
            cx,
            cy,
            H,
            W,
            BLOCK_WIDTH,
        )  # type: ignore

        # rescale the camera back to original dimensions before returning
        camera.rescale_output_resolution(camera_downscale)

        if (self.radii).sum() == 0:
            rgb = background.repeat(H, W, 1)
            depth = background.new_ones(*rgb.shape[:2], 1) * 10
            accumulation = background.new_zeros(*rgb.shape[:2], 1)

            return {"rgb": rgb, "depth": depth, "accumulation": accumulation, "background": background}

        # Important to allow xys grads to populate properly
        if self.training:
            if self.xys.requires_grad:
                self.xys.retain_grad()

        if self.config.sh_degree > 0:
            viewdirs = means_crop.detach() - optimized_camera_to_world.detach()[..., :3, 3]  # (N, 3)
            viewdirs = viewdirs / viewdirs.norm(dim=-1, keepdim=True)
            n = min(self.step // self.config.sh_degree_interval, self.config.sh_degree)
            rgbs = spherical_harmonics(n, viewdirs, colors_crop)
            rgbs = torch.clamp(rgbs + 0.5, min=0.0)  # type: ignore
        else:
            rgbs = torch.sigmoid(colors_crop[:, 0, :])

        assert (num_tiles_hit > 0).any()  # type: ignore

        # apply the compensation of screen space blurring to gaussians
        opacities = None
        if self.config.rasterize_mode == "antialiased":
            opacities = torch.sigmoid(opacities_crop) * comp[:, None]
        elif self.config.rasterize_mode == "classic":
            opacities = torch.sigmoid(opacities_crop)
        else:
            raise ValueError("Unknown rasterize_mode: %s", self.config.rasterize_mode)

        rgb, alpha = rasterize_gaussians(  # type: ignore
            self.xys,
            depths,
            self.radii,
            conics,
            num_tiles_hit,  # type: ignore
            rgbs,
            opacities,
            H,
            W,
            BLOCK_WIDTH,
            background=background,
            return_alpha=True,
        )  # type: ignore
        alpha = alpha[..., None]
        rgb = torch.clamp(rgb, max=1.0)  # type: ignore
        depth_im = None
        if self.config.output_depth_during_training or not self.training:
            depth_im = rasterize_gaussians(  # type: ignore
                self.xys,
                depths,
                self.radii,
                conics,
                num_tiles_hit,  # type: ignore
                depths[:, None].repeat(1, 3),
                opacities,
                H,
                W,
                BLOCK_WIDTH,
                background=torch.zeros(3, device=self.device),
            )[..., 0:1]  # type: ignore
            depth_im = torch.where(alpha > 0, depth_im / alpha, depth_im.detach().max())

        return {"rgb": rgb, "depth": depth_im, "accumulation": alpha, "background": background}  # type: ignore

    def get_gt_img(self, image: torch.Tensor):
        """Compute groundtruth image with iteration dependent downscale factor for evaluation purpose

        Args:
            image: tensor.Tensor in type uint8 or float32
        """
        if image.dtype == torch.uint8:
            image = image.float() / 255.0
        gt_img = self._downscale_if_required(image)
        return gt_img.to(self.device)

    def composite_with_background(self, image, background) -> torch.Tensor:
        """Composite the ground truth image with a background color when it has an alpha channel.

        Args:
            image: the image to composite
            background: the background color
        """
        if image.shape[2] == 4:
            alpha = image[..., -1].unsqueeze(-1).repeat((1, 1, 3))
            return alpha * image[..., :3] + (1 - alpha) * background
        else:
            return image

    def get_metrics_dict(self, outputs, batch) -> Dict[str, torch.Tensor]:
        """Compute and returns metrics.

        Args:
            outputs: the output to compute loss dict to
            batch: ground truth batch corresponding to outputs
        """
        gt_rgb = self.composite_with_background(self.get_gt_img(batch["image"]), outputs["background"])
        metrics_dict = {}
        predicted_rgb = outputs["rgb"]
        metrics_dict["psnr"] = self.psnr(predicted_rgb, gt_rgb)

        metrics_dict["gaussian_count"] = self.num_points
        self.camera_optimizer.get_metrics_dict(metrics_dict)
        return metrics_dict

    def get_loss_dict(self, outputs, batch, metrics_dict=None) -> Dict[str, torch.Tensor]:
        """Computes and returns the losses dict.

        Args:
            outputs: the output to compute loss dict to
            batch: ground truth batch corresponding to outputs
            metrics_dict: dictionary of metrics, some of which we can use for loss
        """
        gt_img = self.composite_with_background(self.get_gt_img(batch["image"]), outputs["background"])
        pred_img = outputs["rgb"]

        # Set masked part of both ground-truth and rendered image to black.
        # This is a little bit sketchy for the SSIM loss.
        if "mask" in batch:
            # batch["mask"] : [H, W, 1]
            mask = self._downscale_if_required(batch["mask"])
            mask = mask.to(self.device)
            assert mask.shape[:2] == gt_img.shape[:2] == pred_img.shape[:2]
            gt_img = gt_img * mask
            pred_img = pred_img * mask

        Ll1 = torch.abs(gt_img - pred_img).mean()
        simloss = 1 - self.ssim(gt_img.permute(2, 0, 1)[None, ...], pred_img.permute(2, 0, 1)[None, ...])
        if self.config.use_scale_regularization and self.step % 10 == 0:
            scale_exp = torch.exp(self.scales)
            scale_reg = (
                torch.maximum(
                    scale_exp.amax(dim=-1) / scale_exp.amin(dim=-1),
                    torch.tensor(self.config.max_gauss_ratio),
                )
                - self.config.max_gauss_ratio
            )
            scale_reg = 0.1 * scale_reg.mean()
        else:
            scale_reg = torch.tensor(0.0).to(self.device)

        loss_dict = {
            "main_loss": (1 - self.config.ssim_lambda) * Ll1 + self.config.ssim_lambda * simloss,
            "scale_reg": scale_reg,
        }
        if self.training:
            # Add loss from camera optimizer
            self.camera_optimizer.get_loss_dict(loss_dict)
        return loss_dict

    @torch.no_grad()
    def get_outputs_for_camera(self, camera: Cameras, obb_box: Optional[OrientedBox] = None) -> Dict[str, torch.Tensor]:
        """Takes in a camera, generates the raybundle, and computes the output of the model.
        Overridden for a camera-based gaussian model.

        Args:
            camera: generates raybundle
        """
        assert camera is not None, "must provide camera to gaussian model"
        self.set_crop(obb_box)
        outs = self.get_outputs(camera.to(self.device))
        return outs  # type: ignore

    def psnr_masked(self, image, rgb, mask):
        assert mask.dtype == torch.bool
        masked_psnr = self.psnr(
            image[mask.expand_as(image)], rgb[mask.expand_as(rgb)]
        )
        return masked_psnr

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        """Writes the test image outputs.

        Args:
            image_idx: Index of the image.
            step: Current step.
            batch: Batch of data.
            outputs: Outputs of the model.

        Returns:
            A dictionary of metrics.
        """
        gt_rgb = self.composite_with_background(self.get_gt_img(batch["image"]), outputs["background"])
        d = self._get_downscale_factor()
        if d > 1:
            # torchvision can be slow to import, so we do it lazily.
            import torchvision.transforms.functional as TF

            newsize = [batch["image"].shape[0] // d, batch["image"].shape[1] // d]
            predicted_rgb = TF.resize(outputs["rgb"].permute(2, 0, 1), newsize, antialias=None).permute(1, 2, 0)
        else:
            predicted_rgb = outputs["rgb"]

        combined_rgb = torch.cat([gt_rgb, predicted_rgb], dim=1)

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        gt_rgb = torch.moveaxis(gt_rgb, -1, 0)[None, ...]
        predicted_rgb = torch.moveaxis(predicted_rgb, -1, 0)[None, ...]

        psnr = self.psnr(gt_rgb, predicted_rgb)
        ssim = self.ssim(gt_rgb, predicted_rgb)
        lpips = self.lpips(gt_rgb, predicted_rgb)

        # all of these metrics will be logged as scalars
        metrics_dict = {"psnr": float(psnr.item()), "ssim": float(ssim)}  # type: ignore
        metrics_dict["lpips"] = float(lpips)

        # Compute PSNR separately for in and out of the masked areas
        if "mask" in batch:
            mask = batch["mask"].to(self.device)
            mask = torch.moveaxis(mask, -1, 0)[None, ...]
            psnr_in = self.psnr_masked(gt_rgb, predicted_rgb, mask)
            psnr_out = self.psnr_masked(gt_rgb, predicted_rgb, ~mask)
            metrics_dict["psnr_in"] = float(psnr_in.item())
            metrics_dict["psnr_out"] = float(psnr_out.item())

        images_dict = {"img": combined_rgb}

        return metrics_dict, images_dict
