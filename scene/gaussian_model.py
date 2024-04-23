#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import numpy as np
from matplotlib import pyplot as plt
from torch import nn
import os
from functools import cache
from sklearn.neighbors import NearestNeighbors
import numpy as np

from plyfile import PlyData, PlyElement
from utils.general_utils import strip_symmetric, build_scaling_rotation, build_rotation
from enum import Enum

from utils.sh_utils import RGB2SH
from utils.loss_utils import l1_loss, ssim
from utils.image_utils import psnr
from utils.system_utils import mkdir_p
from utils.general_utils import inverse_sigmoid, get_expon_lr_func

from arguments import (
    CompressionParams,
    ModelParams,
    OptimizationParams,
    PipelineParams,
    get_combined_args,
)
from compression.vq import CompressionSettings, compress_gaussians
from simple_knn._C import distCUDA2
# from utils.splats import to_full_cov, extract_rot_scale
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer, GaussianRasterizerIndexed
from utils.sh_utils import eval_sh

# from utils.graphics_utils import BasicPointCloud


class ColorMode(Enum):
    NOT_INDEXED = 0
    ALL_INDEXED = 1

class GaussianModel:
    def setup_functions(self, use_factor_scaling):
        def build_covariance_from_scaling_rotation(
            scaling, scaling_modifier, rotation, strip_sym=True
        ):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            if strip_sym:
                return strip_symmetric(actual_covariance)
            else:
                return actual_covariance

        if use_factor_scaling:
            self.scaling_activation = lambda x: torch.nn.functional.normalize(torch.nn.functional.relu(x))
            self.scaling_inverse_activation = lambda x: x
            self.scaling_factor_activation = torch.exp
            self.scaling_factor_inverse_activation = torch.log
        else:
            self.scaling_activation = torch.exp
            self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize

    def __init__(self, sh_degree: int, quantization=True, use_factor_scaling=True, device="cuda"):
        self.device = torch.device(device)

        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0, device=self.device, requires_grad=True)
        self._scaling = torch.empty(0)
        if use_factor_scaling:
            self._scaling_factor = torch.empty(0)
        else:
            self._scaling_factor = None
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0

        # quantization related stuff
        self._feature_indices = None
        self._gaussian_indices = None

        self.quantization = quantization
        self.color_index_mode = ColorMode.NOT_INDEXED

        self.features_dc_qa = torch.ao.quantization.FakeQuantize(dtype=torch.qint8).to(self.device)
        self.features_dc_qa = torch.ao.quantization.FakeQuantize(dtype=torch.qint8).to(self.device)
        self.features_rest_qa = torch.ao.quantization.FakeQuantize(dtype=torch.qint8).to(self.device)
        self.features_rest_qa = torch.ao.quantization.FakeQuantize(dtype=torch.qint8).to(self.device)
        self.opacity_qa = torch.ao.quantization.FakeQuantize(dtype=torch.qint8).to(self.device)
        self.scaling_qa = torch.ao.quantization.FakeQuantize(dtype=torch.qint8).to(self.device)
        if use_factor_scaling:
            self.scaling_factor_qa = torch.ao.quantization.FakeQuantize(dtype=torch.qint8).to(self.device)
        self.rotation_qa = torch.ao.quantization.FakeQuantize(dtype=torch.qint8).to(self.device)
        self.xyz_qa = FakeQuantizationHalf.apply

        if not self.quantization:
            self.features_dc_qa.disable_fake_quant()
            self.features_dc_qa.disable_observer()
            self.features_rest_qa.disable_fake_quant()
            self.features_rest_qa.disable_observer()
            
            self.scaling_qa.disable_fake_quant()
            self.scaling_qa.disable_observer()
            if use_factor_scaling:
                self.scaling_factor_qa.disable_fake_quant()
                self.scaling_factor_qa.disable_observer()

            self.rotation_qa.disable_fake_quant()
            self.rotation_qa.disable_observer()
            self.xyz_qa = lambda x: x

        self.setup_functions(use_factor_scaling)

    def check_state(self):
        n = self._xyz.shape[0]
        assert (self._opacity.shape[0] == n)

        if self.is_gaussian_indexed:
            assert (self._gaussian_indices.shape[0] == n)
            assert (self._gaussian_indices.max() < self._scaling.shape[0])
        else:
            assert (self._rotation.shape[0] == n)
            assert (self._scaling.shape[0] == n)

        if self.is_color_indexed:
            assert (self._feature_indices.shape[0] == n)
            assert (self._feature_indices.max() < self._features_dc.shape[0])
        else:
            assert (self._features_dc.shape[0] == n)
            assert (self._features_rest.shape[0] == n)

    def zero_grad(self):
        if self._xyz.grad is not None:
            self._xyz.grad.zero_()
        if self._features_dc.grad is not None:
            self._features_dc.grad.zero_()
        if self._features_rest.grad is not None:
            self._features_rest.grad.zero_()
        if self._scaling.grad is not None:
            self._scaling.grad.zero_()
        if self._rotation.grad is not None:
            self._rotation.grad.zero_()
        if self._opacity.grad is not None:
            self._opacity.grad.zero_()
        if self.max_radii2D.grad is not None:
            self.max_radii2D.grad.zero_()
        if self.xyz_gradient_accum is not None:
            self.xyz_gradient_accum.zero_()
        if self.denom is not None:
            self.denom.zero_()

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )

    def restore(self, model_args, training_args):
        (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            xyz_gradient_accum,
            denom,
            opt_dict,
            self.spatial_lr_scale,
        ) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)


    @property
    def get_scaling(self):
        scaling_n = self.scaling_qa(self.scaling_activation(self._scaling)) # self.get_scaling_normalized #
        if self._scaling_factor is None:
            return scaling_n
        scaling_factor = self.scaling_factor_activation(self.scaling_factor_qa(self._scaling_factor)) # self.get_scaling_factor
        if self.is_gaussian_indexed:
            return scaling_factor * scaling_n[self._gaussian_indices]
        else:
            return scaling_factor * scaling_n

    @property
    def get_scaling_normalized(self):
        return self.scaling_qa(self.scaling_activation(self._scaling))

    @property
    def get_scaling_factor(self):
        if self._scaling_factor is not None:
            return self.scaling_factor_activation(self.scaling_factor_qa(self._scaling_factor))
        else:
            return 1.0


    @property
    def get_rotation(self):
        rotation = self.rotation_activation(self.rotation_qa(self._rotation))
        if self.is_gaussian_indexed:
            return rotation[self._gaussian_indices]
        else:
            return rotation

    @property
    def _rotation_post_activation(self):
        return self.rotation_activation(self.rotation_qa(self._rotation))

    @property
    def get_xyz(self):
        return self.xyz_qa(self._xyz)

    @property
    def get_features(self):
        features_dc = self.features_dc_qa(self._features_dc)
        features_rest = self.features_rest_qa(self._features_rest)
        _ret = torch.cat((features_dc, features_rest), dim=1)
        return _ret[self._feature_indices] if self.color_index_mode == ColorMode.ALL_INDEXED else _ret
        
    @property
    def _get_features_raw(self):
        features_dc = self.features_dc_qa(self._features_dc)
        features_rest = self.features_rest_qa(self._features_rest)
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_opacity(self):
        return self.opacity_qa(self.opacity_activation(self._opacity))

    def get_covariance(self, scaling_modifier=1, strip_sym: bool = True):
        return self.covariance_activation(
            self.get_scaling, scaling_modifier, self.get_rotation, strip_sym
        )

    def get_normalized_covariance(self, scaling_modifier=1, strip_sym: bool = True):
        scaling_n = self.scaling_qa(self.scaling_activation(self._scaling))
        return self.covariance_activation(
            scaling_n, scaling_modifier, self.get_rotation, strip_sym
        )

    @property
    def is_color_indexed(self):
        return self._feature_indices is not None

    @property
    def is_gaussian_indexed(self):
        return self._gaussian_indices is not None

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device=self.device)
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device=self.device)

        l = [
            {"params": [self._xyz], "lr": training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz", },
            {"params": [self._features_dc], "lr": training_args.feature_lr, "name": "f_dc", },
            {"params": [self._features_rest], "lr": training_args.feature_lr / 20.0, "name": "f_rest", },
            {"params": [self._opacity], "lr": training_args.opacity_lr, "name": "opacity", },
            {"params": [self._scaling], "lr": training_args.scaling_lr, "name": "scaling", },
            {"params": [self._rotation], "lr": training_args.rotation_lr, "name": "rotation", },
        ]
        if self._scaling_factor is not None:
            l.append({"params": [self._scaling_factor], "lr": training_args.scaling_lr, "name": "scaling_factor", })

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(
            lr_init=training_args.position_lr_init * self.spatial_lr_scale,
            lr_final=training_args.position_lr_final * self.spatial_lr_scale,
            lr_delay_mult=training_args.position_lr_delay_mult,
            max_steps=training_args.position_lr_max_steps,
        )

    def update_learning_rate(self, iteration):
        """Learning rate scheduling per step"""
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group["lr"] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ["x", "y", "z", "nx", "ny", "nz"]
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1] * self._features_dc.shape[2]):
            l.append("f_dc_{}".format(i))
        for i in range(self._features_rest.shape[1] * self._features_rest.shape[2]):
            l.append("f_rest_{}".format(i))
        l.append("opacity")
        for i in range(self._scaling.shape[1]):
            l.append("scale_{}".format(i))
        # l.append("scale_factor")
        for i in range(self._rotation.shape[1]):
            l.append("rot_{}".format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        if self.is_gaussian_indexed or self.is_color_indexed:
            print(
                "WARNING: indexed colors/gaussians are not supported for ply files and are converted to dense attributes"
            )

        color_features = self.get_features.detach()

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = (
            color_features[:, :1]
            .transpose(1, 2)
            .flatten(start_dim=1)
            .contiguous()
            .cpu()
            .numpy()
        )
        f_rest = (
            color_features[:, 1:]
            .transpose(1, 2)
            .flatten(start_dim=1)
            .contiguous()
            .cpu()
            .numpy()
        )
        opacities = self._opacity.detach().cpu().numpy()
        scale = self.get_scaling.detach()
        if self._scaling_factor is not None:
            scale = self.scaling_factor_inverse_activation(scale)
        else:
            scale = self.scaling_inverse_activation(scale)
        scale = scale.cpu().numpy()

        rotation = self.get_rotation.detach().cpu().numpy()

        dtype_full = [
            (attribute, "f4") for attribute in self.construct_list_of_attributes()
        ]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate(
            (xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1
        )
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, "vertex")
        PlyData([el]).write(path)

    def load(self, path: str, override_quantization=False):
        ext = os.path.splitext(path)[1]
        if ext == ".ply":
            self.load_ply(path)
        elif ext == ".npz":
            self.load_npz(path, override_quantization)
        else:
            raise NotImplementedError(f"file ending '{ext}' not supported")

    def load_ply(self, path):
        print("Loading ply...")
        plydata = PlyData.read(path)
        vertices = plydata['vertex']
        keys = [p.name for p in vertices.properties]
        assert "x" in keys and "y" in keys and "z" in keys

        xyz = np.stack((
                np.asarray(vertices["x"]),
                np.asarray(vertices["y"]),
                np.asarray(vertices["z"]),
            ),
            axis=1,
        )
        n_points = xyz.shape[0]
        if "opacity" in keys:
            opacities = np.asarray(vertices["opacity"])[..., np.newaxis]
        else:
            #opacities = np.log(0.1 / 0.9) * np.ones((n_points, 1))
            x = 0.1 * np.ones((n_points, 1))
            opacities = np.log(x / (1 - x))

        features_dc = np.zeros((n_points, 3, 1))
        color_codes = ['red', 'green', 'blue'] if 'red' in keys else ["f_dc_0", "f_dc_1", "f_dc_2"]
        for i in range(3):
            features_dc[:, i, 0] = np.asarray(vertices[color_codes[i]])
        features_dc /= 255.0 if 'red' in keys else 1.0
        if 'red' in keys:
            features_dc = RGB2SH(features_dc) # bake into SH
        # print(path, keys, features_dc.max())

        extra_f_names = [p for p in keys if p.startswith("f_rest_")]
        if len(extra_f_names) > 0:
            extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split("_")[-1]))

            # determine active_degree
            degree = {3 * ((active + 1) ** 2 - 1) : active for active in range(self.max_sh_degree+1)}

            self.active_sh_degree = degree.get(len(extra_f_names), None)
            assert self.active_sh_degree # not None
            # assert len(extra_f_names) == 3 * ((self.max_sh_degree + 1) ** 2 - 1)
            # self.active_sh_degree = self.max_sh_degree

            features_extra = np.zeros((n_points, len(extra_f_names)))
            for idx, attr_name in enumerate(extra_f_names):
                features_extra[:, idx] = np.asarray(vertices[attr_name])
            # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
            features_extra = features_extra.reshape((n_points, 3, len(extra_f_names) // 3))
        else:
            self.active_sh_degree = 0 # self.max_sh_degree # 0
            features_extra = np.zeros((n_points, 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p for p in keys if p.startswith("scale_") and not p.startswith("scale_factor")]
        if len(scale_names) > 0:
            scale_names = sorted(scale_names, key=lambda x: int(x.split("_")[-1]))
            scales = np.zeros((n_points, len(scale_names)))
            for idx, attr_name in enumerate(scale_names):
                scales[:, idx] = np.asarray(vertices[attr_name])
            scaling = torch.tensor(scales, dtype=torch.float, device=self.device)
        else:
            print("clamping using cuda!")
            dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(xyz).float().cuda()).to(self.device), 0.0000001)
            print("done")
            scaling = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)#.detach().cpu().numpy()

        rot_names = [p for p in keys if p.startswith("rot")]
        if len(rot_names) > 0:
            rot_names = sorted(rot_names, key=lambda x: int(x.split("_")[-1]))
            rots = np.zeros((n_points, len(rot_names)))
            for idx, attr_name in enumerate(rot_names):
                rots[:, idx] = np.asarray(vertices[attr_name])
        else:
            rots = np.zeros((n_points, 4))
            rots[:, 0] = 1

        self._xyz = nn.Parameter(
            torch.tensor(xyz, dtype=torch.float, device=self.device).requires_grad_(True)
        )
        self._features_dc = nn.Parameter(
            torch.tensor(features_dc, dtype=torch.float, device=self.device)
            .transpose(1, 2).contiguous().requires_grad_(True)
        )
        self._features_rest = nn.Parameter(
            torch.tensor(features_extra, dtype=torch.float, device=self.device)
            .transpose(1, 2).contiguous().requires_grad_(True)
        )

        self._opacity = nn.Parameter(
            torch.tensor(opacities, dtype=torch.float, device=self.device).requires_grad_(True)
        )

        if self._scaling_factor is not None:
            scaling = self.scaling_factor_activation(scaling)
            scaling_norm = scaling.norm(2, -1, keepdim=True)
            self._scaling = nn.Parameter(
                self.scaling_inverse_activation(scaling / scaling_norm).requires_grad_(True)
            )
            self._scaling_factor = nn.Parameter(
                self.scaling_factor_inverse_activation(scaling_norm).detach().requires_grad_(True)
            )
        else:
            self._scaling = nn.Parameter(scaling.requires_grad_(True))

        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device=self.device).requires_grad_(True))
        self.max_radii2D = torch.zeros((n_points), device=self.device)


    def save_npz(
        self,
        path,
        compress: bool = True,
        half_precision: bool = False,
        sort_morton=False,
    ):
        with torch.no_grad():
            if sort_morton:
                self._sort_morton()
            if isinstance(path, str):
                mkdir_p(os.path.dirname(os.path.abspath(path)))

            dtype = torch.half if half_precision else torch.float32
            save_dict = dict()
            save_dict["quantization"] = self.quantization
            # save position
            if self.quantization:
                save_dict["xyz"] = self.get_xyz.detach().half().cpu().numpy()
            else:
                save_dict["xyz"] = self._xyz.detach().cpu().numpy()

            # save color features
            if self.quantization:
                features_dc_q = torch.quantize_per_tensor(
                    self._features_dc.detach(),
                    self.features_dc_qa.scale,
                    self.features_dc_qa.zero_point,
                    self.features_dc_qa.dtype,
                ).int_repr()
                save_dict["features_dc"] = features_dc_q.cpu().numpy()
                save_dict["features_dc_scale"] = self.features_dc_qa.scale.cpu().numpy()
                save_dict["features_dc_zero_point"] = self.features_dc_qa.zero_point.cpu().numpy()

                features_rest_q = torch.quantize_per_tensor(
                    self._features_rest.detach(),
                    self.features_rest_qa.scale,
                    self.features_rest_qa.zero_point,
                    self.features_rest_qa.dtype,
                ).int_repr()
                save_dict["features_rest"] = features_rest_q.cpu().numpy()
                save_dict["features_rest_scale"] = self.features_rest_qa.scale.cpu().numpy()
                save_dict["features_rest_zero_point"] = self.features_rest_qa.zero_point.cpu().numpy()
            else:
                save_dict["features_dc"] = self._features_dc.detach().cpu().numpy()
                save_dict["features_rest"] = self._features_rest.detach().cpu().numpy()

            # save opacity
            if self.quantization:
                opacity = self.opacity_activation(self._opacity).detach()
                opacity_q = torch.quantize_per_tensor(
                    opacity,
                    scale=self.opacity_qa.scale,
                    zero_point=self.opacity_qa.zero_point,
                    dtype=self.opacity_qa.dtype,
                ).int_repr()
                save_dict["opacity"] = opacity_q.cpu().numpy()
                save_dict["opacity_scale"] = self.opacity_qa.scale.cpu().numpy()
                save_dict["opacity_zero_point"] = self.opacity_qa.zero_point.cpu().numpy()
            else:
                save_dict["opacity"] = self._opacity.detach().to(dtype).cpu().numpy()

            # save indices
            if self.is_color_indexed:
                save_dict["feature_indices"] = self._feature_indices.detach().contiguous().cpu().int().numpy()
            if self.is_gaussian_indexed:
                save_dict["gaussian_indices"] = self._gaussian_indices.detach().contiguous().cpu().int().numpy()

            # save scaling
            if self.quantization:
                scaling = self.scaling_activation(self._scaling.detach())
                scaling_q = torch.quantize_per_tensor(
                    scaling,
                    scale=self.scaling_qa.scale,
                    zero_point=self.scaling_qa.zero_point,
                    dtype=self.scaling_qa.dtype,
                ).int_repr()
                save_dict["scaling"] = scaling_q.cpu().numpy()
                save_dict["scaling_scale"] = self.scaling_qa.scale.cpu().numpy()
                save_dict[
                    "scaling_zero_point"
                ] = self.scaling_qa.zero_point.cpu().numpy()
                if self._scaling_factor is not None:
                    scaling_factor = self._scaling_factor.detach()
                    scaling_factor_q = torch.quantize_per_tensor(
                        scaling_factor,
                        scale=self.scaling_factor_qa.scale,
                        zero_point=self.scaling_factor_qa.zero_point,
                        dtype=self.scaling_factor_qa.dtype,
                    ).int_repr()
                    save_dict["scaling_factor"] = scaling_factor_q.cpu().numpy()
                    save_dict[
                        "scaling_factor_scale"
                    ] = self.scaling_factor_qa.scale.cpu().numpy()
                    save_dict[
                        "scaling_factor_zero_point"
                    ] = self.scaling_factor_qa.zero_point.cpu().numpy()
            else:
                save_dict["scaling"] = self._scaling.detach().to(dtype).cpu().numpy()
                if self._scaling_factor is not None:
                    save_dict["scaling_factor"] = self._scaling_factor.detach().to(dtype).cpu().numpy()

            # save rotation
            if self.quantization:
                rotation = self.rotation_activation(self._rotation).detach()
                rotation_q = torch.quantize_per_tensor(
                    rotation,
                    scale=self.rotation_qa.scale,
                    zero_point=self.rotation_qa.zero_point,
                    dtype=self.rotation_qa.dtype,
                ).int_repr()
                save_dict["rotation"] = rotation_q.cpu().numpy()
                save_dict["rotation_scale"] = self.rotation_qa.scale.cpu().numpy()
                save_dict["rotation_zero_point"] = self.rotation_qa.zero_point.cpu().numpy()
            else:
                save_dict["rotation"] = self._rotation.detach().to(dtype).cpu().numpy()

            save_fn = np.savez_compressed if compress else np.savez
            save_fn(path, **save_dict)

    def load_npz(self, path,override_quantization=False):
        state_dict = np.load(path)

        quantization = state_dict["quantization"]
        if not override_quantization and self.quantization != quantization:
            print("WARNING: model is not quantisation aware but loaded model is")
        if override_quantization:
            self.quantization = quantization

        # load position
        self._xyz = nn.Parameter(
            torch.from_numpy(state_dict["xyz"]).float().to(self.device), requires_grad=True
        )

        # load color
        if quantization:
            features_rest_q = torch.from_numpy(state_dict["features_rest"]).int().to(self.device)
            features_rest_scale = torch.from_numpy(state_dict["features_rest_scale"]).to(self.device)
            features_rest_zero_point = torch.from_numpy(state_dict["features_rest_zero_point"]).to(self.device)
            features_rest = (features_rest_q - features_rest_zero_point) * features_rest_scale
            self._features_rest = nn.Parameter(features_rest, requires_grad=True)
            self.features_rest_qa.scale = features_rest_scale
            self.features_rest_qa.zero_point = features_rest_zero_point
            self.features_rest_qa.activation_post_process.min_val = features_rest.min()
            self.features_rest_qa.activation_post_process.max_val = features_rest.max()

            features_dc_q = torch.from_numpy(state_dict["features_dc"]).int().to(self.device)
            features_dc_scale = torch.from_numpy(state_dict["features_dc_scale"]).to(self.device)
            features_dc_zero_point = torch.from_numpy(
                state_dict["features_dc_zero_point"]
            ).to(self.device)
            features_dc = (features_dc_q - features_dc_zero_point) * features_dc_scale
            self._features_dc = nn.Parameter(features_dc, requires_grad=True)

            self.features_dc_qa.scale = features_dc_scale
            self.features_dc_qa.zero_point = features_dc_zero_point
            self.features_dc_qa.activation_post_process.min_val = features_dc.min()
            self.features_dc_qa.activation_post_process.max_val = features_dc.max()
        else:
            features_dc = torch.from_numpy(state_dict["features_dc"]).float().to(self.device)
            features_rest = torch.from_numpy(state_dict["features_rest"]).float().to(self.device)
            self._features_dc = nn.Parameter(features_dc, requires_grad=True)
            self._features_rest = nn.Parameter(features_rest, requires_grad=True)

        # load opacity
        if quantization:
            opacity_q = torch.from_numpy(state_dict["opacity"]).int().to(self.device)
            opacity_scale = torch.from_numpy(state_dict["opacity_scale"]).to(self.device)
            opacity_zero_point = torch.from_numpy(state_dict["opacity_zero_point"]).to(self.device)
            opacity = (opacity_q - opacity_zero_point) * opacity_scale
            self._opacity = nn.Parameter(self.inverse_opacity_activation(opacity), requires_grad=True)
            self.opacity_qa.scale = opacity_scale
            self.opacity_qa.zero_point = opacity_zero_point
            self.opacity_qa.activation_post_process.min_val = opacity.min()
            self.opacity_qa.activation_post_process.max_val = opacity.max()
        else:
            self._opacity = nn.Parameter(torch.from_numpy(state_dict["opacity"]).float().to(self.device), requires_grad=True)

        # load scaling
        if quantization:
            scaling_q = torch.from_numpy(state_dict["scaling"]).int().to(self.device)
            scaling_scale = torch.from_numpy(state_dict["scaling_scale"]).to(self.device)
            scaling_zero_point = torch.from_numpy(state_dict["scaling_zero_point"]).to(self.device)
            scaling = (scaling_q - scaling_zero_point) * scaling_scale
            self._scaling = nn.Parameter(self.scaling_inverse_activation(scaling), requires_grad=True)
            self.scaling_qa.scale = scaling_scale
            self.scaling_qa.zero_point = scaling_zero_point
            self.scaling_qa.activation_post_process.min_val = scaling.min()
            self.scaling_qa.activation_post_process.max_val = scaling.max()

            scaling_factor_q = (torch.from_numpy(state_dict["scaling_factor"]).int().to(self.device))
            scaling_factor_scale = torch.from_numpy(state_dict["scaling_factor_scale"]).to(self.device)
            scaling_factor_zero_point = torch.from_numpy(state_dict["scaling_factor_zero_point"]).to(self.device)
            scaling_factor = (scaling_factor_q - scaling_factor_zero_point) * scaling_factor_scale
            self._scaling_factor = nn.Parameter(scaling_factor, requires_grad=True)
            self.scaling_factor_qa.scale = scaling_factor_scale
            self.scaling_factor_qa.zero_point = scaling_factor_zero_point
            self.scaling_factor_qa.activation_post_process.min_val = scaling_factor.min()
            self.scaling_factor_qa.activation_post_process.max_val = scaling_factor.max()
        else:
            self._scaling_factor = nn.Parameter(torch.from_numpy(state_dict["scaling_factor"]).float().to(self.device),
                requires_grad=True)
            self._scaling = nn.Parameter(torch.from_numpy(state_dict["scaling"]).float().to(self.device),
                requires_grad=True)
        # load rotation
        if quantization:
            rotation_q = torch.from_numpy(state_dict["rotation"]).int().to(self.device)
            rotation_scale = torch.from_numpy(state_dict["rotation_scale"]).to(self.device)
            rotation_zero_point = torch.from_numpy(state_dict["rotation_zero_point"]).to(self.device)
            rotation = (rotation_q - rotation_zero_point) * rotation_scale
            self._rotation = nn.Parameter(rotation, requires_grad=True)
            self.rotation_qa.scale = rotation_scale
            self.rotation_qa.zero_point = rotation_zero_point
            self.rotation_qa.activation_post_process.min_val = rotation.min()
            self.rotation_qa.activation_post_process.max_val = rotation.max()
        else:
            self._rotation = nn.Parameter(
                torch.from_numpy(state_dict["rotation"]).float().to(self.device),
                requires_grad=True,
            )

        if "gaussian_indices" in list(state_dict.keys()):
            self._gaussian_indices = nn.Parameter(
                torch.from_numpy(state_dict["gaussian_indices"]).long().to(self.device),
                requires_grad=False,
            )

        self.color_index_mode = ColorMode.NOT_INDEXED
        if "feature_indices" in list(state_dict.keys()):
            self._feature_indices = nn.Parameter(
                torch.from_numpy(state_dict["feature_indices"]).long().to(self.device),
                requires_grad=False,
            )
            self.color_index_mode = ColorMode.ALL_INDEXED

        self.active_sh_degree = self.max_sh_degree
    @cache
    def markVisible(self, P, T=1):
        n = self.get_xyz.shape[0]
        data = torch.cat([self.get_xyz, torch.ones((n, 1)).to(self.device)], axis=1) # n x 4
        u = data @ P[:, 0]
        v = data @ P[:, 1]
        z = data @ P[:, 3] + 1e-7
        u /= z
        v /= z
        # z = data @ W[:, 2] # n x 4
        visible = (u >= -T) & (u <= T) & (v >= -T) & (v <= T) #
        # idx = torch.nonzero(visible)
        # print(idx.min(), idx.max())

        # construct image from bit array
        # m = int(np.ceil(np.sqrt(n)))
        # vis = np.zeros(m*m).astype(bool)
        # vis[:n] = visible.detach().cpu().numpy()
        # f = plt.figure()
        # plt.imshow(vis.reshape(m, m))
        # plt.plot()

        return visible
        # return torch.ones(n, dtype=torch.bool, device=self.device)

    def render(self,
            viewpoint_camera,
            pipe: PipelineParams,
            bg_color: torch.Tensor,
            scaling_modifier=1.0,
            override_color=None,
            clamp_color: bool = True,
            cov3d: torch.Tensor = None,
    ):
        """
        Render the scene.

        Background tensor (bg_color) must be on GPU!
        """

        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
        screenspace_points = torch.zeros_like(self.get_xyz, dtype=self.get_xyz.dtype, \
                                              requires_grad=True, device=self.device)
        try:
            screenspace_points.retain_grad()
        except:
            pass

        # Set up rasterization configuration
        tanfovx = math.tan(viewpoint_camera.intrinsic[0, 0] * 0.5)
        tanfovy = math.tan(viewpoint_camera.intrinsic[1, 1] * 0.5)

        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color.cuda(),
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform.cuda(),
            projmatrix=viewpoint_camera.full_proj_transform.cuda(),
            sh_degree=self.active_sh_degree,
            campos=viewpoint_camera.camera_center.cuda(),
            prefiltered=True, #False,
            debug=pipe.debug,
            clamp_color=clamp_color,
        )

        render_indexed = (self.color_index_mode == ColorMode.ALL_INDEXED) and self.is_gaussian_indexed

        means3D = self.get_xyz
        means2D = screenspace_points
        opacity = self.get_opacity

        rasterizer = GaussianRasterizerIndexed(raster_settings=raster_settings) if render_indexed \
            else GaussianRasterizer(raster_settings=raster_settings)

        # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
        # scaling / rotation by the rasterizer.
        scales = None
        rotations = None
        cov3D_precomp = cov3d
        if cov3D_precomp is None:
            if pipe.compute_cov3D_python:
                cov3D_precomp = self.get_covariance(scaling_modifier)
            else:
                scales = self.get_scaling_normalized if render_indexed else self.get_scaling
                rotations = self._rotation_post_activation if render_indexed else self.get_rotation

        scale_factors = self.get_scaling_factor if render_indexed else None

        shs = None
        colors_precomp = None
        if override_color is None:
            if pipe.convert_SHs_python:
                shs_view = self.get_features.transpose(1, 2).view(-1, 3, (self.max_sh_degree + 1) ** 2)
                dir_pp = self.get_xyz - viewpoint_camera.camera_center.repeat(self.get_features.shape[0], 1)
                dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
                sh2rgb = eval_sh(self.active_sh_degree, shs_view, dir_pp_normalized)
                colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
            else:
                shs = self._get_features_raw if render_indexed else self.get_features
        else:
            colors_precomp = override_color

        # precalculate visible points
        visible = self.markVisible(viewpoint_camera.full_proj_transform)

        # if self.device == torch.device("cuda"):
        #     visible = rasterizer.markVisible(self.get_xyz)
        # else:
        #     # visible = rasterizer.markVisible(self.get_xyz.cuda()).to(self.device)
        #     visible = self.markVisible(viewpoint_camera.full_proj_transform)


        # visible = torch.ones(self.get_xyz.shape[0], dtype=torch.bool, device=self.get_xyz.device)

        if render_indexed:
            rendered_image, radii = rasterizer(
                means3D=means3D[visible, :].cuda(),
                means2D=means2D[visible, :].cuda(),
                shs=shs.cuda() if shs is not None else None,
                sh_indices=self._feature_indices[visible].cuda(),
                g_indices=self._gaussian_indices[visible].cuda(),
                colors_precomp=None,
                opacities=opacity[visible].cuda(),
                scales=scales.cuda() if scales is not None else None,
                scale_factors=scale_factors[visible, :].cuda(),
                rotations=rotations.cuda() if rotations is not None else None,
                cov3D_precomp=cov3D_precomp[visible, :].cuda() if cov3D_precomp is not None else None,
            )
        else:
            rendered_image, radii = rasterizer(
                means3D=means3D[visible, :].cuda(),
                means2D=means2D[visible, :].cuda(),
                shs=shs[visible, :, :].cuda(),
                colors_precomp=colors_precomp[visible, :].cuda() if colors_precomp is not None else None,
                opacities=opacity[visible, :].cuda(),
                scales=scales[visible, :].cuda() if scales is not None else None,
                rotations=rotations[visible, :].cuda() if rotations is not None else None,
                cov3D_precomp=cov3D_precomp[visible, :].cuda() if cov3D_precomp is not None else None,
            )
        # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
        # They will be excluded from value updates used in the splitting criteria.
        return {
            "render": rendered_image.to(self.device),
            "viewspace_points": screenspace_points.to(self.device),
            "visibility_filter": radii.to(self.device) > 0,
            "radii": radii.to(self.device),
            "visible": visible.to(self.device)
        }

    # convert indexed to non-indexed
    def to_unindexed(self):
        if self.color_index_mode == ColorMode.NOT_INDEXED:
            return # nothing to do
        assert self.is_gaussian_indexed and self.is_color_indexed
        self._features_dc = nn.Parameter(self._features_dc[self._feature_indices, :, :], requires_grad=True)
        self._features_rest = nn.Parameter(self._features_rest[self._feature_indices, :, :], requires_grad=True)
        self._feature_indices = None
        self._rotation = nn.Parameter(self._rotation[self._gaussian_indices, :], requires_grad=True)
        self._scaling = nn.Parameter(self._scaling[self._gaussian_indices, :], requires_grad=True)
        self._gaussian_indices = None
        self.color_index_mode = ColorMode.NOT_INDEXED

    # convert non-indexed to indexed (without compression)
    def to_indexed(self):
        if self.color_index_mode == ColorMode.ALL_INDEXED:
            return # nothing to do
        n = self._features_dc.shape[0]
        dev = self._features_dc.device
        self._feature_indices = nn.Parameter(torch.arange(0, n, dtype=torch.long, device=dev), requires_grad=False)
        self._gaussian_indices = nn.Parameter(torch.arange(0, n, dtype=torch.long, device=dev), requires_grad=False)
        self.color_index_mode = ColorMode.ALL_INDEXED
        assert self.is_gaussian_indexed and self.is_color_indexed

    def calc_importance(self, scene, pipeline_params: PipelineParams, use_gt=False):
        cov3d_scaled = self.get_covariance().detach()
        scaling_factor = self.get_scaling_factor.detach()
        coeff = scaling_factor.square()
        cov3d = (cov3d_scaled / coeff).requires_grad_(True)

        accum1 = torch.zeros_like(self._features_dc).requires_grad_(False)
        accum2 = torch.zeros_like(self._features_rest).requires_grad_(False)
        accum3 = torch.zeros_like(cov3d).requires_grad_(False)
        background = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device=self.device)

        num_pixels = 0
        iterator, name = scene.getSomeCameras()

        # self._features_dc.retain_grad()
        # self._features_rest.retain_grad()
        # cov3d.retain_grad()

        for camera in iterator:
            self.zero_grad()  # all grads to zero
            if cov3d.grad is not None:
                cov3d.grad.zero_()
            image = self.render(camera, pipeline_params, background, clamp_color=False, cov3d=cov3d * coeff)["render"]
            # gradients are accumulated during cycle each time backward is called
            if not use_gt:
                image.sum().backward()
            else:
                lambda_dssim = 0.2
                gt_image = camera.original_image.to(self.device)
                loss = (1.0 - lambda_dssim) * l1_loss(image, gt_image) + lambda_dssim * (1.0 - ssim(image, gt_image))
                loss.backward()

            accum1 += torch.abs(self._features_dc.grad)
            accum2 += torch.abs(self._features_rest.grad)
            accum3 += torch.abs(cov3d.grad)
            num_pixels += image.shape[1] * image.shape[2]

        importance = torch.cat([accum1, accum2], 1).flatten(-2)
        cov_grad = accum3
        torch.cuda.empty_cache()
        return importance / num_pixels, cov_grad / num_pixels

    def to_compressed(self, scene, pipeline_params: PipelineParams, comp_params: CompressionParams):
        self.to_unindexed() # always unwrap

        color_importance, gaussian_sensitivity = self.calc_importance(scene, pipeline_params, use_gt=True)
        with torch.no_grad():
            color_importance_n = color_importance.amax(-1)
            gaussian_importance_n = gaussian_sensitivity.amax(-1)

            torch.cuda.empty_cache()

            color_compression_settings = CompressionSettings(
                codebook_size=comp_params.color_codebook_size,
                importance_prune=comp_params.color_importance_prune,
                importance_include=None,  # comp_params.color_importance_include,
                importance_include_relative=0.9,
                steps=int(comp_params.color_cluster_iterations),
                decay=comp_params.color_decay,
                batch_size=comp_params.color_batch_size,
            )

            gaussian_compression_settings = CompressionSettings(
                codebook_size=comp_params.gaussian_codebook_size,
                importance_prune=None,
                importance_include=None,  # comp_params.gaussian_importance_include,#None
                importance_include_relative=0.75,
                steps=int(comp_params.gaussian_cluster_iterations),
                decay=comp_params.gaussian_decay,
                batch_size=comp_params.gaussian_batch_size,
            )

            compress_gaussians(self, color_importance_n, gaussian_importance_n,
                               color_compression_settings if not comp_params.not_compress_color else None,
                               gaussian_compression_settings if not comp_params.not_compress_gaussians else None,
                               comp_params.color_compress_non_dir, prune_threshold=comp_params.prune_threshold)

            # print(len(self._rotation), len(np.unique(self._gaussian_indices.detach().cpu().numpy())))

        torch.cuda.empty_cache()

        # if comp_params.finetune_iterations > 0:
        #     finetune(scene, model_params, optim_params, comp_params, pipeline_params, testing_iterations=[-1],
        #              debug_from=-1)

    def _sort_morton(self):
        with torch.no_grad():
            pp_min = self._xyz.min(0).values
            pp_diap = self._xyz.max(0).values - pp_min
            xyz_q = ((2**21 - 1) * (self._xyz - pp_min) / pp_diap).long()

            order = mortonEncode(xyz_q, pp_diap.argsort()).sort().indices

            self._xyz = nn.Parameter(self._xyz[order], requires_grad=True)
            self._opacity = nn.Parameter(self._opacity[order], requires_grad=True)
            if self._scaling_factor is not None:
                self._scaling_factor = nn.Parameter(self._scaling_factor[order], requires_grad=True)

            if self.is_color_indexed:
                self._feature_indices = nn.Parameter(self._feature_indices[order], requires_grad=False)
            else:
                self._features_rest = nn.Parameter(self._features_rest[order], requires_grad=True)
                self._features_dc = nn.Parameter(self._features_dc[order], requires_grad=True)

            if self.is_gaussian_indexed:
                self._gaussian_indices = nn.Parameter(self._gaussian_indices[order], requires_grad=False)
            else:
                self._scaling = nn.Parameter(self._scaling[order], requires_grad=True)
                self._rotation = nn.Parameter(self._rotation[order], requires_grad=True)
            if len(self.xyz_gradient_accum) > 0:
                self.xyz_gradient_accum = self.xyz_gradient_accum[order]
                self.denom = self.denom[order]



    def mask_splats(self, mask: torch.Tensor):
        with torch.no_grad():
            self._xyz = nn.Parameter(self._xyz[mask], requires_grad=True)
            self._opacity = nn.Parameter(self._opacity[mask], requires_grad=True)
            if len(self.xyz_gradient_accum) > 0:
                self.xyz_gradient_accum = self.xyz_gradient_accum[mask.to(self.xyz_gradient_accum.device)]
                self.denom = self.denom[mask.to(self.denom.device)]

            if self._scaling_factor is not None:
                self._scaling_factor = nn.Parameter(self._scaling_factor[mask], requires_grad=True)
            if self.is_color_indexed:
                self._feature_indices = nn.Parameter(self._feature_indices[mask], requires_grad=False)
            else:
                self._features_dc = nn.Parameter(self._features_dc[mask], requires_grad=True)
                self._features_rest = nn.Parameter(self._features_rest[mask], requires_grad=True)
            if self.is_gaussian_indexed:
                self._gaussian_indices = nn.Parameter(self._gaussian_indices[mask], requires_grad=False)
            else:
                self._scaling = nn.Parameter(self._scaling[mask], requires_grad=True)
                self._rotation = nn.Parameter(self._rotation[mask], requires_grad=True)

    def set_color_indexed(self, features: torch.Tensor, indices: torch.Tensor):
        self._feature_indices = nn.Parameter(indices, requires_grad=False)
        self._features_dc = nn.Parameter(features[:, :1].detach(), requires_grad=True)
        self._features_rest = nn.Parameter(features[:, 1:].detach(), requires_grad=True)
        self.color_index_mode = ColorMode.ALL_INDEXED

    def set_gaussian_indexed(
        self, rotation: torch.Tensor, scaling: torch.Tensor, indices: torch.Tensor
    ):
        self._gaussian_indices = nn.Parameter(indices.detach(), requires_grad=False)
        self._rotation = nn.Parameter(rotation.detach(), requires_grad=True)
        self._scaling = nn.Parameter(scaling.detach(), requires_grad=True)

    def replace_tensor_to_optimizer(self, tensor, #mask, vals,
                                    name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                if stored_state is not None:
                    stored_state["exp_avg"] = torch.zeros_like(tensor)
                    stored_state["exp_avg_sq"] = torch.zeros_like(tensor)
                    # stored_state["exp_avg"][mask] = vals[1]
                    # stored_state["exp_avg_sq"][mask] = vals[2]

                    del self.optimizer.state[group['params'][0]]
                    group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                    # group["params"][0][mask] = vals[0]
                    self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask, names=None):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if names is None or group["name"] in names:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                if stored_state is not None and len(stored_state["exp_avg"]) > 0:
                    stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                    stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                    del self.optimizer.state[group['params'][0]]
                    group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                    self.optimizer.state[group['params'][0]] = stored_state

                    optimizable_tensors[group["name"]] = group["params"][0]
                else:
                    if len(group["params"][0]) > 0:
                        group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                    optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        # valid corresponds to index array
        # feat corresponds to features array
        def calc_valid(n_feats, index, valid):
            print(f'index_max={index.max()}, n_feats={n_feats}')
            assert(index.max() < n_feats)
            valid_ids_list = index[valid]
            feat_valid = torch.zeros(n_feats, dtype=torch.bool, device=valid.device)
            feat_valid[valid_ids_list] = True
            unique_ids = torch.sort(torch.unique(valid_ids_list))[0]
            index_map = torch.zeros(unique_ids[-1] + 1, dtype=index.dtype, device=index.device)
            for i, u in enumerate(unique_ids):
                index_map[u] = i
            return feat_valid, index_map[valid_ids_list]


        valid_points_mask = ~mask
        # prune_optimizer applies same mask on all inputs
        # but we need to apply different criteria on features and scaling, when we have indices
        names = set(group["name"] for group in self.optimizer.param_groups)
        color_names = {"f_dc", "f_rest"}
        gaus_names = {"scaling", "rotation"}
        if self.is_color_indexed:
            names -= color_names
        if self.is_gaussian_indexed:
            names -= gaus_names

        optimizable_tensors = self._prune_optimizer(valid_points_mask, names=names)

        if self.is_color_indexed and self.is_gaussian_indexed:
            assert(len(valid_points_mask) == len(self._feature_indices))
            assert(len(valid_points_mask) == len(self._gaussian_indices))
            print(self._features_dc.shape[0], self._feature_indices.shape[0], valid_points_mask.shape[0])
            print(self._scaling.shape[0], self._gaussian_indices.shape[0], valid_points_mask.shape[0])

        if self.is_color_indexed:
            # 1. prune only features, that has no indices
            # 2. rebuild links (indices) after pruning
            color_valid_points_mask, self._feature_indices = calc_valid(self._features_dc.shape[0], self._feature_indices, valid_points_mask)
            optimizable_tensors.update(self._prune_optimizer(color_valid_points_mask, names=color_names))

        if self.is_gaussian_indexed:
            gaus_valid_points_mask, self._gaussian_indices = calc_valid(self._scaling.shape[0], self._gaussian_indices, valid_points_mask)
            optimizable_tensors.update(self._prune_optimizer(gaus_valid_points_mask, names=gaus_names))

        self._xyz = optimizable_tensors["xyz"]
        self._opacity = optimizable_tensors["opacity"]
        if self._scaling_factor is not None:
            self._scaling_factor = optimizable_tensors["scaling_factor"]
        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]

        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]


    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)),
                                                    dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)),
                                                       dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling,
                              new_scaling_factor,
                              new_rotation):
        d = {"xyz": new_xyz,
             "f_dc": new_features_dc,
             "f_rest": new_features_rest,
             "opacity": new_opacities,
             "scaling": new_scaling,
             "scaling_factor": new_scaling_factor,
             "rotation": new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        print('Before', self._features_dc.shape, new_features_dc.shape)
        self._features_dc = optimizable_tensors["f_dc"]
        print('After', self._features_dc.shape)
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        if self._scaling_factor is not None:
            self._scaling_factor = optimizable_tensors["scaling_factor"]

        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device=self.device)
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device=self.device)
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device=self.device)

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        if self.is_gaussian_indexed and self.is_color_indexed:
            print('F', len(self._features_dc), self._feature_indices.max())
            print('R', len(self._rotation), self._gaussian_indices.max())
            assert (self._feature_indices.max() < len(self._features_dc))
            assert (self._feature_indices.max() < len(self._features_rest))

            assert (self._gaussian_indices.max() < len(self._rotation))
            assert (self._gaussian_indices.max() < len(self._scaling))

        n_init_points = self.get_xyz.shape[0]

        scaling = self.get_scaling #[n_init_points, 1]
        rotation = self._rotation
        if self.is_gaussian_indexed:
            scaling = scaling[self._gaussian_indices]
            rotation = rotation[self._gaussian_indices]

        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device=self.device)
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(scaling, dim=1).values > self.percent_dense * scene_extent)
        scaling = scaling[selected_pts_mask]
        rotation = rotation[selected_pts_mask]

        stds = scaling.repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device=self.device)
        samples = torch.normal(mean=means, std=stds)

        rots = build_rotation(rotation).repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(stds / (0.8 * N))

        if self._scaling_factor is not None:
            new_scaling_factor = self._scaling_factor[selected_pts_mask].repeat(N, 1)
        else:
            new_scaling_factor = None

        new_rotation = rotation.repeat(N, 1)

        features_dc = self._features_dc[self._feature_indices] if self.is_color_indexed else self._features_dc
        new_features_dc = features_dc[selected_pts_mask].repeat(N, 1, 1)
        if len(self._features_rest) > 0:
            features_rest = self._features_rest[self._feature_indices] if self.is_color_indexed else self._features_rest
            new_features_rest = features_rest[selected_pts_mask].repeat(N, 1, 1)
        else:
            new_features_rest = self._features_rest

        n_new_points = selected_pts_mask.sum() * N
        if self.is_gaussian_indexed: # add new unique indices, linked to newly created gaussians
            new_gaussian_indices = torch.arange(n_new_points, dtype=self._gaussian_indices.dtype, device=self._gaussian_indices.device) + len(self._rotation)
            self._gaussian_indices = torch.cat((self._gaussian_indices, new_gaussian_indices), dim=0)

        if self.is_color_indexed:
            new_feature_indices = torch.arange(n_new_points, dtype=self._feature_indices.dtype, device=self._feature_indices.device) + len(self._features_dc)
            self._feature_indices = torch.cat((self._feature_indices, new_feature_indices), dim=0)


        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling,
                                   new_scaling_factor, new_rotation)
        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device=self.device, dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads=None, grad_threshold=None, scene_extent=None, selected_pts_mask=None, new_xyz=None):
        # Extract points that satisfy the gradient condition
        if selected_pts_mask is None:
            selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
            selected_pts_mask = torch.logical_and(selected_pts_mask,
                                                  torch.max(self.get_scaling, dim=1).values <= self.percent_dense * scene_extent)
            # print("densify_and_clone: ", selected_pts_mask.sum())
        if new_xyz is None:
            new_xyz = self._xyz[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        if self._scaling_factor is not None:
            new_scaling_factor = self._scaling_factor[selected_pts_mask]
        else:
            new_scaling_factor = None

        if self.is_color_indexed:
            print('FCC', len(self._features_dc), self._feature_indices.max())
        if self.is_gaussian_indexed:
            print('RCC', len(self._rotation), self._gaussian_indices.max())

        if self.is_color_indexed:
            new_features_dc = self._features_dc[self._feature_indices][selected_pts_mask]
            new_features_rest = self._features_rest[self._feature_indices][selected_pts_mask] if len(self._features_rest) > 0 else self._features_rest
            print('->', new_features_dc.shape, new_features_rest.shape)
        else:
            new_features_dc = self._features_dc[selected_pts_mask]
            new_features_rest = self._features_rest[selected_pts_mask] if len(self._features_rest) > 0 else self._features_rest

        if self.is_gaussian_indexed:
            new_scaling = self._scaling[self._gaussian_indices][selected_pts_mask]
            new_rotation = self._rotation[self._gaussian_indices][selected_pts_mask]
        else:
            new_scaling = self._scaling[selected_pts_mask]
            new_rotation = self._rotation[selected_pts_mask]

        n_new_points = selected_pts_mask.sum()

        if self.is_gaussian_indexed and n_new_points > 0:
            new_gaussian_indices = torch.arange(n_new_points, dtype=self._gaussian_indices.dtype,
                                                device=self._gaussian_indices.device) + len(self._rotation)
            self._gaussian_indices = torch.cat((self._gaussian_indices, new_gaussian_indices), dim=0)

        if self.is_color_indexed and n_new_points > 0:
            new_feature_indices = torch.arange(n_new_points, dtype=self._feature_indices.dtype,
                                               device=self._feature_indices.device) + len(self._features_dc)
            self._feature_indices = torch.cat((self._feature_indices, new_feature_indices), dim=0)
        if self.is_color_indexed:
            print('FCC.', len(self._features_dc), self._feature_indices.max())
        if self.is_gaussian_indexed:
            print('RCC.', len(self._rotation), self._gaussian_indices.max())
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, \
                                   new_scaling, new_scaling_factor, new_rotation)
        if self.is_color_indexed:
            print('FCC->', len(self._features_dc), self._feature_indices.max())
        if self.is_gaussian_indexed:
            print('RCC->', len(self._rotation), self._gaussian_indices.max())

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0
        self.check_state()
        self.densify_and_clone(grads, max_grad, extent)
        self.check_state()
        self.densify_and_split(grads, max_grad, extent)
        self.check_state()
        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)
        torch.cuda.empty_cache()
    def densify_initial(self):
        # add new points with specified precision
        n = len(self._xyz)
        pp_min = self._xyz.min(dim=0)[0]
        pp_max = self._xyz.max(dim=0)[0]
        volume = torch.prod(pp_max - pp_min).item() / n
        average_step = volume**(1.0 / 3)

        # find 3 nearest neighbours for each point
        k = 3
        # along all axis with length greater than average step, perform n-section
        # find k - nearest neighbours for each xyz and generate n points along the ray
        data = self._xyz.detach().cpu().numpy()

        nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree').fit(data)
        _, indices = nbrs.kneighbors(data)
        idx = torch.arange(n, dtype=torch.long, device=self.device)
        for nb in range(1, k+1):
            delta_pt = data[indices[:, nb]] - data
            # faraway = delta_pt.sqr().sum().sqrt() >= 2 * average_step
            relative_distance = np.sqrt((delta_pt**2.0).sum(axis=1)) / average_step # n x 1
            max_relative_distance = relative_distance.max()
            for dist in range(1, int(max_relative_distance)):
                # slot = (relative_distance >= dist) & (relative_distance < dist + 1) # m x 1
                slot = (relative_distance >= dist+1)
                if slot.sum() > 1:
                    # alpha = torch.Tensor(relative_distance[slot] - dist).to(device=self.device) # m x 1
                    alpha = torch.Tensor(dist / relative_distance[slot]).to(device=self.device)  # m x 1
                    selected = torch.Tensor(indices[slot, nb]).to(dtype=torch.long, device=self.device) # m x 1
                    slot = idx[slot] # n x 1

                    coords = []
                    for i in range(3):
                        coords.append(self._xyz[slot, i] * (1.0 - alpha) + alpha * self._xyz[selected, i])
                    coords = torch.stack(coords, dim=1)
                    self.densify_and_clone(selected_pts_mask=slot, new_xyz=coords)
        print('Densification completed')
        return


    def reset_opacity(self):
        # print("Resetting opacity")
        # print(self.get_opacity.min(), self.get_opacity.max(), '->')
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity) * 0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]
        # print('->', self._opacity.min(), self._opacity.max(), self.get_opacity.min(), self.get_opacity.max())

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter, :2], \
                                                             dim=-1, keepdim=True)
        self.denom[update_filter] += 1


class FakeQuantizationHalf(torch.autograd.Function):
    """performs fake quantization for half precision"""

    @staticmethod
    def forward(_, x: torch.Tensor) -> torch.Tensor:
        return x.half().float()

    @staticmethod
    def backward(_, grad_output: torch.Tensor) -> torch.Tensor:
        return grad_output


def splitBy3(a):
    x = a & 0x1FFFFF  # we only look at the first 21 bits
    x = (x | x << 32) & 0x1F00000000FFFF
    x = (x | x << 16) & 0x1F0000FF0000FF
    x = (x | x << 8) & 0x100F00F00F00F00F
    x = (x | x << 4) & 0x10C30C30C30C30C3
    x = (x | x << 2) & 0x1249249249249249
    return x


def mortonEncode(pos: torch.Tensor, ordering=[0, 1, 2]) -> torch.Tensor:
    arr = pos.unbind(-1)
    x, y, z = arr[ordering[0]], arr[ordering[1]], arr[ordering[2]]
    answer = torch.zeros(len(pos), dtype=torch.long, device=pos.device)
    answer |= splitBy3(x) | splitBy3(y) << 1 | splitBy3(z) << 2
    return answer