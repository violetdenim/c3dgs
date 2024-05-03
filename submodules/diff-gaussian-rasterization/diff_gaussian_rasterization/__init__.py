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

from typing import NamedTuple
import torch.nn as nn
import torch
import math
from . import _C

# TOOD add indexed methods

def getProjectionMatrix(intrinsic):
    znear, zfar, z_sign = 0.01, 100.0, 1.0

    tanHalfFovY = math.tan((intrinsic[1, 1] / 2))
    tanHalfFovX = math.tan((intrinsic[0, 0] / 2))

    return torch.Tensor([
        [1.0 / tanHalfFovX, 0.0, 0.0, 0.0],
        [0.0, 1.0 / tanHalfFovY, 0.0, 0.0],
        [0.0, 0.0, z_sign * zfar / (zfar - znear), -(zfar * znear) / (zfar - znear)],
        [0.0, 0.0, z_sign, 0.0]
    ]).transpose(0, 1).cuda()

def cpu_deep_copy_tuple(input_tuple):
    copied_tensors = [
        item.cpu().clone() if isinstance(item, torch.Tensor) else item
        for item in input_tuple
    ]
    return tuple(copied_tensors)


def rasterize_gaussians(
    means3D,
    means2D,
    sh,
    colors_precomp,
    opacities,
    scales,
    rotations,
    cov3Ds_precomp,
    raster_settings,
    # extrinsic
):
    return _RasterizeGaussians.apply(
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
        # extrinsic
    )


def rasterize_gaussians_indexed(
    means3D,
    means2D,
    sh,
    sh_indices,
    g_indices,
    colors_precomp,
    opacities,
    scales,
    scale_factors,
    rotations,
    cov3Ds_precomp,
    raster_settings,
    # extrinsic
):
    return _RasterizeGaussiansIndexed.apply(
        means3D,
        means2D,
        sh,
        sh_indices,
        g_indices,
        colors_precomp,
        opacities,
        scales,
        scale_factors,
        rotations,
        cov3Ds_precomp,
        raster_settings,
        # extrinsic
    )


class _RasterizeGaussians(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
        # extrinsic,
    ):
        # Restructure arguments the way that the C++ lib expects them
        tanfovx = float(math.tan(raster_settings.intrinsic[0, 0] * 0.5))
        tanfovy = float(math.tan(raster_settings.intrinsic[1, 1] * 0.5))
        image_height = int(raster_settings.intrinsic[1, 2])
        image_width = int(raster_settings.intrinsic[0, 2])
        extrinsic = raster_settings.extrinsic

        args = (
            raster_settings.bg,
            means3D,
            colors_precomp,
            opacities,
            scales,
            rotations,
            raster_settings.scale_modifier,
            cov3Ds_precomp,
            extrinsic,
            extrinsic @ getProjectionMatrix(raster_settings.intrinsic),
            tanfovx, tanfovy, image_height, image_width,
            sh,
            raster_settings.sh_degree,
            extrinsic.inverse()[3, :3],
            raster_settings.prefiltered,
            raster_settings.debug,
            raster_settings.clamp_color,
        )

        # Invoke C++/CUDA rasterizer
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(
                args
            )  # Copy them before they can be corrupted
            try:
                (
                    num_rendered,
                    color,
                    radii,
                    geomBuffer,
                    binningBuffer,
                    imgBuffer,
                ) = _C.rasterize_gaussians(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_fw.dump")
                print(
                    "\nAn error occured in forward. Please forward snapshot_fw.dump for debugging."
                )
                raise ex
        else:
            (
                num_rendered,
                color,
                radii,
                geomBuffer,
                binningBuffer,
                imgBuffer,
            ) = _C.rasterize_gaussians(*args)

        # Keep relevant tensors for backward
        ctx.raster_settings = raster_settings
        ctx.num_rendered = num_rendered
        ctx.save_for_backward(
            # extrinsic,
            colors_precomp,
            means3D,
            scales,
            rotations,
            cov3Ds_precomp,
            radii,
            sh,
            geomBuffer,
            binningBuffer,
            imgBuffer,
        )
        return color, radii#, extrinsic

    @staticmethod
    def backward(ctx, grad_out_color, *params):
        # Restore necessary values from context
        num_rendered = ctx.num_rendered
        raster_settings = ctx.raster_settings
        (
            # extrinsic,
            colors_precomp,
            means3D,
            scales,
            rotations,
            cov3Ds_precomp,
            radii,
            sh,
            geomBuffer,
            binningBuffer,
            imgBuffer,
        ) = ctx.saved_tensors

        # Restructure args as C++ method expects them
        tanfovx = float(math.tan(raster_settings.intrinsic[0, 0] * 0.5))
        tanfovy = float(math.tan(raster_settings.intrinsic[1, 1] * 0.5))
        image_height = int(raster_settings.intrinsic[1, 2])
        image_width = int(raster_settings.intrinsic[0, 2])

        #extrinsic = params[1] # !!!
        extrinsic = raster_settings.extrinsic
        args = (
            raster_settings.bg,
            means3D,
            radii,
            colors_precomp,
            scales,
            rotations,
            raster_settings.scale_modifier,
            cov3Ds_precomp,
            extrinsic,
            extrinsic @ getProjectionMatrix(raster_settings.intrinsic), #raster_settings.projmatrix,
            # raster_settings.tanfovx,
            # raster_settings.tanfovy,
            tanfovx, tanfovy,
            grad_out_color,
            sh,
            raster_settings.sh_degree,
            extrinsic.inverse()[3, :3],#raster_settings.campos,
            geomBuffer,
            num_rendered,
            binningBuffer,
            imgBuffer,
            raster_settings.debug,
        )

        # Compute gradients for relevant tensors by invoking backward method
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(
                args
            )  # Copy them before they can be corrupted
            try:
                (
                    grad_means2D,
                    grad_colors_precomp,
                    grad_opacities,
                    grad_means3D,
                    grad_cov3Ds_precomp,
                    grad_sh,
                    grad_scales,
                    grad_rotations,
                    # grad_matrix,
                ) = _C.rasterize_gaussians_backward(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_bw.dump")
                print(
                    "\nAn error occured in backward. Writing snapshot_bw.dump for debugging.\n"
                )
                raise ex
        else:
            (
                grad_means2D,
                grad_colors_precomp,
                grad_opacities,
                grad_means3D,
                grad_cov3Ds_precomp,
                grad_sh,
                grad_scales,
                grad_rotations,
                # grad_matrix,
            ) = _C.rasterize_gaussians_backward(*args)

        grads = (
            grad_means3D,
            grad_means2D,
            grad_sh,
            grad_colors_precomp,
            grad_opacities,
            grad_scales,
            grad_rotations,
            grad_cov3Ds_precomp,
            None,
            # grad_matrix,
        )

        return grads


class _RasterizeGaussiansIndexed(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        means3D,
        means2D,
        sh,
        sh_indices,
        g_inidices,
        colors_precomp,
        opacities,
        scales,
        scale_factors,
        rotations,
        cov3Ds_precomp,
        raster_settings
        # extrinsic
    ):
        # Restructure arguments the way that the C++ lib expects them
        tanfovx = float(math.tan(raster_settings.intrinsic[0, 0] * 0.5))
        tanfovy = float(math.tan(raster_settings.intrinsic[1, 1] * 0.5))
        image_height = int(raster_settings.intrinsic[1, 2])
        image_width = int(raster_settings.intrinsic[0, 2])

        # extrinsic.requires_grad = True
        extrinsic = raster_settings.extrinsic
        args = (
            raster_settings.bg,
            means3D,
            colors_precomp,
            opacities,
            scales,
            scale_factors,
            rotations,
            raster_settings.scale_modifier,
            cov3Ds_precomp,
            extrinsic,
            extrinsic @ getProjectionMatrix(raster_settings.intrinsic), #raster_settings.projmatrix,
            tanfovx, tanfovy, image_height, image_width,
            sh,
            raster_settings.sh_degree,
            extrinsic.inverse()[3, :3],#raster_settings.campos,
            sh_indices,
            g_inidices,
            raster_settings.prefiltered,
            raster_settings.debug,
            raster_settings.clamp_color,
        )

        # Invoke C++/CUDA rasterizer
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(
                args
            )  # Copy them before they can be corrupted
            try:
                (
                    num_rendered,
                    color,
                    radii,
                    geomBuffer,
                    binningBuffer,
                    imgBuffer,
                ) = _C.rasterize_gaussians_indexed(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_fw.dump")
                print(
                    "\nAn error occured in forward. Please forward snapshot_fw.dump for debugging."
                )
                raise ex
        else:
            (
                num_rendered,
                color,
                radii,
                geomBuffer,
                binningBuffer,
                imgBuffer,
            ) = _C.rasterize_gaussians_indexed(*args)

        # Keep relevant tensors for backward
        ctx.raster_settings = raster_settings
        ctx.num_rendered = num_rendered
        ctx.save_for_backward(
            # extrinsic,
            colors_precomp,
            means3D,
            scales,
            scale_factors,
            rotations,
            cov3Ds_precomp,
            radii,
            sh,
            geomBuffer,
            binningBuffer,
            imgBuffer,
            sh_indices,
            g_inidices,
        )
        return color, radii#, extrinsic

    @staticmethod
    def backward(ctx, grad_out_color, *params):
        # Restore necessary values from context
        num_rendered = ctx.num_rendered
        raster_settings = ctx.raster_settings
        (
            # extrinsic,
            colors_precomp,
            means3D,
            scales,
            scale_factors,
            rotations,
            cov3Ds_precomp,
            radii,
            sh,
            geomBuffer,
            binningBuffer,
            imgBuffer,
            sh_indices,
            g_inidices,
        ) = ctx.saved_tensors
        tanfovx = float(math.tan(raster_settings.intrinsic[0, 0] * 0.5))
        tanfovy = float(math.tan(raster_settings.intrinsic[1, 1] * 0.5))

        extrinsic = raster_settings.extrinsic
        # Restructure args as C++ method expects them
        args = (
            raster_settings.bg,
            means3D,
            radii,
            colors_precomp,
            scales,
            scale_factors,
            rotations,
            raster_settings.scale_modifier,
            cov3Ds_precomp,
            extrinsic,
            extrinsic @ getProjectionMatrix(raster_settings.intrinsic), #raster_settings.projmatrix,
            # raster_settings.tanfovx,
            # raster_settings.tanfovy,
            tanfovx, tanfovy,
            grad_out_color,
            sh,
            raster_settings.sh_degree,
            extrinsic.inverse()[3, :3],#raster_settings.campos,
            geomBuffer,
            num_rendered,
            binningBuffer,
            imgBuffer,
            raster_settings.debug,
            sh_indices,
            g_inidices,
        )

        # Compute gradients for relevant tensors by invoking backward method
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(
                args
            )  # Copy them before they can be corrupted
            try:
                (
                    grad_means2D,
                    grad_colors_precomp,
                    grad_opacities,
                    grad_means3D,
                    grad_cov3Ds_precomp,
                    grad_sh,
                    grad_scales,
                    grad_scale_factors,
                    grad_rotations,
                    # grad_matrix,
                ) = _C.rasterize_gaussians_backward_indexed(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_bw.dump")
                print(
                    "\nAn error occured in backward. Writing snapshot_bw.dump for debugging.\n"
                )
                raise ex
        else:
            (
                grad_means2D,
                grad_colors_precomp,
                grad_opacities,
                grad_means3D,
                grad_cov3Ds_precomp,
                grad_sh,
                grad_scales,
                grad_scale_factors,
                grad_rotations,
                # grad_matrix,
            ) = _C.rasterize_gaussians_backward_indexed(*args)

        grads = (
            grad_means3D,
            grad_means2D,
            grad_sh,
            None,
            None,
            grad_colors_precomp,
            grad_opacities,
            grad_scales,
            grad_scale_factors,
            grad_rotations,
            grad_cov3Ds_precomp,
            None,
            # grad_matrix,
        )
        return grads


class GaussianRasterizationSettings(NamedTuple):
    intrinsic: torch.Tensor
    extrinsic: torch.Tensor

    bg: torch.Tensor
    scale_modifier: float
    sh_degree: int
    prefiltered: bool
    debug: bool
    clamp_color: bool


class GaussianRasterizer(nn.Module):
    def __init__(self, raster_settings):
        super().__init__()
        self.raster_settings = raster_settings

    def markVisible(self, positions, extrinsic):
        # Mark visible points (based on frustum culling for camera) with a boolean
        with torch.no_grad():
            raster_settings = self.raster_settings
            visible = _C.mark_visible(
                positions, extrinsic, extrinsic @ getProjectionMatrix(raster_settings.intrinsic)
            )
        return visible

    def forward(
        self,
        means3D,
        means2D,
        opacities,
        shs=None,
        colors_precomp=None,
        scales=None,
        rotations=None,
        cov3D_precomp=None,
        extrinsic=None,
    ):
        raster_settings = self.raster_settings

        if (shs is None and colors_precomp is None) or (
            shs is not None and colors_precomp is not None
        ):
            raise Exception(
                "Please provide excatly one of either SHs or precomputed colors!"
            )

        if ((scales is None or rotations is None) and cov3D_precomp is None) or (
            (scales is not None or rotations is not None) and cov3D_precomp is not None
        ):
            raise Exception(
                "Please provide exactly one of either scale/rotation pair or precomputed 3D covariance!"
            )

        if shs is None:
            shs = torch.Tensor([])
        if colors_precomp is None:
            colors_precomp = torch.Tensor([])

        if scales is None:
            scales = torch.Tensor([])
        if rotations is None:
            rotations = torch.Tensor([])
        if cov3D_precomp is None:
            cov3D_precomp = torch.Tensor([])

        # Invoke C++/CUDA rasterization routine
        return rasterize_gaussians(
            means3D,
            means2D,
            shs,
            colors_precomp,
            opacities,
            scales,
            rotations,
            cov3D_precomp,
            raster_settings,
            # extrinsic=extrinsic
        )


class GaussianRasterizerIndexed(nn.Module):
    def __init__(self, raster_settings):
        super().__init__()
        self.raster_settings = raster_settings

    def markVisible(self, positions, extrinsic):
        # Mark visible points (based on frustum culling for camera) with a boolean
        with torch.no_grad():
            raster_settings = self.raster_settings
            visible = _C.mark_visible(
                positions, extrinsic, extrinsic @ getProjectionMatrix(raster_settings.intrinsic), #raster_settings.projmatrix,
            )

        return visible

    def forward(
        self,
        means3D,
        means2D,
        opacities,
        sh_indices,
        g_indices,
        shs=None,
        colors_precomp=None,
        scales=None,
        scale_factors=None,
        rotations=None,
        cov3D_precomp=None,
        # extrinsic=None
    ):
        raster_settings = self.raster_settings

        if (shs is None and colors_precomp is None) or (
            shs is not None and colors_precomp is not None
        ):
            raise Exception(
                "Please provide excatly one of either SHs or precomputed colors!"
            )

        if ((scales is None or rotations is None) and cov3D_precomp is None) or (
            (scales is not None or rotations is not None) and cov3D_precomp is not None
        ):
            raise Exception(
                "Please provide exactly one of either scale/rotation pair or precomputed 3D covariance!"
            )

        if shs is None:
            shs = torch.Tensor([])
        if colors_precomp is None:
            colors_precomp = torch.Tensor([])

        if scales is None:
            scales = torch.Tensor([])
        if scale_factors is None:
            scale_factors = torch.Tensor([])
        if rotations is None:
            rotations = torch.Tensor([])
        if cov3D_precomp is None:
            cov3D_precomp = torch.Tensor([])

        # Invoke C++/CUDA rasterization routine
        return rasterize_gaussians_indexed(
            means3D,
            means2D,
            shs,
            sh_indices,
            g_indices,
            colors_precomp,
            opacities,
            scales,
            scale_factors,
            rotations,
            cov3D_precomp,
            raster_settings,
            # extrinsic
        )
