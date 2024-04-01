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
import numpy as np
import torch
import math
from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
    GaussianRasterizerIndexed,
)
from scene.gaussian_model import ColorMode, GaussianModel
from utils.sh_utils import eval_sh


def render(
    viewpoint_camera,
    pc: GaussianModel,
    pipe,
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
    screenspace_points = (torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0)
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
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
        clamp_color=clamp_color,
    )

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    if pc.color_index_mode == ColorMode.ALL_INDEXED and pc.is_gaussian_indexed:
        rasterizer = GaussianRasterizerIndexed(raster_settings=raster_settings)
        shs = pc._get_features_raw
        scales = pc.get_scaling_normalized
        scale_factors = pc.get_scaling_factor
        rotations = pc._rotation_post_activation

        # Rasterize visible Gaussians to image, obtain their radii (on screen).
        # rendered_image, radii = rasterizer(
        #     means3D=means3D,
        #     means2D=means2D,
        #     shs=shs,
        #     sh_indices=pc._feature_indices,
        #     g_indices=pc._gaussian_indices,
        #     colors_precomp=None,
        #     opacities=opacity,
        #     scales=scales,
        #     scale_factors=scale_factors,
        #     rotations=rotations,
        #     cov3D_precomp=None,
        # )

        # precalculate visible points to speed up rasterization
        visible = rasterizer.markVisible(pc.get_xyz)
        rendered_image, radii = rasterizer(
            means3D=means3D[visible, :],
            means2D=means2D[visible, :],
            shs=shs,
            sh_indices=pc._feature_indices[visible],
            g_indices=pc._gaussian_indices[visible],
            colors_precomp=None,
            opacities=opacity[visible],
            scales=scales, #[visible, :] if scales is not None else None,
            scale_factors=scale_factors[visible, :],
            rotations=rotations, #[visible, :] if rotations is not None else None,
            cov3D_precomp=None,
        )
    else:
        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
        # scaling / rotation by the rasterizer.
        scales = None
        rotations = None
        cov3D_precomp = cov3d
        if cov3D_precomp is None:
            if pipe.compute_cov3D_python:
                cov3D_precomp = pc.get_covariance(scaling_modifier)
            else:
                scales = pc.get_scaling
                rotations = pc.get_rotation

        # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
        # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
        shs = None
        colors_precomp = None
        if override_color is None:
            if pipe.convert_SHs_python:
                shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2)
                dir_pp = pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1)
                dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
                sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
                colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
            else:
                shs = pc.get_features
        else:
            colors_precomp = override_color

        # # precalculate visible points to speed up rasterization
        # visible = rasterizer.markVisible(pc.get_xyz)
        # # Rasterize visible Gaussians to image, obtain their radii (on screen).
        # rendered_image, radii = rasterizer(
        #     means3D=means3D[visible, :],
        #     means2D=means2D[visible, :],
        #     shs=shs[visible, :, :],
        #     colors_precomp=colors_precomp[visible, :] if colors_precomp is not None else None,
        #     opacities=opacity[visible, :],
        #     scales=scales[visible, :] if scales is not None else None,
        #     rotations=rotations[visible, :] if rotations is not None else None,
        #     cov3D_precomp=cov3D_precomp[visible, :] if cov3D_precomp is not None else None,
        # )
        visible = torch.ones(pc.get_xyz.shape[0], dtype=torch.bool, device=pc.get_xyz.device)
        rendered_image, radii = rasterizer(
            means3D=means3D, means2D=means2D, shs=shs, colors_precomp=colors_precomp,
            opacities=opacity, scales=scales, rotations=rotations, cov3D_precomp=cov3D_precomp
        )

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter": radii > 0,
        "radii": radii,
        "visible": visible
    }
