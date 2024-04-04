import os
import pickle
import shutil

import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render
import sys
from scene import Scene, GaussianModel as GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams

import numpy as np
from _compress import calc_importance

def training(dataset: ModelParams, opt: OptimizationParams, pipeline: PipelineParams, \
             testing_iterations, saving_iterations):
    prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree, quantization=True, use_factor_scaling=True)
    scene = Scene(dataset, gaussians, load_iteration=-1, shuffle=True, save_memory=False)
    # it's important to run this after scene initialization, not before!
    gaussians.training_setup(opt)
    gaussians.update_learning_rate(0)

    bg = torch.rand((3), device="cuda") if opt.random_background else torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    data_count = len(scene)
    epoch_count = opt.iterations // data_count

    calc_epoch = lambda i: max(1, i * epoch_count // opt.iterations)

    # recalculate all settings in terms of epoch instead of iterations
    saving_epochs = [calc_epoch(iter) for iter in saving_iterations]
    testing_epochs = [calc_epoch(iter) for iter in testing_iterations]

    densify_until_epoch = calc_epoch(opt.densify_until_iter)
    densify_from_epoch = calc_epoch(opt.densify_from_iter)
    densification_interval = calc_epoch(opt.densification_interval)
    opacity_reset_interval = calc_epoch(opt.opacity_reset_interval)
    degree_up = calc_epoch(1000)

    iteration = 0
    for epoch in (progress_bar := tqdm(range(epoch_count), desc="Training progress")):
        epoch_stats = {"loss": 0.0, "ssim": 0.0, "PSNR": 0.0, "N": 0}

        scaling = gaussians.scaling_qa(gaussians.scaling_activation(gaussians._scaling.detach()))
        cov3d = gaussians.covariance_activation(scaling, 1.0, gaussians.get_rotation.detach(), True).requires_grad_(
            True)
        scaling_factor = gaussians.scaling_factor_activation(
            gaussians.scaling_factor_qa(gaussians._scaling_factor.detach()))

        dc_gradient_accum = torch.zeros_like(gaussians._features_dc)
        rest_gradient_accum = torch.zeros_like(gaussians._features_rest)
        cov3d_gradient_accum = torch.zeros_like(cov3d)

        gaussians._features_dc.retain_grad()
        gaussians._features_rest.retain_grad()
        cov3d.retain_grad()

        for viewpoint_cam in scene.getTrainCameras():
            if gaussians._features_dc.grad is not None:
                gaussians._features_dc.grad.zero_()
            if gaussians._features_rest.grad is not None:
                gaussians._features_rest.grad.zero_()
            if cov3d.grad is not None:
                cov3d.grad.zero_()

            gaussians.update_learning_rate(iteration)

            # render_pkg = render(viewpoint_cam, gaussians, pipeline, bg)
            cov3d_scaled = cov3d * scaling_factor.square()
            render_pkg = render(viewpoint_cam, gaussians, pipeline_params, bg, clamp_color=False, cov3d=cov3d_scaled)

            image, viewspace_point_tensor, visibility_filter, radii, visible = (
                render_pkg["render"], render_pkg["viewspace_points"], \
                render_pkg["visibility_filter"], render_pkg["radii"], render_pkg["visible"])
            # Loss
            gt_image = viewpoint_cam.original_image.cuda()
            l1_diff = l1_loss(image, gt_image)
            _ssim = ssim(image, gt_image)
            loss = (1.0 - opt.lambda_dssim) * l1_diff + opt.lambda_dssim * (1.0 - _ssim)
            loss.backward()

            _vis_filter = torch.zeros_like(visible)
            _vis_filter[visible] = visibility_filter
            dc_gradient_accum[_vis_filter, ...] += torch.abs(gaussians._features_dc.grad[_vis_filter, ...])
            rest_gradient_accum[_vis_filter, ...] += torch.abs(gaussians._features_rest.grad[_vis_filter, ...])
            cov3d_gradient_accum[_vis_filter, ...] += torch.abs(cov3d.grad[_vis_filter, ...])

            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log

            iteration_stats = {"loss": loss.item(), "ema_loss": ema_loss_for_log, "ssim": _ssim.item(),
                                      "PSNR": psnr(image, gt_image).mean().item(),
                                      "N": len(gaussians.get_xyz)}

            for k in epoch_stats.keys():
                epoch_stats[k] += iteration_stats[k]

            progress_bar.set_postfix(iteration_stats)
            gaussians.optimizer.step()
            gaussians.optimizer.zero_grad(set_to_none=True)

            if epoch < densify_until_epoch:
                # Keep track of max radii in image-space for pruning
                _vis_filter = torch.zeros_like(visible)
                _vis_filter[visible] = visibility_filter
                gaussians.max_radii2D[_vis_filter] = torch.max(gaussians.max_radii2D[_vis_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, _vis_filter)
            iteration += 1

        importance = torch.cat([dc_gradient_accum, rest_gradient_accum], 1).flatten(-2)
        deriv_color_importance, deriv_gaussian_sensitivity = importance.detach() / data_count, cov3d_gradient_accum.detach() / data_count
        print('1', deriv_color_importance.max(), deriv_gaussian_sensitivity.max(), deriv_color_importance.shape, deriv_gaussian_sensitivity.shape)
        color_importance, gaussian_sensitivity = calc_importance(gaussians, scene, pipeline, silent=True)
        print('2', color_importance.max(), gaussian_sensitivity.max(), color_importance.shape, gaussian_sensitivity.shape)

        if epoch in testing_epochs:
            print(f"\n[EPOCH {epoch}] " + ",".join([f"{k}: {v / data_count:.4f}" for k, v in epoch_stats.items()]))
        with torch.no_grad():
            if epoch in saving_epochs:
                # print(f"\n[EPOCH {epoch}] Saving Gaussians")
                scene.save(epoch)

            # Densification
            if epoch < densify_until_epoch:
                if epoch > densify_from_epoch and epoch % densification_interval == 0:
                    # print(f"\n[EPOCH {epoch}] Dense and prune")
                    size_threshold = 20 if epoch > opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)

                if epoch > 0 and epoch % opacity_reset_interval == 0:
                    # print(f"\n[EPOCH {epoch}] Resetting opacity")
                    gaussians.reset_opacity()



        # Every 1000 its we increase the levels of SH up to a maximum degree
        if epoch % degree_up == 0:
            gaussians.oneupSHdegree()

def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str = os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print(f"Output folder: {args.model_path}")

    if os.path.exists(args.model_path):
        shutil.rmtree(args.model_path)

    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

def training_report(iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene: Scene, renderFunc,
                    renderArgs):
    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras': scene.getTestCameras()},
                              {'name': 'train',
                               'cameras': [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in
                                           range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)

                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
        torch.cuda.empty_cache()


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)

    args = parser.parse_args(sys.argv[1:])

    print("Optimizing " + args.model_path)
    os.makedirs(args.model_path, exist_ok=True) # Create output folder

    # Initialize system state (RNG)
    # safe_state(args.quiet)

    # Start GUI server, configure and run training
    torch.autograd.set_detect_anomaly(False)

    model_params = lp.extract(args)
    optim_params = op.extract(args)
    pipeline_params = pp.extract(args)

    training(model_params, optim_params, pipeline_params,
             [1_000, 3_000, 7_000, 15_000, 30_000],
             [1_000, 3_000, 7_000, 15_000, 30_000])

