import os
import shutil

import torch
from utils.loss_utils import l1_loss, ssim
import sys
from scene import Scene, GaussianModel as GaussianModel
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams, CompressionParams
from compression.vq import CompressionSettings, compress_gaussians

def training(dataset: ModelParams, opt: OptimizationParams, pipeline: PipelineParams, \
             comp_params: CompressionParams,
             testing_iterations, saving_iterations):
    prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree, quantization=True, use_factor_scaling=True)
    scene = Scene(dataset, gaussians, load_iteration=-1, shuffle=True, save_memory=False)
    # it's important to run this after scene initialization, not before!
    gaussians.training_setup(opt)
    gaussians.update_learning_rate(0)
    # experiment: train in indexed mode
    # gaussians.to_indexed()

    bg = torch.rand((3), device="cuda") if opt.random_background else torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    viewpoint_stack = None
    ema_loss_for_log = 0.0
    data_count = len(scene)
    epoch_count = 10 # opt.iterations // data_count
    epoch_compression = 4

    calc_epoch = lambda i: max(1, i * epoch_count // opt.iterations)

    # recalculate all settings in terms of epoch instead of iterations
    saving_epochs = range(epoch_count)
    testing_epochs = range(epoch_count)

    degree_up = 1

    iteration = 0
    #DEBUG ONLY!
    data_step = 10

    for epoch in (progress_bar := tqdm(range(epoch_count), desc="Training progress")):
        epoch_stats = {"loss": 0.0, "ssim": 0.0, "PSNR": 0.0, "N": 0}

        calc_compression_stats = (epoch == epoch_compression)
        if calc_compression_stats:
            gaussians.to_unindexed()
            dc_gradient_accum       = torch.zeros_like(gaussians._features_dc).requires_grad_(False)
            rest_gradient_accum     = torch.zeros_like(gaussians._features_rest).requires_grad_(False)
            cov3d_gradient_accum    = torch.zeros_like(gaussians.get_covariance()).requires_grad_(False)

        num_pixels = 0
        for viewpoint_cam in scene.getTrainCameras()[::data_step]:
            gaussians.update_learning_rate(iteration)

            # render_pkg = render(viewpoint_cam, gaussians, pipeline, bg)
            cov3d_scaled = gaussians.get_covariance().detach()
            scaling_factor = gaussians.get_scaling_factor.detach()
            coeff = scaling_factor.square()
            cov3d = (cov3d_scaled / coeff).requires_grad_(True)

            render_pkg = gaussians.render(viewpoint_cam, pipeline_params, bg, clamp_color=False, cov3d=cov3d * coeff)

            image, viewspace_point_tensor, visibility_filter, radii, visible = (
                render_pkg["render"], render_pkg["viewspace_points"],
                render_pkg["visibility_filter"], render_pkg["radii"],
                render_pkg["visible"])

            # Loss
            gt_image = viewpoint_cam.original_image.cuda()
            l1_diff = l1_loss(image, gt_image)
            _ssim = ssim(image, gt_image)
            loss = (1.0 - opt.lambda_dssim) * l1_diff + opt.lambda_dssim * (1.0 - _ssim)
            loss.backward()
            if calc_compression_stats:
                _vis_filter = torch.zeros_like(visible)
                _vis_filter[visible] = visibility_filter
                dc_gradient_accum[_vis_filter, ...]     += torch.abs(gaussians._features_dc.grad[_vis_filter, ...])
                rest_gradient_accum[_vis_filter, ...]   += torch.abs(gaussians._features_rest.grad[_vis_filter, ...])
                cov3d_gradient_accum[_vis_filter, ...]  += torch.abs(cov3d.grad[_vis_filter, ...])

            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log

            iteration_stats = {"loss": loss.item(), "ema_loss": ema_loss_for_log, "ssim": _ssim.item(),
                                      "PSNR": psnr(image, gt_image).mean().item(), "N": len(gaussians.get_xyz)}

            for k in epoch_stats.keys():
                epoch_stats[k] += iteration_stats[k]

            progress_bar.set_postfix(iteration_stats)
            gaussians.optimizer.step()
            gaussians.optimizer.zero_grad(set_to_none=True)
            cov3d.grad.zero_()

            iteration += 1

            num_pixels += image.shape[1] * image.shape[2]

        if calc_compression_stats:
            color_importance = torch.cat([dc_gradient_accum, rest_gradient_accum], 1).flatten(-2) / num_pixels
            gaussian_sensitivity = cov3d_gradient_accum.detach() / num_pixels

            color_importance_n = color_importance.amax(-1)
            gaussian_importance_n = gaussian_sensitivity.amax(-1)

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
            # gaussians.check_state()
            # n_initial = gaussians._xyz.shape[0]
            print('Compression')
            compress_gaussians(gaussians, color_importance_n, gaussian_importance_n,
                               color_compression_settings if not comp_params.not_compress_color else None,
                               gaussian_compression_settings if not comp_params.not_compress_gaussians else None,
                               comp_params.color_compress_non_dir,
                               prune_threshold=-1,#comp_params.prune_threshold,
                               silent=True)
            gaussians.check_state()
        # n_compressed = gaussians._xyz.shape[0]
        # gaussians.to_unindexed() # always uncompress back - so only unification is actually performed
        # n_uncompressed = gaussians._xyz.shape[0]
        # print(n_initial, n_compressed, n_uncompressed, gaussians.xyz_gradient_accum.shape, gaussians.denom.shape)
        # gaussians.check_state()

        if epoch in testing_epochs:
            print(f"\n[EPOCH {epoch}] " + ",".join([f"{k}: {v/(data_count/data_step):.4f}" for k, v in epoch_stats.items()]))
        with torch.no_grad():
            if epoch in saving_epochs:
                scene.save(epoch)
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
    cp = CompressionParams(parser)

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
    comp_params = cp.extract(args)

    training(model_params, optim_params, pipeline_params, comp_params,
             [1_000, 3_000, 7_000, 15_000, 30_000],
             [1_000, 3_000, 7_000, 15_000, 30_000])

