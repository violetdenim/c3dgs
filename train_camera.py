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
from matplotlib import pyplot as plt
import numpy as np

def training(dataset: ModelParams, opt: OptimizationParams, comp_params: CompressionParams):
    device = "cuda" # "cpu"
    prepare_output_and_logger(dataset)
    dataset.data_device = device
    gaussians = GaussianModel(dataset.sh_degree, quantization=True, use_factor_scaling=True, device=device, is_splitted=True)
    scene = Scene(dataset, gaussians, load_iteration=-1, shuffle=False, save_memory=False)
    gaussians.training_setup(opt)
    # implementing morton sorting
    gaussians.densify_initial()#0.1)
    gaussians.to_indexed()
    gaussians._sort_morton()

    # it's important to run this after scene initialization, not before!
    # training also should be set up after all sorting indicing etc.
    gaussians.training_setup(opt)
    gaussians.update_learning_rate(0)

    bg = torch.rand((3), device=device) if opt.random_background else torch.tensor([0, 0, 0], dtype=torch.float32, device=device)
    data_count = len(scene)
    print(f"Data count: {data_count}")
    epoch_count = 5000 # opt.iterations // data_count

    # recalculate all settings in terms of epoch instead of iterations
    saving_epochs = range(0, epoch_count, 1000)
    testing_epochs = range(0, epoch_count, 1000)

    degree_up = 1

    iteration = 0
    #DEBUG ONLY!
    data_step = 1 #10
    show_img = True


    metric_keys = ["loss"]
    full_stats = {k: [] for k in metric_keys}
    image_axis = None

    lr_cam = 1e-03#1e-4
    original_extrinsics = []
    param_groups = []
    for i, viewpoint_cam in enumerate(scene.getTrainCameras()[0:1]):
        param_groups.append({"params": [viewpoint_cam.extrinsic], "lr": lr_cam, "name": f"extr{i}"})
        original_extrinsics.append(viewpoint_cam.extrinsic.clone())
        # spoil initial solution:
        viewpoint_cam.extrinsic += 0.1 * (torch.rand((4, 4), device=gaussians.device) - 0.5)
        viewpoint_cam.extrinsic.requires_grad_(True)
        viewpoint_cam.extrinsic.retain_grad()

    # for group in param_groups:
    #     gaussians.optimizer.add_param_group(group)
    gaussians.optimizer = torch.optim.Adam(param_groups)

    for epoch in (progress_bar := tqdm(range(epoch_count), desc="Training progress")):
        epoch_stats = {key: 0.0 for key in metric_keys}
        num_pixels = 0

        for i_camera, viewpoint_cam in enumerate(scene.getTrainCameras()[0:1]):
            # gaussians.update_learning_rate(iteration)

            render_pkg = gaussians.render(viewpoint_cam, pipeline_params, bg)

            image, viewspace_point_tensor, visibility_filter, radii, visible = (
                render_pkg["render"], render_pkg["viewspace_points"],
                render_pkg["visibility_filter"], render_pkg["radii"],
                render_pkg["visible"])
            if iteration % data_count == 0 and epoch % 10 == 0 and show_img:
                show_image = np.clip(image.detach().cpu().numpy().transpose(1, 2, 0), 0, 1)
                if image_axis is None:
                    plt.ion()
                    fig = plt.figure()
                    image_axis = plt.imshow(show_image)
                else:
                    image_axis.set_data(show_image)
                    plt.draw()
                    fig.canvas.flush_events()

            # Loss
            gt_image = viewpoint_cam.original_image.to(device)
            l1_diff = l1_loss(image, gt_image)
            _ssim = ssim(image, gt_image)
            loss = (1.0 - opt.lambda_dssim) * l1_diff + opt.lambda_dssim * (1.0 - _ssim)
            loss.backward()

            #loss = torch.exp(torch.abs(original_extrinsics[i_camera] - viewpoint_cam.extrinsic)).sum()
            # loss = torch.exp(torch.mul(original_extrinsics[i_camera], viewpoint_cam.extrinsic)).sum()
            # loss.backward()
            print(loss.item(), viewpoint_cam.extrinsic.grad)
            # print(loss2.item(), viewpoint_cam.extrinsic)


            # Progress bar
            iteration_stats = {"loss": loss.item()}

            for k in epoch_stats.keys():
                epoch_stats[k] += iteration_stats[k]

            progress_bar.set_postfix(iteration_stats)
            gaussians.optimizer.step()
            gaussians.optimizer.zero_grad(set_to_none=True)
            # if viewpoint_cam.extrinsic.grad is not None:
            #     viewpoint_cam.extrinsic.grad.zero_()

            iteration += 1
            num_pixels += image.shape[1] * image.shape[2]


        if epoch in testing_epochs:
            print(f"\n[EPOCH {epoch}] " + ",".join([f"{k}: {v/(data_count/data_step):.4f}" for k, v in epoch_stats.items()]))

        for k, v in epoch_stats.items():
            full_stats[k].append(v)

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

    training(model_params, optim_params, comp_params)

