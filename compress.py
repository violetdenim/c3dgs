# %%
import gc
import json
import os
import time
import uuid
from argparse import ArgumentParser, Namespace
from os import path
from shutil import copyfile
from typing import Dict, Tuple
from scene.cameras import Camera

import numpy as np
import torch
from time import sleep
from tqdm import tqdm
from utils.loss_utils import l1_loss, ssim
from utils.image_utils import psnr
import torchvision
from matplotlib import pyplot as plt

# %%
from arguments import (
    CompressionParams,
    ModelParams,
    OptimizationParams,
    PipelineParams,
    get_combined_args,
)
from compression.vq import CompressionSettings, compress_gaussians
from scene.gaussian_model import GaussianModel
from lpipsPyTorch import lpips
from scene import Scene
from finetune import finetune

def unique_output_folder():
    if os.getenv("OAR_JOB_ID"):
        unique_str = os.getenv("OAR_JOB_ID")
    else:
        unique_str = str(uuid.uuid4())
    return os.path.join("./output_vq/", unique_str[0:10])


def calc_importance(gaussians: GaussianModel, scene: Scene, pipeline_params, silent=False) -> Tuple[torch.Tensor, torch.Tensor]:
    # gaussians.zero_grad()
    scaling = gaussians.scaling_qa(gaussians.scaling_activation(gaussians._scaling.detach()))
    cov3d = gaussians.covariance_activation(scaling, 1.0, gaussians.get_rotation.detach(), True).requires_grad_(True)
    scaling_factor = gaussians.scaling_factor_activation(gaussians.scaling_factor_qa(gaussians._scaling_factor.detach()))
    # hook is called on gradients each time gaussians are called
    h1 = gaussians._features_dc.register_hook(lambda grad: grad.abs())
    h2 = gaussians._features_rest.register_hook(lambda grad: grad.abs())
    h3 = cov3d.register_hook(lambda grad: grad.abs())
    background = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device="cuda")

    # gaussians._features_dc.retain_grad()
    # gaussians._features_rest.retain_grad()
    # cov3d.retain_grad()

    num_pixels = 0

    iterator, name = scene.getSomeCameras()
    if not silent:
        iterator = tqdm(iterator, desc="Calculating importance")
    for camera in iterator:
        cov3d_scaled = cov3d * scaling_factor.square()
        rendering = gaussians.render(camera, pipeline_params, background, clamp_color=False, cov3d=cov3d_scaled)["render"]

        # gradients are accumulated during cycle each time backward is called
        loss = rendering.sum()
        loss.backward()
        num_pixels += rendering.shape[1] * rendering.shape[2]

    importance = torch.cat([gaussians._features_dc.grad, gaussians._features_rest.grad], 1).flatten(-2)
    cov_grad = cov3d.grad
    h1.remove()
    h2.remove()
    h3.remove()
    torch.cuda.empty_cache()
    return importance.detach() / num_pixels, cov_grad.detach() / num_pixels

def calc_importance_experimental(gaussians: GaussianModel, scene: Scene, pipeline_params: PipelineParams, silent=False, use_gt=False) -> Tuple[torch.Tensor, torch.Tensor]:
    cov3d_scaled = gaussians.get_covariance().detach()
    scaling_factor = gaussians.get_scaling_factor.detach()
    coeff = scaling_factor.square()
    cov3d = (cov3d_scaled / coeff).requires_grad_(True)

    accum1 = torch.zeros_like(gaussians._features_dc).requires_grad_(False)
    accum2 = torch.zeros_like(gaussians._features_rest).requires_grad_(False)
    accum3 = torch.zeros_like(cov3d).requires_grad_(False)
    background = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device="cuda")

    num_pixels = 0
    iterator, name = scene.getSomeCameras()
    if not silent:
        iterator = tqdm(iterator, desc="Calculating importance")

    for camera in iterator:
        gaussians.zero_grad()  # all grads to zero
        if cov3d.grad is not None:
            cov3d.grad.zero_()
        image = gaussians.render(camera, pipeline_params, background, clamp_color=False, cov3d=cov3d * coeff)["render"]
        # gradients are accumulated during cycle each time backward is called
        if not use_gt:
            image.sum().backward()
        else:
            lambda_dssim = 0.2
            gt_image = camera.original_image.cuda()
            loss = (1.0 - lambda_dssim) * l1_loss(image, gt_image) + lambda_dssim * (1.0 - ssim(image, gt_image))
            loss.backward()

        accum1 += torch.abs(gaussians._features_dc.grad)
        accum2 += torch.abs(gaussians._features_rest.grad)
        accum3 += torch.abs(cov3d.grad)
        num_pixels += image.shape[1] * image.shape[2]

    importance = torch.cat([accum1, accum2], 1).flatten(-2)
    cov_grad = accum3
    torch.cuda.empty_cache()
    return importance / num_pixels, cov_grad / num_pixels

def render_and_eval(
    gaussians: GaussianModel,
    scene: Scene,
    model_params: ModelParams,
    pipeline_params: PipelineParams,
    iteration: int
) -> Dict[str, float]:
    with torch.no_grad():
        ssims = []
        psnrs = []
        lpipss = []

        views, name = scene.getSomeCameras()

        bg_color = [1, 1, 1] if model_params.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        render_path = os.path.join(model_params.model_path, name, f"ours_{iteration}", "renders")
        gts_path = os.path.join(model_params.model_path, name, f"ours_{iteration}", "gt")

        os.makedirs(render_path, exist_ok=True)
        os.makedirs(gts_path, exist_ok=True)

        for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
            rendering = gaussians.render(view, pipeline_params, background)["render"]
            gt = view.original_image

            torchvision.utils.save_image(rendering, os.path.join(render_path, f"{idx:05d}.png"))
            torchvision.utils.save_image(gt, os.path.join(gts_path, f"{idx:05d}.png"))

            rendering = rendering.unsqueeze(0)
            gt = gt.unsqueeze(0)
            ssims.append(ssim(rendering, gt))
            psnrs.append(psnr(rendering, gt))
            lpipss.append(lpips(rendering, gt, net_type="vgg"))
            gc.collect()
            torch.cuda.empty_cache()

        return {
            "SSIM": torch.tensor(ssims).mean().item(),
            "PSNR": torch.tensor(psnrs).mean().item(),
            "LPIPS": torch.tensor(lpipss).mean().item(),
        }


def check_equal_fields(gaussians1: GaussianModel, gaussians2: GaussianModel, do_print=False, skip=[]):
    def list_vars(object, skip=[]):
        return [var for var in vars(object) if (var not in skip) and (not callable(getattr(object, var)))]
    def process(var, a, b, do_print):
        if do_print:
            print(var, a, b)
        if a is None:
            assert (b is None)
            return
        if isinstance(a, list):
            assert(len(a) == len(b))
            for idx, (_a, _b) in enumerate(zip(a, b)):
                process(f"{var}_{idx}", _a, _b, do_print=do_print)
            return
        if isinstance(a, dict):
            for k in a.keys():
                process(f"{var}_{k}", a[k], b[k], do_print=do_print)
            return
        if isinstance(a, torch.Tensor):
            assert (a.numel() == b.numel())
            if a.numel() > 0:
                assert (torch.abs(a - b).sum() < 1e-5)
            return
        if isinstance(a, np.ndarray):
            assert (np.abs(a - b).sum() < 1e-5)
            return
        if isinstance(a, Camera):
            for _var in list_vars(a, skip=skip):
                process(f"{var}_{_var}", a.__dict__[_var], b.__dict__[_var], do_print=do_print)
            return
        assert (a == b)

    for var in list_vars(gaussians1, skip=skip):
        a, b = gaussians1.__dict__[var], gaussians2.__dict__[var]
        process(var, a, b, do_print=do_print)

def run_vq(
    model_params: ModelParams,
    optim_params: OptimizationParams,
    pipeline_params: PipelineParams,
    comp_params: CompressionParams,
):
    gaussians = GaussianModel(model_params.sh_degree, \
                              quantization=not optim_params.not_quantization_aware, \
                              use_factor_scaling=True)

    scene = Scene(model_params, gaussians, load_iteration=comp_params.load_iteration, shuffle=False, save_memory=True)

    if comp_params.start_checkpoint:
        (checkpoint_params, first_iter) = torch.load(comp_params.start_checkpoint)
        gaussians.restore(checkpoint_params, optim_params)

    timings = {}
    # %%
    start_time = time.time()
    color_importance, gaussian_sensitivity = calc_importance_experimental(gaussians, scene, pipeline_params, use_gt=True)
    # color_importance, gaussian_sensitivity = calc_importance(gaussians, scene, pipeline_params)
    end_time = time.time()
    timings["sensitivity_calculation"] = end_time-start_time
    # %%
    print("vq compression..")
    with torch.no_grad():
        start_time = time.time()
        color_importance_n = color_importance.amax(-1)
        gaussian_importance_n = gaussian_sensitivity.amax(-1)

        torch.cuda.empty_cache()

        color_compression_settings = CompressionSettings(
            codebook_size=comp_params.color_codebook_size,
            importance_prune=comp_params.color_importance_prune,
            importance_include=None, #comp_params.color_importance_include,
            importance_include_relative=0.9,
            steps=int(comp_params.color_cluster_iterations),
            decay=comp_params.color_decay,
            batch_size=comp_params.color_batch_size,
        )

        gaussian_compression_settings = CompressionSettings(
            codebook_size=comp_params.gaussian_codebook_size,
            importance_prune=None,
            importance_include=None, #comp_params.gaussian_importance_include,#None
            importance_include_relative=0.75,
            steps=int(comp_params.gaussian_cluster_iterations),
            decay=comp_params.gaussian_decay,
            batch_size=comp_params.gaussian_batch_size,
        )

        compress_gaussians(gaussians, color_importance_n, gaussian_importance_n,
            color_compression_settings if not comp_params.not_compress_color else None,
            gaussian_compression_settings if not comp_params.not_compress_gaussians else None,
            comp_params.color_compress_non_dir, prune_threshold=comp_params.prune_threshold,
        )
        end_time = time.time()
        timings["clustering"] = end_time-start_time

    gc.collect()
    torch.cuda.empty_cache()
    os.makedirs(comp_params.output_vq, exist_ok=True)
    copyfile(
        path.join(model_params.model_path, "cfg_args"),
        path.join(comp_params.output_vq, "cfg_args"),
    )
    model_params.model_path = comp_params.output_vq

    with open(os.path.join(comp_params.output_vq, "cfg_args_comp"), "w") as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(comp_params))))

    iteration = scene.loaded_iter + comp_params.finetune_iterations
    if comp_params.finetune_iterations > 0:
        start_time = time.time()
        finetune(scene, model_params, optim_params, comp_params, pipeline_params, debug_from=-1)
        end_time = time.time()
        timings["finetune"]=end_time-start_time

        # %%
    out_file = path.join(
        comp_params.output_vq,
        f"point_cloud/iteration_{iteration}/point_cloud.npz",
    )
    start_time = time.time()
    gaussians.save_npz(out_file, sort_morton=not comp_params.not_sort_morton)
    end_time = time.time()
    timings["encode"] = end_time-start_time
    timings["total"] = sum(timings.values())
    with open(f"{comp_params.output_vq}/times.json","w") as f:
        json.dump(timings,f)
    file_size = os.path.getsize(out_file) / 1024**2
    print(f"saved vq finetuned model to {out_file}")

    # eval model
    print("evaluating...")
    metrics = render_and_eval(gaussians, scene, model_params, pipeline_params, iteration)
    metrics["size"] = file_size
    # print(metrics)
    with open(f"{comp_params.output_vq}/results.json", "w") as f:
        json.dump({f"ours_{iteration}": metrics}, f, indent=4)

if __name__ == "__main__":
    parser = ArgumentParser(description="Compression script parameters")
    model = ModelParams(parser, sentinel=True)
    model.data_device = "cuda"
    pipeline = PipelineParams(parser)
    op = OptimizationParams(parser)
    comp = CompressionParams(parser)
    args = get_combined_args(parser)

    if args.output_vq is None:
        args.output_vq = unique_output_folder()

    model_params = model.extract(args)
    optim_params = op.extract(args)
    pipeline_params = pipeline.extract(args)
    comp_params = comp.extract(args)
    # print(args.model_path, args.source_path, args.images)
    run_vq(model_params, optim_params, pipeline_params, comp_params)
