from scene.gaussian_model import GaussianModel
from scene import Scene
from arguments import ModelParams, PipelineParams, CompressionParams, OptimizationParams
import torch
from matplotlib import pyplot as plt
import os
from finetune import finetune

if __name__ == '__main__':
    gaussians = GaussianModel(3, quantization=True, use_factor_scaling=True)

    # cfgfilepath = os.path.join('pine_orig', "cfg_args")
    # with open(cfgfilepath) as cfg_file:
    #     cfgfile_string = cfg_file.read()
    # print(cfgfile_string)

    params = ModelParams()
    params.sh_degree = 3
    params._source_path = '/home/zipa/data/NerfScenes/pine'
    params._model_path = "pine_compressed" #"pine_orig" #
    params._images = "images"
    params._resolution = -1
    params._white_background = False
    params.data_device = "cuda"
    params.eval = False
    params = params.extract()

    pipe = PipelineParams()
    comp = CompressionParams()

    scene = Scene(params, gaussians, load_iteration=-1, shuffle=False, save_memory=True)
    cameras, _ = scene.getSomeCameras()
    camera = cameras[10]
    print("Is indexed", gaussians.is_gaussian_indexed)
    # gaussians.training_setup(OptimizationParams())
    gaussians.to_compressed(scene, pipe, comp)
    finetune(scene, params, OptimizationParams(), comp, pipe, debug_from=-1)
    # gaussians.to_indexed()
    # gaussians.to_unindexed()

    with torch.no_grad():
        cov3Dp = gaussians.get_covariance(1.0)
        render_pkg = gaussians.render(camera, pipe, \
                            torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device="cuda"), cov3d=cov3Dp)
        img = render_pkg["render"].detach().cpu().numpy().transpose(1, 2, 0)
        plt.imshow(img)
        plt.show()