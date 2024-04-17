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

import os
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
# from scene.gaussian_model_src import GaussianModel as GaussianModelSrc
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
import shutil
from glob import glob

class Scene:

    gaussians : GaussianModel #| GaussianModelSrc

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, \
                 resolution_scales=[1.0], override_quantization=False, save_memory=False):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            sub_path = os.path.join(self.model_path, "point_cloud")
            if load_iteration == -1 and os.path.exists(sub_path):
                self.loaded_iter = searchForMaxIteration(sub_path)
            else:
                self.loaded_iter = load_iteration
            print(f"Loading trained model at iteration {self.loaded_iter}")

        self.train_cameras = {}
        self.test_cameras = {}

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "transforms_dust3r.json")):
            scene_info = sceneLoadTypeCallbacks["Dust3r"](args.source_path, args.white_background, args.eval)
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            # with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
            #     dest_file.write(src_file.read())
            shutil.copyfile(scene_info.ply_path, os.path.join(self.model_path, "input.ply"))

            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, save_memory=save_memory,args=args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, save_memory=save_memory, args=args)
        if self.loaded_iter and os.path.exists(sub_path):
            self.gaussians.load(
                glob(
                    os.path.join(
                        self.model_path,
                        "point_cloud",
                        "iteration_" + str(self.loaded_iter),
                        "point_cloud.*",
                    )
                )[0],
                override_quantization=override_quantization
            )
        else:
            self.gaussians.load_ply(scene_info.ply_path)#scene_info.point_cloud) # load from sparse ply
            self.gaussians.spatial_lr_scale = scene_info.nerf_normalization["radius"]

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, f"point_cloud/iteration_{iteration}")
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]

    def getSomeCameras(self, scale=1.0):
        ret = self.getTestCameras(scale)
        if len(ret) > 0:
            return ret, "test"
        return self.getTrainCameras(scale), "train"

    def __len__(self, scale=1.0):
        return len(self.train_cameras[scale]) + len(self.test_cameras[scale])