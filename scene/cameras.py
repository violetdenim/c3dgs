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
import cv2
import torch
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
def mat_to_quat(m, normed=True):
    w = torch.sqrt(1.0 + m[0, 0] + m[1, 1] + m[2,2]) / 2.0
    w4 = 4.0 * w
    x = (m[2, 1] - m[1, 2]) / w4
    y = (m[0, 2] - m[2, 0]) / w4
    z = (m[1, 0] - m[0, 1]) / w4
    if normed:
        norm2 = (x*x + y*y + z*z + w*w)**0.5
        x, y, z, w = x/norm2, y/norm2, z/norm2, w/norm2

    return x, y, z, w, m[0, 3], m[1, 3], m[2, 3]

class Camera(nn.Module):
    def __init__(self, colmap_id, extrinsic, intrinsic, h, w, #R, T, FoVx, FoVy
                 image_name, image_path, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0,
                 data_device="cuda", save_memory=False):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        #self.extrinsic = torch.FloatTensor(extrinsic).transpose(0, 1).cuda()
        self.extrinsic_vector = torch.FloatTensor(mat_to_quat(torch.FloatTensor(extrinsic)))
        self.intrinsic = torch.FloatTensor(intrinsic).cuda()
        self.intrinsic[0, 2] = w
        self.intrinsic[1, 2] = h
        self.image_name = image_name
        self.image_path = image_path

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")
        # self.image_width = w
        # self.image_height = h

        # self.trans = trans
        # self.scale = scale

        # self.world_view_transform = torch.tensor(getWorld2View2(None, None, self.extrinsic, trans, scale)).transpose(0, 1).to(data_device)
        # self.projection_matrix = getProjectionMatrix(znear=0.01, zfar=100.0, \
        #                                              fovX=self.intrinsic[0, 0], fovY=self.intrinsic[1, 1], \
        #                                              z_sign=1.0).transpose(0, 1).to(data_device)
        # self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        # self.camera_center = self.world_view_transform.inverse()[3, :3]

        self.save_memory = save_memory
        self._image = None

    @property
    def original_image(self):
        if self._image is None:
            image = cv2.imread(self.image_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.0
            if image.shape[2] == 4:
                alpha = image[:, :, 3]
                for i in range(3):
                    image[:, :, i] *= alpha
            image = image[:, :, :3]
            # ATTENTION: flip up and down (for DUST3R ONLY!)
            image = image[::-1, ::-1, :]

            image = cv2.resize(image[:, :, ::-1], (int(self.intrinsic[0, 2]), int(self.intrinsic[1, 2])))
            #(self.image_width, self.image_height))
            image = torch.from_numpy(image).to(self.data_device).permute(2, 0, 1)
            image = image.clamp(0.0, 1.0)

            if self.save_memory:
                return image
            self._image = image
            return self._image
        if not self.save_memory:
            return self._image
        else:
            image, self._image = self._image, None
            return image

class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]

