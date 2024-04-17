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

from scene.cameras import Camera
import numpy as np

WARNED = False

def loadCam(args, id, cam_info, resolution_scale, save_memory=False):
    orig_w, orig_h = cam_info.width, cam_info.height#cam_info.image.size

    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))
        # cam_info.intrinsic[:2, :] /= scale

    return Camera(colmap_id=cam_info.uid, extrinsic=cam_info.extrinsic, intrinsic=cam_info.intrinsic,
                  h=resolution[1], w=resolution[0],
                  image_name=cam_info.image_name, image_path=cam_info.image_path, uid=id, data_device=args.data_device,
                  save_memory=save_memory)

def cameraList_from_camInfos(cam_infos, resolution_scale, save_memory, args):
    return [loadCam(args, id, c, resolution_scale, save_memory=save_memory) for id, c in enumerate(cam_infos)]

def camera_to_JSON(id, camera : Camera):
    Rt = camera.extrinsic #np.eye(4)
    # Rt[:3, :3] = camera.R.transpose()
    # Rt[:3, 3] = camera.T

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]

    serializable_intrinsics = [x.tolist() for x in camera.intrinsic]

    camera_entry = {
        'id': id,
        'img_name': camera.image_name,
        'width': camera.width,
        'height': camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'intrinsic': serializable_intrinsics
    }
    return camera_entry
