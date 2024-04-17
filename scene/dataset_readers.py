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
import sys
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from utils.graphics_utils import BasicPointCloud


class CameraInfo(NamedTuple):
    uid: int
    extrinsic: np.ndarray
    intrinsic: np.ndarray
    # R: np.array
    # T: np.array
    # FovY: np.array
    # FovX: np.array
    # image: np.array
    image_path: str
    image_name: str
    width: int
    height: int


class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str


def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []
    for cam in cam_info:
        W2C = getWorld2View2(None, None, cam.extrinsic)  # cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1
    translate = -center

    return {"translate": translate, "radius": radius}


def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    for key, extr in cam_extrinsics.items():
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write(f"Reading camera {key}/{len(cam_extrinsics)}")
        sys.stdout.flush()
        intr = cam_intrinsics[extr.camera_id]
        height, width, uid = intr.height, intr.width, intr.id
        # R = np.transpose(qvec2rotmat(extr.qvec))
        # T = np.array(extr.tvec)

        cam_extrinsics = np.eye(4)
        cam_extrinsics[:3, :3] = qvec2rotmat(extr.qvec)
        cam_extrinsics[:3, 3] = np.array(extr.tvec)

        if intr.model == "SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model == "PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        # image = Image.open(image_path)
        intrinsics = np.asarray([[FovX, 0, width / 2], [0, FovY, height / 2], [0, 0, 1]])
        cam_info = CameraInfo(uid=uid, extrinsic=cam_extrinsics, intrinsic=intrinsics,
                              image_path=image_path, image_name=image_name, width=width, height=height)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos


def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']

    keys = [p.name for p in vertices.properties]

    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    if 'red' in keys:
        colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    else:
        colors = np.vstack([vertices['f_dc_0'], vertices['f_dc_1'], vertices['f_dc_2']]).T  # / 255.0
    if 'nx' in keys:
        normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    else:
        normals = np.zeros_like(positions)
    return BasicPointCloud(points=positions, colors=colors, normals=normals)


def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
             ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
             ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]

    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)


def readColmapSceneInfo(path, images, eval, llffhold=8):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images

    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics,
                                           images_folder=os.path.join(path, reading_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key=lambda x: x.image_name)

    # poses = np.load(path + "/poses_bounds.npy")
    # print(cam_infos[0], poses[0])
    # exit(0)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except Exception as e:
        print(f"Error fetching point cloud. {e}")
        pcd = None

    return SceneInfo(point_cloud=pcd,
                     train_cameras=train_cam_infos,
                     test_cameras=test_cam_infos,
                     nerf_normalization=nerf_normalization,
                     ply_path=ply_path)


def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    def read_img(img_path):
        image = Image.open(img_path)
        im_data = np.array(image.convert("RGBA"))
        bg = np.array([1, 1, 1]) if white_background else np.array([0, 0, 0])
        norm_data = im_data / 255.0
        arr = norm_data[:, :, :3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
        image = Image.fromarray(np.array(arr * 255.0, dtype=np.byte), "RGB")
        return image

    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)

        fovx = contents.get("camera_angle_x", None)
        fovy = None
        fl_x = contents.get("fl_x", None)
        fl_y = contents.get("fl_y", None)
        w, h = contents.get('w', None), contents.get('h', None)
        cx, cy = contents.get('cx', None), contents.get('cy', None)

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, os.path.splitext(frame["file_path"])[0] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            # w2c[:3, :3] = np.transpose(w2c[:3, :3])
            # R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            # T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem

            if w is None:
                image = read_img(image_path)
                w, h = image.size[0], image.size[1]

            if fl_x is not None:
                fovx = focal2fov(fl_x, w)
            if fl_y is not None:
                fovy = focal2fov(fl_y, h)

            if fovx is None:
                fovx = w / 2
            if fovy is None:
                fovy = focal2fov(fov2focal(fovx, w), h)
            if cx is None:
                cx = w / 2
            if cy is None:
                cy = h / 2
            intrinsics = np.asarray([[fovx, 0, cx], [0, fovy, cy], [0, 0, 1]])

            cam_infos.append(CameraInfo(uid=idx, extrinsic=w2c, intrinsic=intrinsics,
                                        image_path=image_path, image_name=image_name,
                                        width=w, height=h))
    return cam_infos


def readCamerasFromTransformsDust3r(path, transformsfile, white_background, extension=".png"):
    def read_img(img_path):
        image = Image.open(img_path)
        im_data = np.array(image.convert("RGBA"))
        bg = np.array([1, 1, 1]) if white_background else np.array([0, 0, 0])
        norm_data = im_data / 255.0
        arr = norm_data[:, :, :3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
        image = Image.fromarray(np.array(arr * 255.0, dtype=np.byte), "RGB")
        return image

    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            # TBD: replace with relative path!
            cam_name = frame["file_path"]

            # NeRF 'transform_matrix' is a camera-to-world transform
            extrinsics = np.array(frame["transform_matrix"])
            extrinsics = np.linalg.inv(extrinsics)
            intrinsics = np.array(frame['intrinsic_matrix'])

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = read_img(image_path)
            w, h = image.size[0], image.size[1]

            intrinsics[0][0] = focal2fov(intrinsics[0][0], w)
            intrinsics[1][1] = focal2fov(intrinsics[1][1], h)

            cam_infos.append(CameraInfo(uid=idx, extrinsic=extrinsics, intrinsic=intrinsics,
                                        image_path=image_path, image_name=image_name,
                                        width=w, height=h))#image.size[0], height=image.size[1]))
    return cam_infos

def readDustrInfo(path, white_background, eval):
    train_cam_infos = readCamerasFromTransformsDust3r(path, "transforms_dust3r.json", white_background)
    nerf_normalization = {"translate": [0.0, 0.0, 0.0], "radius": 1.0} #getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "scene.ply")
    try:
        pcd = fetchPly(ply_path)
    except:
        print(f"Failed to fetch {ply_path}")
        pcd = None

    return SceneInfo(point_cloud=pcd,
                     train_cameras=train_cam_infos,
                     test_cameras=[],
                     nerf_normalization=nerf_normalization,
                     ply_path=ply_path)

def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    test_cam_infos = []
    if os.path.exists(os.path.join(path, "transforms_test.json")):
        print("Reading Test Transforms")
        test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)
        if not eval:
            train_cam_infos.extend(test_cam_infos)
            test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")

        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        # pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        print(f"Failed to fetch {ply_path}")
        pcd = None

    return SceneInfo(point_cloud=pcd,
                     train_cameras=train_cam_infos,
                     test_cameras=test_cam_infos,
                     nerf_normalization=nerf_normalization,
                     ply_path=ply_path)


sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender": readNerfSyntheticInfo,
    "Dust3r": readDustrInfo
}