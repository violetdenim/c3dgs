import argparse
import math
import gradio
import os
import torch
import numpy as np
import cv2
import open3d as o3d
import tempfile
import functools
import trimesh
import copy
from scipy.spatial.transform import Rotation

from dust3r.inference import inference, load_model
from dust3r.image_pairs import make_pairs
from dust3r.utils.image import load_images, rgb
from dust3r.utils.device import to_numpy
from dust3r.viz import add_scene_cam, CAM_COLORS, OPENGL, pts3d_to_trimesh, cat_meshes
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode

import json
import matplotlib.pyplot as pl
pl.ion()

torch.backends.cuda.matmul.allow_tf32 = True  # for gpu >= Ampere and pytorch >= 1.12
batch_size = 1


def _convert_scene_output_to_glb(outdir, imgs, pts3d, mask, focals, cams2world, cam_size=0.05,
                                 cam_color=None, as_pointcloud=False, transparent_cams=False, silent=False):
    assert len(pts3d) == len(mask) <= len(imgs) <= len(cams2world) == len(focals)
    pts3d = to_numpy(pts3d)
    imgs = to_numpy(imgs)
    focals = to_numpy(focals)
    cams2world = to_numpy(cams2world)

    scene = trimesh.Scene()

    # full pointcloud
    if as_pointcloud:
        pts = np.concatenate([p[m] for p, m in zip(pts3d, mask)])
        col = np.concatenate([p[m] for p, m in zip(imgs, mask)])
        pct = trimesh.PointCloud(pts.reshape(-1, 3), colors=col.reshape(-1, 3))
        scene.add_geometry(pct)
    else:
        meshes = []
        for i in range(len(imgs)):
            meshes.append(pts3d_to_trimesh(imgs[i], pts3d[i], mask[i]))
        mesh = trimesh.Trimesh(**cat_meshes(meshes))
        scene.add_geometry(mesh)

    # add each camera
    for i, pose_c2w in enumerate(cams2world):
        if isinstance(cam_color, list):
            camera_edge_color = cam_color[i]
        else:
            camera_edge_color = cam_color or CAM_COLORS[i % len(CAM_COLORS)]
        add_scene_cam(scene, pose_c2w, camera_edge_color,
                      None if transparent_cams else imgs[i], focals[i],
                      imsize=imgs[i].shape[1::-1], screen_width=cam_size)

    rot = np.eye(4)
    rot[:3, :3] = Rotation.from_euler('y', np.deg2rad(180)).as_matrix()
    scene.apply_transform(np.linalg.inv(cams2world[0] @ OPENGL @ rot))
    outfile = os.path.join(outdir, 'scene.glb')
    # outfile = os.path.join(outdir, 'scene.obj')
    if not silent:
        print('(exporting 3D scene to', outfile, ')')
    scene.export(file_obj=outfile)

    return outfile

def get_3D_model_from_scene(outdir, silent, scene, min_conf_thr=3, as_pointcloud=False, mask_sky=False,
                            clean_depth=False, transparent_cams=False, cam_size=0.05):
    """
    extract 3D_model (glb file) from a reconstructed scene
    """
    if scene is None:
        return None
    # post processes
    if clean_depth:
        scene = scene.clean_pointcloud()
    if mask_sky:
        scene = scene.mask_sky()

    # get optimized values from scene
    rgbimg = scene.imgs
    # focals = scene.get_focals().cpu()
    # cams2world = scene.get_im_poses().cpu()
    # 3D pointcloud from depthmap, poses and intrinsics
    pts3d = to_numpy(scene.get_pts3d())
    scene.min_conf_thr = float(scene.conf_trf(torch.tensor(min_conf_thr)))
    msk = to_numpy(scene.get_masks())

    pts = np.concatenate([p[m] for p, m in zip(pts3d, msk)])
    col = np.concatenate([p[m] for p, m in zip(rgbimg, msk)])

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.reshape(-1, 3))
    pcd.colors = o3d.utility.Vector3dVector(col.reshape(-1, 3))
    pcd.estimate_normals()
    outfile = os.path.join(outdir, 'scene.ply')

    # rot = np.eye(4)
    # rot[:3, :3] = Rotation.from_euler('y', np.deg2rad(180)).as_matrix()
    # transformation = np.linalg.inv(scene.get_im_poses()[0].detach().cpu().numpy() @ OPENGL @ rot)
    # print(transformation)
    # pcd.transform(transformation)
    o3d.io.write_point_cloud(outfile, pcd)
    return outfile
    # return _convert_scene_output_to_glb(outdir, rgbimg, pts3d, msk, focals, cams2world, as_pointcloud=as_pointcloud,
    #                                     transparent_cams=transparent_cams, cam_size=cam_size, silent=silent)

def get_reconstructed_scene(outdir, model, device, silent, image_size, filelist, schedule, niter, min_conf_thr,
                            as_pointcloud, mask_sky, clean_depth, transparent_cams, cam_size,
                            scenegraph_type, winsize, refid):
    """
    from a list of images, run dust3r inference, global aligner.
    then run get_3D_model_from_scene
    """
    imgs = load_images(filelist, size=image_size, verbose=not silent)

    if len(imgs) == 1:
        imgs = [imgs[0], copy.deepcopy(imgs[0])]
        imgs[1]['idx'] = 1
    if scenegraph_type == "swin":
        scenegraph_type = scenegraph_type + "-" + str(winsize)
    elif scenegraph_type == "oneref":
        scenegraph_type = scenegraph_type + "-" + str(refid)

    pairs = make_pairs(imgs, scene_graph=scenegraph_type, prefilter=None, symmetrize=True)
    output = inference(pairs, model, device, batch_size=batch_size, verbose=not silent)

    mode = GlobalAlignerMode.PointCloudOptimizer if len(imgs) > 2 else GlobalAlignerMode.PairViewer
    scene = global_aligner(output, device=device, mode=mode, verbose=not silent)
    lr = 0.01

    if mode == GlobalAlignerMode.PointCloudOptimizer:
        loss = scene.compute_global_alignment(init='mst', niter=niter, schedule=schedule, lr=lr)

    outfile = get_3D_model_from_scene(outdir, silent, scene, min_conf_thr, as_pointcloud, mask_sky,
                                      clean_depth, transparent_cams, cam_size)

    # also return rgb, depth and confidence imgs
    # depth is normalized with the max value for all images
    # we apply the jet colormap on the confidence maps

    # rgbimg = scene.imgs
    # depths = to_numpy(scene.get_depthmaps())
    # confs = to_numpy([c for c in scene.im_conf])
    # cmap = pl.get_cmap('jet')
    # depths_max = max([d.max() for d in depths])
    # depths = [d/depths_max for d in depths]
    # confs_max = max([d.max() for d in confs])
    # confs = [cmap(d/confs_max) for d in confs]
    #
    # imgs = []
    # for i in range(len(rgbimg)):
    #     imgs.append(rgbimg[i])
    #     imgs.append(rgb(depths[i]))
    #     imgs.append(rgb(confs[i]))

    return scene#, outfile #, imgs


def export_to_json(intrinsics, poses, filelist, outfile):
    dump_dict = {"frames": []}
    for i, (intr, pose, file) in enumerate(zip(intrinsics, poses, filelist)):
        w, h = cv2.imread(file).shape[:2]
        scale = 512.0 / max(w, h)
        _intr = intr.clone()
        _intr[:2, :] /= scale

        dump_dict["frames"].append({
            "file_path": file,
            "transform_matrix": pose.tolist(),
            "intrinsic_matrix": _intr.tolist(),
            "colmap_im_id": i
        })

    with open(outfile, 'w') as fp:
        json.dump(dump_dict, fp)

if __name__ == "__main__":
    input_dir = '/home/zipa/data/urbanscene3d/PolyTech'
    weights = 'checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth'
    outdir = 'tmp'
    os.makedirs(outdir, 0o777, True)

    filelist = [os.path.join(input_dir, f) for f in sorted(os.listdir(input_dir))[:3]]
    model = load_model(weights, "cuda", verbose=False)
    scene = get_reconstructed_scene(outdir, model, "cuda", False, 512, filelist, "linear", 300, 3,
                            True, False, True, False, 0.05,
                            "complete", 1, 0)
    # export pointcloud as initial for gaussian splatting procedure
    export_to_json(scene.get_intrinsics(), scene.get_im_poses(), filelist, os.path.join(outdir, "transforms_dust3r.json"))
