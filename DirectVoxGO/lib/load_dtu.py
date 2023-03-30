import os
import torch
import numpy as np
import imageio
import cv2

from glob import glob


def load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv2.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K / K[2, 2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return intrinsics, pose


def load_dtu_data(basedir):
    all_c2w = []
    intrinsics_all = []
    images = []

    render_cameras_name = 'cameras_sphere.npz'
    camera_dict = np.load(os.path.join(basedir, render_cameras_name))
    camera_dict = camera_dict
    images_lis = sorted(glob(os.path.join(basedir, 'image/*.png')))
    masks_lis = sorted(glob(os.path.join(basedir, 'mask/*.png')))
    n_imgs = len(images_lis)

    for i in range(n_imgs):
        image = imageio.imread(images_lis[i]).astype(np.float32) / 255.
        image = image[..., :3]
        mask = imageio.imread(masks_lis[i]).astype(np.float32) / 255.
        mask = mask[..., :3]
        image *= mask

        # white bg
        # image = image + (1 - mask[..., :3])
        # image[image > 1.] = 1.

        world_mat = camera_dict['world_mat_%d' % i].astype(np.float32)
        scale_mat = camera_dict['scale_mat_%d' % i].astype(np.float32)
        P = world_mat @ scale_mat
        P = P[:3, :4]
        intrinsics, pose = load_K_Rt_from_P(None, P)

        # down sample
        [halfres_h, halfres_w] = [hw // 2 for hw in image.shape[:2]]
        image = cv2.resize(
            image, (halfres_w, halfres_h), interpolation=cv2.INTER_AREA)
        intrinsics[:2] /= 2

        images.append(image)
        intrinsics_all.append(intrinsics)
        all_c2w.append(pose)

    imgs = np.stack(images, 0).astype(np.float32)
    poses = np.stack(all_c2w).astype(np.float32)
    H, W = imgs[0].shape[:2]
    K = intrinsics_all[0]
    focal = float(K[0, 0])

    # need to be set
    render_poses = poses

    i_test = np.arange(n_imgs)[::8][1:-1]
    i_train = np.array([j for j in range(n_imgs) if j not in i_test])
    i_split = [i_train, i_test, i_test]

    return imgs, poses, render_poses, [H, W, focal], K, i_split
