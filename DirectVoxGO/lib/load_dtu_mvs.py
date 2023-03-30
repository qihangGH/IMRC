import os
import torch
import numpy as np
import imageio
import cv2
from PIL import Image


def load_dtu_mvs_data(basedir, light_idx=0):
    # basedir /Rectified/scan1_train
    scan_idx = int(basedir.split('/')[-1].split('_')[0][4:])
    basedir = os.path.dirname(os.path.dirname(basedir))
    img_filename = [os.path.join(
        basedir, 'Rectified/scan{}_train/rect_{:0>3}_{}_r5000.png'.format(scan_idx, i + 1, light_idx))
        for i in range(49)]
    mask_filename = [os.path.join(
        basedir, 'Depths/scan{}_train/depth_visual_{:0>4}.png'.format(scan_idx, i))
        for i in range(49)]
    proj_mat_filename = [os.path.join(basedir, 'Cameras/train/{:0>8}_cam.txt').format(i) for i in range(49)]

    all_w2c = []
    all_gt = []
    for i in range(49):
        intrinsics, w2c = read_cam_file(proj_mat_filename[i])
        all_w2c.append(w2c)
        mask = read_img(mask_filename[i])
        mask = cv2.resize(mask, (int(mask.shape[1] * 4), int(mask.shape[0] * 4)), interpolation=cv2.INTER_LINEAR)
        im_gt = read_img(img_filename[i])
        im_gt *= mask[..., None]
        all_gt.append(im_gt)

    imgs = np.stack(all_gt, 0).astype(np.float32)
    poses = np.linalg.inv(np.stack(all_w2c)).astype(np.float32)
    H, W = imgs[0].shape[:2]
    K = intrinsics
    # TODO: check if focal is used
    focal = float(K[0, 0])

    # scan = basedir.split('/')[-1][8:]
    # TODO: render poses
    render_poses = np.load(r'/lkq/lkq/NeuS/public_data/dtu_scan40/render_c2w_40.npy')

    i_test = np.arange(49)[::8][1:-1]
    i_train = np.array([j for j in range(49) if j not in i_test])
    # TODO: check if i_val is used for model selection
    i_split = [i_train, i_test, i_test]

    return imgs, poses, render_poses, [H, W, focal], K, i_split


def read_img(filename):
    img = Image.open(filename)
    # scale 0~255 to 0~1
    np_img = np.array(img, dtype=np.float32) / 255.
    return np_img


def read_cam_file(filename):
    with open(filename) as f:
        lines = f.readlines()
        lines = [line.rstrip() for line in lines]
    # extrinsics: line [1,5), 4x4 matrix
    extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ').reshape((4, 4))
    # from millimeter to meter
    extrinsics[:3, 3] /= 1000.
    # intrinsics: line [7-10), 3x3 matrix
    intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ').reshape((3, 3))
    intrinsics[:2] *= 4
    # depth_min & depth_interval: line 11
    # depth_min = float(lines[11].split()[0])
    # depth_interval = float(lines[11].split()[1]) * self.interval_scale
    return intrinsics, extrinsics