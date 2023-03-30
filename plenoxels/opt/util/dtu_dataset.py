from .util import Rays, Intrin, select_or_shuffle_rays
from .dataset_base import DatasetBase
import torch
from typing import NamedTuple, Optional, Union
import os
from os import path
import imageio
from PIL import Image
from tqdm import tqdm
import cv2
import json
import numpy as np
from glob import glob
import matplotlib.pyplot as plt


class DTUMVSDataset(DatasetBase):
    """
    DTU dataset loader
    """

    focal: float
    c2w: torch.Tensor  # (n_images, 4, 4)
    gt: torch.Tensor  # (n_images, h, w, 3)
    h: int
    w: int
    n_images: int
    rays: Optional[Rays]
    split: str

    def __init__(
            self,
            root,
            split,
            light_idx=0,
            epoch_size: Optional[int] = None,
            device: Union[str, torch.device] = "cpu",
            scene_scale: Optional[float] = None,
            factor: int = 1,
            scale: Optional[float] = None,
            permutation: bool = True,
            white_bkgd: bool = False,
            n_images=None,
            hold_every=8,
            **kwargs
    ):
        super().__init__()
        # assert path.isdir(root), f"'{root}' is not a directory"

        self.scene_radius = 1.1

        if scene_scale is None:
            scene_scale = 1.0
        if scale is None:
            scale = 1.0
        self.device = device
        self.permutation = permutation
        self.epoch_size = epoch_size

        # split_name = split if split != "test_train" else "train"
        scan_idx = int(root.split('/')[-1].split('_')[0][4:])
        self.data_path = os.path.dirname(os.path.dirname(root))
        print("LOAD DATA", self.data_path)
        print(f"scan idx: {scan_idx}\tlight idx: {light_idx}")

        img_filename = [os.path.join(
            self.data_path, 'Rectified/scan{}_train/rect_{:0>3}_{}_r5000.png'.format(scan_idx, i + 1, light_idx))
            for i in range(49)]
        mask_filename = [os.path.join(
            self.data_path, 'Depths/scan{}_train/depth_visual_{:0>4}.png'.format(scan_idx, i))
            for i in range(49)]
        proj_mat_filename = [os.path.join(self.data_path, 'Cameras/train/{:0>8}_cam.txt').format(i) for i in range(49)]
        i_test = np.arange(len(img_filename))[::hold_every][1:-1]
        i_train = np.array([j for j in range(len(img_filename)) if j not in i_test])
        idx = i_train if split == "train" else i_test
        print(f"Number of images: {len(idx)}")
        all_w2c = []
        all_gt = []
        all_gt_white = []
        for i in idx:
            intrinsics, w2c = read_cam_file(proj_mat_filename[i])
            all_w2c.append(w2c)
            mask = read_img(mask_filename[i])
            mask = cv2.resize(mask, (int(mask.shape[1] * 4), int(mask.shape[0] * 4)), interpolation=cv2.INTER_LINEAR)
            im_gt = read_img(img_filename[i])
            im_gt *= mask[..., None]
            im_white = im_gt + (1. - mask[..., None])
            im_white[im_white > 1.] = 1.
            all_gt.append(im_gt)
            all_gt_white.append(im_white)
        all_w2c = np.stack(all_w2c, axis=0)
        all_gt = torch.from_numpy(np.stack(all_gt, axis=0)).float()
        all_gt_white = torch.from_numpy(np.stack(all_gt_white, axis=0)).float()
        self.c2w = torch.from_numpy(np.linalg.inv(all_w2c)).float()

        if kwargs["high_fq"]:
            gt = [all_gt[i] if i % 2 == 0 else all_gt_white[i] for i in range(len(all_gt))]
            self.gt = torch.stack(gt)
            print("\033[1;31mUse high frequency background.\033[0m")
        elif white_bkgd:
            self.gt = all_gt_white
            print("Use white background.")
        else:
            self.gt = all_gt
            print("Use black background.")

        self.n_images, self.h_full, self.w_full, _ = self.gt.shape
        self.intrins_full: Intrin = Intrin(float(intrinsics[0, 0]), float(intrinsics[1, 1]),
                                           float(intrinsics[0, 2]), float(intrinsics[1, 2]))
        self.split = split
        self.scene_scale = scene_scale
        if self.split == "train":
            self.gen_rays(factor=factor)
        else:
            # Rays are not needed for testing
            self.h, self.w = self.h_full, self.w_full
            self.intrins: Intrin = self.intrins_full

        self.should_use_background = False  # Give warning


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
