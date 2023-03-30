from .util import Rays, Intrin, select_or_shuffle_rays
from .dataset_base import DatasetBase
import torch
from typing import NamedTuple, Optional, Union
import os
from os import path
import imageio
from tqdm import tqdm
import cv2
import json
import numpy as np
from glob import glob


# This function is borrowed from IDR: https://github.com/lioryariv/idr
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


class DTUDataset(DatasetBase):
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
        assert path.isdir(root), f"'{root}' is not a directory"

        if scene_scale is None:
            scene_scale = 1.0
        if scale is None:
            scale = 1.0
        self.device = device
        self.permutation = permutation
        self.epoch_size = epoch_size
        all_c2w = []
        intrinsics_all = []
        images = []
        images_white = []

        # split_name = split if split != "test_train" else "train"

        self.data_path = root
        print("LOAD DATA", self.data_path)
        self.render_cameras_name = 'cameras_sphere.npz'
        camera_dict = np.load(os.path.join(self.data_path, self.render_cameras_name))
        self.camera_dict = camera_dict
        images_lis = sorted(glob(os.path.join(self.data_path, 'image/*.png')))
        masks_lis = sorted(glob(os.path.join(self.data_path, 'mask/*.png')))
        n_imgs = len(images_lis)

        i_test = np.arange(n_imgs)[::hold_every][1:-1]
        i_train = np.array([j for j in range(n_imgs) if j not in i_test])
        id_list = i_train if split == "train" else i_test
        if split == "all":
            id_list = np.arange(n_imgs)
        print(f"Number of images for {split}: {len(id_list)}")

        for i in id_list:
            image = imageio.imread(images_lis[i]).astype(np.float32) / 255.
            image = image[..., :3]
            mask = imageio.imread(masks_lis[i]).astype(np.float32) / 255.
            mask = mask[..., :3]
            image *= mask

            image_white = image + (1 - mask[..., :3])
            image_white[image_white > 1.] = 1.

            world_mat = camera_dict['world_mat_%d' % i].astype(np.float32)
            scale_mat = camera_dict['scale_mat_%d' % i].astype(np.float32)
            P = world_mat @ scale_mat
            P = P[:3, :4]
            intrinsics, pose = load_K_Rt_from_P(None, P)

            # down sample
            [halfres_h, halfres_w] = [hw // 2 for hw in image.shape[:2]]
            image = cv2.resize(
                image, (halfres_w, halfres_h), interpolation=cv2.INTER_AREA)
            image_white = cv2.resize(
                image_white, (halfres_w, halfres_h), interpolation=cv2.INTER_AREA)
            intrinsics[:2] /= 2

            images.append(torch.from_numpy(image).float())
            images_white.append(torch.from_numpy(image_white).float())
            intrinsics_all.append(intrinsics)
            all_c2w.append(torch.from_numpy(pose).float())

        if kwargs["high_fq"]:
            gt = [images[i] if i % 2 == 0 else images_white[i] for i in range(len(images))]
            self.gt = torch.stack(gt)
            print("Use high frequency background.")
        elif white_bkgd:
            self.gt = torch.stack(images_white)
            print("Use white background.")
        else:
            self.gt = torch.stack(images)  # .cpu()  # [n_images, H, W, 3]
            print("Use black background.")

        self.intrinsics_all = np.stack(intrinsics_all)  # .to(self.device)   # [n_images, 4, 4]
        self.c2w = torch.stack(all_c2w)  # .to(self.device)  # [n_images, 4, 4]
        # self.intrinsics_all_inv = torch.inverse(self.intrinsics_all)  # [n_images, 4, 4]

        self.fx = float(self.intrinsics_all[0][0, 0])
        self.fy = float(self.intrinsics_all[0][1, 1])
        self.cx = float(self.intrinsics_all[0][0, 2])
        self.cy = float(self.intrinsics_all[0][1, 2])

        ## TODO check
        # self.c2w[:, :3, 3] *= scene_scale

        self.n_images, self.h_full, self.w_full, _ = self.gt.shape
        # Choose a subset of training images
        if n_images is not None:
            if n_images > self.n_images:
                print(f'using {self.n_images} available training views instead of the requested {n_images}.')
                n_images = self.n_images
            self.n_images = n_images
            self.gt = self.gt[0:n_images, ...]
            self.c2w = self.c2w[0:n_images, ...]

        self.intrins_full: Intrin = Intrin(self.fx, self.fy,
                                           self.cx,
                                           self.cy)

        self.split = split
        self.scene_scale = scene_scale
        if self.split == "train":
            self.gen_rays(factor=factor)
        else:
            # Rays are not needed for testing
            self.h, self.w = self.h_full, self.w_full
            self.intrins: Intrin = self.intrins_full

        self.should_use_background = False  # Give warning
