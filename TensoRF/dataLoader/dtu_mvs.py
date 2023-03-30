import numpy as np
import torch, cv2
from torch.utils.data import Dataset
import json
import imageio
from tqdm import tqdm
from glob import glob
import os
from PIL import Image
from torchvision import transforms as T
import matplotlib.pyplot as plt

from .ray_utils import *


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


class DTUMVSDataset(Dataset):
    def __init__(self, datadir, split='train', downsample=1.0, is_stack=False, N_vis=-1, hold_every=8):

        self.N_vis = N_vis
        self.root_dir = datadir
        self.split = split
        self.is_stack = is_stack
        self.hold_every = hold_every
        self.img_wh = (640, 512)

        self.white_bg = False

        # self.scene_bbox = torch.tensor([[-1.5, -1.5, -1.5], [1.5, 1.5, 1.5]])
        self.scene_bbox = torch.tensor([[-1., -1., -1.], [1., 1., 1.]])
        self.read_meta()
        self.define_proj_mat()

        self.near_far = [0., 6.0]

        self.center = torch.mean(self.scene_bbox, axis=0).float().view(1, 1, 3)
        self.radius = (self.scene_bbox[1] - self.center).float().view(1, 1, 3)
        self.downsample = downsample

    def read_meta(self):
        w, h = self.img_wh
        light_idx = 0
        scan_idx = int(self.root_dir.split('/')[-1].split('_')[0][4:])
        basedir = os.path.dirname(os.path.dirname(self.root_dir))
        img_filename = [os.path.join(
            basedir, 'Rectified/scan{}_train/rect_{:0>3}_{}_r5000.png'.format(scan_idx, i + 1, light_idx))
            for i in range(49)]
        mask_filename = [os.path.join(
            basedir, 'Depths/scan{}_train/depth_visual_{:0>4}.png'.format(scan_idx, i))
            for i in range(49)]
        proj_mat_filename = [os.path.join(basedir, 'Cameras/train/{:0>8}_cam.txt').format(i) for i in range(49)]

        i_test = np.arange(49)[::self.hold_every][1:-1]
        i_train = np.array([j for j in range(49) if j not in i_test])
        id_list = i_train if self.split == "train" else i_test

        self.poses = []
        self.all_rays = []
        self.all_rgbs = []
        self.all_masks = []
        self.downsample = 1.0
        self.images = []
        intrinsics_all = []
        for i in id_list:
            intrinsics, w2c = read_cam_file(proj_mat_filename[i])
            pose = np.linalg.inv(w2c)
            pose = torch.from_numpy(pose).float()
            mask = read_img(mask_filename[i])
            mask = cv2.resize(mask, (int(mask.shape[1] * 4), int(mask.shape[0] * 4)), interpolation=cv2.INTER_LINEAR)
            im_gt = read_img(img_filename[i])
            im_gt *= mask[..., None]
            self.directions = get_ray_directions(
                h, w, [intrinsics[0, 0], intrinsics[1, 1]], [intrinsics[0, 2], intrinsics[1, 2]])  # (h, w, 3)
            self.directions = self.directions / torch.norm(self.directions, dim=-1, keepdim=True)
            rays_o, rays_d = get_rays(self.directions, pose)  # both (h*w, 3)
            self.all_rays += [torch.cat([rays_o, rays_d], 1)]  # (h*w, 6)
            self.all_rgbs.append(torch.from_numpy(im_gt.reshape([-1, 3])).float())
            self.images.append(torch.from_numpy(im_gt).float())
            intrinsics_all.append(intrinsics)
            self.poses.append(pose)

        self.images = torch.stack(self.images, dim=0)
        self.intrinsics = torch.from_numpy(intrinsics_all[0][:3, :3]).float()
        self.poses = torch.stack(self.poses)
        if not self.is_stack:
            self.all_rays = torch.cat(self.all_rays, 0)  # (len(self.meta['frames])*h*w, 3)
            self.all_rgbs = torch.cat(self.all_rgbs, 0)  # (len(self.meta['frames])*h*w, 3)

        #             self.all_depth = torch.cat(self.all_depth, 0)  # (len(self.meta['frames])*h*w, 3)
        else:
            self.all_rays = torch.stack(self.all_rays, 0)  # (len(self.meta['frames]),h*w, 3)
            self.all_rgbs = torch.stack(self.all_rgbs, 0).reshape(-1, *self.img_wh[::-1],
                                                                  3)  # (len(self.meta['frames]),h,w,3)
            # self.all_masks = torch.stack(self.all_masks, 0).reshape(-1,*self.img_wh[::-1])  # (len(self.meta['frames]),h,w,3)

    def define_proj_mat(self):
        self.proj_mat = self.intrinsics.unsqueeze(0) @ torch.inverse(self.poses)[:, :3]

    def world2ndc(self, points, lindisp=None):
        device = points.device
        return (points - self.center.to(device)) / self.radius.to(device)

    def __len__(self):
        return len(self.all_rgbs)

    def __getitem__(self, idx):
        if self.split == 'train':  # use data in the buffers
            sample = {'rays': self.all_rays[idx],
                      'rgbs': self.all_rgbs[idx]}

        else:  # create data for each image separately
            img = self.all_rgbs[idx]
            rays = self.all_rays[idx]
            # mask = self.all_masks[idx]  # for quantity evaluation
            mask = None  # for quantity evaluation

            sample = {'rays': rays,
                      'rgbs': img,
                      'mask': mask}
        return sample


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