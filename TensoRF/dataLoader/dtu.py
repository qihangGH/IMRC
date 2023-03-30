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


class DTUDataset(Dataset):
    def __init__(self, datadir, split='train', downsample=1.0, is_stack=False, N_vis=-1, hold_every=8):

        self.N_vis = N_vis
        self.root_dir = datadir
        self.split = split
        self.is_stack = is_stack
        self.hold_every = hold_every
        self.img_wh = (int(800 / downsample), int(600 / downsample))

        # self.white_bg = True
        self.white_bg = False

        # self.scene_bbox = torch.tensor([[-1.5, -1.5, -1.5], [1.5, 1.5, 1.5]])
        self.scene_bbox = torch.tensor([[-1., -1., -1.], [1., 1., 1.]])
        self.read_meta()
        self.define_proj_mat()

        self.near_far = [0., 6.0]

        self.center = torch.mean(self.scene_bbox, axis=0).float().view(1, 1, 3)
        self.radius = (self.scene_bbox[1] - self.center).float().view(1, 1, 3)
        self.downsample = downsample

        # scan = int(datadir.split('/')[-1][8:])
        # self.render_path = torch.from_numpy(
        #     np.load(os.path.join(datadir, f'render_c2w_{scan}.npy'))
        # ).float()

    def read_meta(self):
        w, h = self.img_wh
        self.render_cameras_name = 'cameras_sphere.npz'
        camera_dict = np.load(os.path.join(self.root_dir, self.render_cameras_name))
        self.camera_dict = camera_dict
        images_lis = sorted(glob(os.path.join(self.root_dir, 'image/*.png')))
        masks_lis = sorted(glob(os.path.join(self.root_dir, 'mask/*.png')))
        n_imgs = len(images_lis)
        i_test = np.arange(n_imgs)[::self.hold_every][1:-1]
        i_train = np.array([j for j in range(n_imgs) if j not in i_test])
        id_list = i_train if self.split == "train" else i_test

        # if self.split == 'test':
        #     id_list = np.hstack([id_list, i_train[:3]])

        self.image_paths = []
        self.poses = []
        self.all_rays = []
        self.all_rgbs = []
        self.all_masks = []
        self.downsample = 1.0
        self.images = []
        intrinsics_all = []
        for i in id_list:
            self.image_paths.append(images_lis[i])
            image = imageio.imread(images_lis[i]).astype(np.float32) / 255.
            image = image[..., :3]
            mask = imageio.imread(masks_lis[i]).astype(np.float32) / 255.
            mask = mask[..., :3]
            image *= mask
            if self.white_bg:
                image += (1 - mask[..., :3])
                image[image > 1.] = 1.

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

            pose = torch.from_numpy(pose).float()

            self.directions = get_ray_directions(
                h, w, [intrinsics[0, 0], intrinsics[1, 1]], [intrinsics[0, 2], intrinsics[1, 2]])  # (h, w, 3)
            self.directions = self.directions / torch.norm(self.directions, dim=-1, keepdim=True)
            rays_o, rays_d = get_rays(self.directions, pose)  # both (h*w, 3)
            self.all_rays += [torch.cat([rays_o, rays_d], 1)]  # (h*w, 6)
            self.all_rgbs.append(torch.from_numpy(image.reshape([-1, 3])).float())
            self.images.append(torch.from_numpy(image).float())
            intrinsics_all.append(intrinsics)
            self.poses.append(pose)
        self.images = torch.stack(self.images, dim=0)
        self.img_wh = (halfres_w, halfres_h)
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
