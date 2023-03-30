import os.path
import time

from util import config_util
from util.dataset import datasets

import argparse
import cv2
from datetime import datetime
import svox2
import torch
import numpy as np
import math

from tqdm import tqdm


def grid2world(points, gsz, radius):
    return points * 2. * radius / gsz + radius * (1. / gsz - 1.)


parser = argparse.ArgumentParser()
parser.add_argument('ckpt_path', type=str)
parser.add_argument('--export_reso', default='256,256,256')
parser.add_argument('--export_radius', default='1.,1.,1.')
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == '__main__':
    grid = svox2.SparseGrid.load(args.ckpt_path, device=device)

    print(grid)

    reso = [int(i) for i in args.export_reso.split(',')]
    radius = np.array([float(i) for i in args.export_radius.split(',')], dtype=np.float32)
    reso_np = np.array(reso)
    ii, jj, kk = np.arange(reso[0]), np.arange(reso[1]), np.arange(reso[2])
    i, j, k = np.meshgrid(ii, jj, kk, indexing='ij')
    grid_coord = np.stack([i, j, k], axis=-1)

    world_coord = grid2world(grid_coord, reso_np, radius).reshape([-1, 3])
    world_coord = torch.from_numpy(world_coord).float().cuda()

    chunk_size = 256 ** 3
    iter_times = len(world_coord) // chunk_size
    if len(world_coord) % chunk_size != 0:
        iter_times += 1
    density_vol = None
    for i in range(iter_times):
        density, _ = grid.sample(world_coord[i * chunk_size:(i + 1) * chunk_size],
                                 use_kernel=True, grid_coords=False, want_colors=False)
        density = density[..., 0]
        density_vol = density if density_vol is None else torch.cat([density_vol, density], dim=0)

    density_vol = density_vol.reshape(reso).detach().cpu().numpy()

    print(density_vol.max(), density_vol.min(), density_vol.shape)

    np.save(os.path.join(os.path.dirname(args.ckpt_path), f'density_volume_{reso[0]}.npy'), density_vol)
