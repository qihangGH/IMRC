import svox2.utils
import os
import os.path

import matplotlib.pyplot as plt

from util import config_util
from util.dataset import datasets

import argparse
import cv2
from datetime import datetime
import svox2
import torch
import numpy as np

from tqdm import tqdm

from svox2 import utils
from util.cf_color_utils import Grid

utils.MAX_SH_BASIS = 16

parser = argparse.ArgumentParser('Visualize residual colors')
config_util.define_common_args(parser)
parser.add_argument('filepath', type=str)
parser.add_argument('--use_ndc', default=False, action='store_true')
parser.add_argument('--basis_dim', default=9)
parser.add_argument('--step', default=None)
parser.add_argument('--split', type=str, default='test')
parser.add_argument('--color_scale', type=float, default=25)
args = parser.parse_args()
config_util.maybe_merge_config_file(args, allow_invalid=True)
# print(args)
args.basis_dim = int(args.basis_dim)


device = "cuda" if torch.cuda.is_available() else "cpu"
factor = 1

print(f'Filepath: {args.filepath}')

dset = datasets[args.dataset_type](
    args.data_dir,
    split="train",
    device=device,
    factor=factor,
    **config_util.build_data_options(args))

color_res = np.load(args.filepath)
color_res_volume = color_res
color_res_volume = torch.from_numpy(color_res_volume).float()

grid = Grid(
    color_res_volume,
    reso=list(color_res.shape),
    center=dset.scene_center,
    radius=dset.scene_radius,
    use_sphere_bound=False,
    basis_dim=args.basis_dim,
    use_z_order=False,
    device=device,
    basis_reso=32,
    basis_type=svox2.__dict__['BASIS_TYPE_SH']
)

config_util.setup_render_opts(grid.opt, args)
grid.opt.background_brightness = 0.
print(grid)
print(grid.opt)


@torch.no_grad()
def render_res_color(split):
    save_dir = args.filepath[:-4]
    os.makedirs(save_dir, exist_ok=True)
    dset = datasets[args.dataset_type](
        args.data_dir,
        split=split,
        device=device,
        factor=factor,
        **config_util.build_data_options(args))
    cameras = [
        svox2.Camera(c2w.to(device=device),
                     dset.intrins.get('fx', i),
                     dset.intrins.get('fy', i),
                     dset.intrins.get('cx', i),
                     dset.intrins.get('cy', i),
                     width=dset.get_image_size(i)[1],
                     height=dset.get_image_size(i)[0],
                     ndc_coeffs=dset.ndc_coeffs) for i, c2w in enumerate(dset.c2w)
    ]
    print('Start render image')
    start_time = datetime.now()
    for i in tqdm(range(dset.gt.size(0))):
        img = grid.volume_render_acc_image(cameras[i])
        img = img.detach().cpu().numpy()
        img = img * args.color_scale
        img[img > 1.] = 1.
        img = plt.cm.viridis(img)[..., :-1]  # output the same value if img > 1
        cv2.imwrite(os.path.join(save_dir, f'{split}_color_res{i}.png'),
                    img[:, :, ::-1] * 255)
    stop_time = datetime.now()
    secs = (stop_time - start_time).total_seconds()
    print(f"Render time: {secs} sec")


if __name__ == '__main__':
    render_res_color(args.split)
