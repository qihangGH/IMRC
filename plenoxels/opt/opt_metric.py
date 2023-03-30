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
from util.util import Timing
from util.cf_color_utils import calc_color_metric
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
config_util.define_common_args(parser)
parser.add_argument('ckpt_path', type=str)
parser.add_argument('--use_ndc', default=False, action='store_true')
parser.add_argument('--render_cf_test', default=False, action='store_true')
parser.add_argument('--render_cf_train', default=False, action='store_true')
args = parser.parse_args()
config_util.maybe_merge_config_file(args, allow_invalid=True)
# print(args.basis_dim)

suffix = '_high_fq' if args.high_fq else ''

device = "cuda" if torch.cuda.is_available() else "cpu"
factor = 1

grid = svox2.SparseGrid.load(args.ckpt_path, device=device)


config_util.setup_render_opts(grid.opt, args)

if args.white_bkgd:
    grid.opt.background_brightness = 1.
else:
    grid.opt.background_brightness = 0.
print(grid)

if args.high_fq:
    dset = datasets[args.dataset_type](
        args.data_dir,
        split='test',
        device=device,
        factor=factor,
        **config_util.build_data_options(args))
    assert args.dataset_type != 'auto'
    print(args.dataset_type)
    dset1 = datasets[args.dataset_type](
        args.data_dir,
        split='train',
        device=device,
        factor=factor,
        **config_util.build_data_options(args))
    if args.dataset_type in ['llff', 'dtu', 'dtu_mvs']:
        dset.c2w = torch.concat([dset.c2w, dset1.c2w], dim=0)
        dset.gt = torch.concat([dset.gt, dset1.gt], dim=0)
else:
    dset = datasets[args.dataset_type](
        args.data_dir,
        split='train',
        device=device,
        factor=factor,
        **config_util.build_data_options(args))

print(f'Dataset type: \033[1;31m{args.dataset_type}\033[0m; '
      f'Num images for color estimation: \033[1;31m{len(dset.gt)}\033[0m')

print(f'Use NDC: \033[1;31m{args.use_ndc}\033[0m')

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

c2ws = dset.c2w.to(device=device).float()
w2cs = torch.inverse(c2ws).contiguous()

intrinsics = []
for i in range(c2ws.shape[0]):
    intrin = [
        dset.intrins.get('fx', i),
        dset.intrins.get('fy', i),
        dset.intrins.get('cx', i),
        dset.intrins.get('cy', i),
        dset.get_image_size(i)[1],
        dset.get_image_size(i)[0]
    ]
    if dset.ndc_coeffs[0] != -1:
        intrin.extend([
            2 * dset.intrins.get('fx', i) / dset.get_image_size(i)[1],
            2 * dset.intrins.get('fy', i) / dset.get_image_size(i)[0],
            1.0
        ])
    intrinsics.append(torch.tensor(intrin))
intrinsics = torch.stack(intrinsics).to(device=device)
images = dset.gt.to(device=device)
print(images.shape)

print(f'Density range: max {torch.max(grid.density_data.data)}, min {torch.min(grid.density_data.data)}')

with torch.no_grad():
    print('Start estimate color ...')
    with Timing("Color estimation") as timer:
        color_estimated, color_weight, color_res = \
            grid.estimate_color_metric(c2ws, w2cs, intrinsics, images)
    with open(os.path.join(os.path.dirname(args.ckpt_path), f"milisecs.txt"), "w") as f:
        f.write("{}".format(np.array(timer.elapsed)))

# Verify the estimated result.
grid.sh_data.data[:] = 0.
grid.sh_data.data = color_estimated.data


@torch.no_grad()
def test_psnr(split='test'):
    save_dir = os.path.join(os.path.dirname(args.ckpt_path), split + f'{suffix}')
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
    images = dset.gt.to(device=device)
    print(images.shape)
    psnr_avg = 0.
    print('Start render image')
    start_time = datetime.now()
    for i in tqdm(range(dset.gt.size(0))):
        img = grid.volume_render_image(cameras[i])
        cv2.imwrite(os.path.join(save_dir,  f'{split}_{i}_render.png'),
                    img.detach().cpu().numpy()[:, :, ::-1] * 255)
        psnr_avg += -10. * torch.log10(torch.sum((img - images[i, :, :, :]) ** 2) / img.numel())

    stop_time = datetime.now()
    secs = (stop_time - start_time).total_seconds()
    psnr_avg = psnr_avg / dset.gt.size(0)
    print(f"Render time: {secs} sec")
    print(f'{split} psnr_avg: {psnr_avg}')
    return psnr_avg


if __name__ == '__main__':
    if args.render_cf_test:
        test_psnr('test')
    if args.render_cf_train:
        test_psnr('train')
    metric, color_res_vol, color_weight_vol = calc_color_metric(grid, color_res, color_weight)
    metric = -10 * np.log10(metric.cpu().numpy())
    np.savetxt(os.path.join(os.path.dirname(args.ckpt_path), f'IMRC{suffix}.txt'),
               np.array([metric]))
    color_res_vol = color_res_vol.detach().cpu().numpy()
    color_weight_vol = color_weight_vol.detach().cpu().numpy()
    links = grid.links.cpu().numpy()
    mask = links >= 0
    color_res_volume = np.zeros_like(links).astype(np.float32)
    color_weight_volume = np.zeros_like(links).astype(np.float32)
    color_res_volume[mask] = color_res_vol[links[mask]]
    color_weight_volume[mask] = color_weight_vol[links[mask]]

    np.save(os.path.join(os.path.dirname(args.ckpt_path), f'color_res_vol_norm2{suffix}.npy'), color_res_volume)
    # np.save(os.path.join(os.path.dirname(args.ckpt_path), f'color_weight_vol.npy'), color_weight_volume)
    print(f'IMRC: {metric}')
