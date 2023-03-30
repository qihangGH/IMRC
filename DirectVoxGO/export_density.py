import os, sys, copy, glob, json, time, random, argparse, math
from shutil import copyfile
from tqdm import tqdm, trange

import mmcv
import imageio
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib import utils, dvgo, dcvgo, dmpigo
from lib.load_data import load_data

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def config_parser():
    '''Define command line arguments
    '''
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--expname', default=None)
    parser.add_argument('--config', required=True, help='config file path')
    parser.add_argument("--seed", type=int, default=777,
                        help='Random seed')
    parser.add_argument("--no_reload", action='store_true',
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--no_reload_optimizer", action='store_true',
                        help='do not reload optimizer state from saved ckpt')
    parser.add_argument("--ft_path", type=str, default='',
                        help='specific weights npy file to reload for coarse network')
    parser.add_argument("--export_bbox_and_cams_only", type=str, default='',
                        help='export scene bbox and camera poses for debugging and 3d visualization')
    parser.add_argument("--export_coarse_only", type=str, default='')

    # export density
    parser.add_argument("--reso", type=str, default=
    "512,512,512"
                        )
    parser.add_argument("--radius", type=str, default=
    "1.,1.,1."
                        )

    # testing options
    parser.add_argument("--render_only", action='store_true',
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_test", action='store_true')
    # parser.add_argument("--render_test", action='store_true', default=True)
    parser.add_argument("--render_train", action='store_true')
    parser.add_argument("--render_video", action='store_true')
    parser.add_argument("--render_video_flipy", action='store_true')
    parser.add_argument("--render_video_rot90", default=0, type=int)
    parser.add_argument("--render_video_factor", type=float, default=0,
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')
    parser.add_argument("--dump_images", action='store_true')
    parser.add_argument("--eval_ssim", action='store_true')
    parser.add_argument("--eval_lpips_alex", action='store_true')
    parser.add_argument("--eval_lpips_vgg", action='store_true')

    # logging/saving options
    parser.add_argument("--i_print", type=int, default=500,
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_weights", type=int, default=100000,
                        help='frequency of weight ckpt saving')
    return parser


def seed_everything():
    '''Seed everything for better reproducibility.
    (some pytorch operation is non-deterministic like the backprop of grid_samples)
    '''
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)


def load_everything(args, cfg):
    '''Load images / poses / camera settings / data split.
    '''
    data_dict = load_data(cfg.data)

    # remove useless field
    kept_keys = {
        'hwf', 'HW', 'Ks', 'near', 'far', 'near_clip',
        'i_train', 'i_val', 'i_test', 'irregular_shape',
        'poses', 'render_poses', 'images'}
    for k in list(data_dict.keys()):
        if k not in kept_keys:
            data_dict.pop(k)

    # construct data tensor
    if data_dict['irregular_shape']:
        data_dict['images'] = [torch.FloatTensor(im, device='cpu') for im in data_dict['images']]
    else:
        data_dict['images'] = torch.FloatTensor(data_dict['images'], device='cpu')
    data_dict['poses'] = torch.Tensor(data_dict['poses'])
    return data_dict


def grid2world(points, gsz, radius):
    return points * 2. * radius / gsz + radius * (1. / gsz - 1.)


if __name__ == '__main__':
    # load setup
    parser = config_parser()
    args = parser.parse_args()
    cfg = mmcv.Config.fromfile(args.config)

    if args.expname is not None:
        cfg.expname = args.expname
    print('Train dir: ', cfg.expname)

    # init environment
    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    seed_everything()

    # load images / poses / camera settings / data split
    data_dict = load_everything(args=args, cfg=cfg)

    # load model for rendering
    if args.ft_path:
        ckpt_path = args.ft_path
    else:
        ckpt_path = os.path.join(cfg.basedir, cfg.expname, 'fine_last.tar')
    ckpt_name = ckpt_path.split('/')[-1][:-4]
    if cfg.data.ndc:
        model_class = dmpigo.DirectMPIGO
    elif cfg.data.unbounded_inward:
        model_class = dcvgo.DirectContractedVoxGO
    else:
        model_class = dvgo.DirectVoxGO
    model = utils.load_model(model_class, ckpt_path).to(device)

    reso = [int(i) for i in args.reso.split(',')]
    radius = np.array([float(i) for i in args.radius.split(',')], dtype=np.float32)
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
    density = None
    with torch.no_grad():
        for i in tqdm(range(iter_times)):
            xyz = world_coord[i * chunk_size:(i + 1) * chunk_size]
            dens = torch.zeros_like(xyz[..., 0])
            mask = model.mask_cache(xyz)
            if mask.any():
                dens_valid = model.density(xyz[mask])
                if cfg.data.dataset_type in ['dtu', 'blender', 'dtu_mvs']:
                    dens_valid = dens_valid + model.act_shift
                elif cfg.data.dataset_type == 'llff':
                    dens_valid = dens_valid + model.act_shift(xyz[mask])
                else:
                    raise ValueError
                dens_valid = F.softplus(dens_valid)
                dens[mask] = dens_valid

            if density is None:
                density = dens
            else:
                density = torch.cat([density, dens], dim=0)

        density_thresh = -math.log(1 - cfg.fine_model_and_render.fast_color_thres) \
                         / (cfg.fine_model_and_render.stepsize * model.voxel_size_ratio)
        print("density thresh:", density_thresh)
        density[density < density_thresh] = 0.

    density = density.reshape(reso).cpu().numpy()
    print("Density range:", density.min(), density.max())
    if cfg.data.dataset_type in ['dtu', 'blender', 'dtu_mvs']:
        density /= model.voxel_size_base.cpu().numpy()
    elif cfg.data.dataset_type == 'llff':
        density *= 256 * (model.mpi_depth - 1) / (2 * model.mpi_depth)
        # Note that we have altered the coordinates.
        # If the model is trained by the original DVGO code, uncomment the line below.
        # density = density[:, ::-1]
    else:
        raise ValueError
    print("Scaled density range:", density.min(), density.max())
    np.save(os.path.join(cfg.basedir, cfg.expname, f'density_volume_{reso[0]}.npy'), density)
