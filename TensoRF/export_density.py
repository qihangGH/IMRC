import os

import torch
from tqdm.auto import tqdm
from opt import config_parser

from renderer import *
from utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def export_density(args):
    ckpt = torch.load(args.ckpt, map_location=device)
    kwargs = ckpt['kwargs']
    kwargs.update({'device': device})
    tensorf = eval(args.model_name)(**kwargs)
    tensorf.load(ckpt)

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
    density_vol = None
    all_alpha_mask = None
    for i in tqdm(range(iter_times)):
        alphas = tensorf.alphaMask.sample_alpha(world_coord[i * chunk_size:(i + 1) * chunk_size])
        alpha_mask = alphas > 0
        density = torch.zeros_like(alphas)
        # print(torch.sum(alpha_mask))
        if alpha_mask.any():
            xyz = tensorf.normalize_coord(world_coord[i * chunk_size:(i + 1) * chunk_size])
            sigma_feature = tensorf.compute_densityfeature(xyz[alpha_mask])
            valid_density = tensorf.feature2density(sigma_feature)
            density[alpha_mask] = valid_density

        if density_vol is None:
            density_vol = density
        else:
            density_vol = torch.cat([density_vol, density], dim=0)

        if all_alpha_mask is None:
            all_alpha_mask = alpha_mask
        else:
            all_alpha_mask = torch.cat([all_alpha_mask, alpha_mask], dim=0)

    print("Activation:", tensorf.fea2denseAct)
    density_vol = density_vol.reshape(reso).cpu().numpy()

    print(density_vol.max(), density_vol.min())
    # compensate for the in-consistence between the sample step size and render step size
    density_vol *= tensorf.distance_scale
    print(density_vol.max(), density_vol.min())

    if args.dataset_name == 'llff':
        density_vol = density_vol[::-1, ::-1]
    np.save(os.path.join(os.path.dirname(args.ckpt), f'density_volume_{reso[0]}.npy'), density_vol)


def grid2world(points, gsz, radius):
    return points * 2. * radius / gsz + radius * (1. / gsz - 1.)


if __name__ == '__main__':
    torch.set_default_dtype(torch.float32)
    torch.manual_seed(20211202)
    np.random.seed(20211202)
    args = config_parser()
    print(args)
    export_density(args)
