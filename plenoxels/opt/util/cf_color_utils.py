import svox2
import torch

from torch import nn
from typing import *

device = "cuda" if torch.cuda.is_available() else "cpu"


class Grid(svox2.SparseGrid):
    def __init__(
            self,
            density_volume,
            reso: Union[int, List[int], Tuple[int, int, int]] = 128,
            radius: Union[float, List[float]] = 1.0,
            center: Union[float, List[float]] = [0.0, 0.0, 0.0],
            basis_type: int = 0,
            basis_dim: int = 9,  # SH/learned basis size; in SH case, square number
            basis_reso: int = 16,  # Learned basis resolution (x^3 embedding grid)
            use_z_order: bool = False,
            use_sphere_bound: bool = False,
            mlp_posenc_size: int = 0,
            mlp_width: int = 16,
            background_nlayers: int = 0,  # BG MSI layers
            background_reso: int = 256,  # BG MSI cubemap face size
            device: Union[torch.device, str] = "cpu"
    ):
        super(Grid, self).__init__(
            reso, radius, center, basis_type, basis_dim, basis_reso,
            use_z_order, use_sphere_bound, mlp_posenc_size, mlp_width,
            background_nlayers, background_reso, device
        )

        self.opt.sigma_thresh = 1e-8
        # white_bkgd needs to be set
        self.opt.background_brightness = 0.
        self.opt.random_sigma_std = 0.
        self.opt.random_sigma_std_background = 0.

        # sparsify to accelerate
        n3 = density_volume.shape[0] * density_volume.shape[1] * density_volume.shape[2]
        init_links = torch.arange(n3, dtype=torch.int32).cuda()
        mask = density_volume.reshape([-1]) > self.opt.sigma_thresh
        capacity = mask.sum()
        self.capacity = capacity

        self.density_data = nn.Parameter(
            torch.zeros(self.capacity, 1, dtype=torch.float32, device=device)
        )
        self.sh_data = nn.Parameter(
            torch.zeros(
                self.capacity, self.basis_dim * 3, dtype=torch.float32, device=device
            )
        )

        data_mask = torch.zeros(n3, dtype=torch.int32, device=device)
        idxs = init_links[mask].long()
        data_mask[idxs] = 1
        data_mask = torch.cumsum(data_mask, dim=0) - 1

        init_links[mask] = data_mask[idxs].int()
        init_links[~mask] = -1

        self.density_data.data = density_volume.reshape([-1, 1])[mask].to(device).float()

        # Init the links inverse.
        init_links_inverse = torch.zeros(capacity, dtype=torch.int32, device=device)
        links_pos = torch.arange(init_links.numel(), device=device, dtype=torch.int32)
        links_valid = init_links[init_links >= 0]
        links_pos_valid = links_pos[init_links >= 0]
        init_links_inverse[links_valid.long()] = links_pos_valid

        init_links = init_links.reshape(density_volume.shape)

        self.register_buffer("links", init_links)
        self.register_buffer("links_inverse", init_links_inverse)
        self.links.data = init_links
        self.links_inverse = init_links_inverse
        self.accelerate()


@torch.no_grad()
def calc_color_metric(grid, color_res, color_weight):
    gsz = grid._grid_size()
    delta = grid.opt.step_size / (grid._scaling * gsz)
    print(f'Delta: \033[1;31m{delta}\033[0m')
    delta = delta[0]
    density = grid.density_data.detach()[..., 0].clone()
    density[density < 0.] = 0.
    weight = 1. - torch.exp(-density * delta)

    return torch.sum(weight * color_res) / (torch.sum(weight * color_weight) + 1e-9), \
           weight * color_res, weight * color_weight
