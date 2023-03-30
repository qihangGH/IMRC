import svox2
import svox
import math
import argparse
from os import path
from tqdm import tqdm
import torch
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--ckpt', type=str, default=
'./ckpt/20220621_human/opt_color_density/human_ckpt.npz'
# '/lkq/lkq/svox2/ckpt/chair_ckpt/opt_color_update_density/ckpt_color.npz'
# 'ckpt/20220614_chair_256/opt_color_update_density/ckpt.npz'
                    )
args = parser.parse_args()

grid = svox2.SparseGrid.load(args.ckpt)
sh_data = grid.sh_data.data
sh_data[:] = 0.
sh_data[:, 0] = 2.0 * math.sqrt(np.pi)
sh_data[:, 9] = 2.0 * math.sqrt(np.pi)
sh_data[:, 18] = 2.0 * math.sqrt(np.pi)
grid.sh_data.data = sh_data
t = grid.to_svox1()
print(t)

out_path = path.splitext(args.ckpt)[0] + '_svox1.npz'
print('Saving', out_path)
t.save(out_path)
