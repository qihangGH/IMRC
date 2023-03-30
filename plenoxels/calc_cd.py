import os
import argparse
import numpy as np
import svox2
import mcubes
import trimesh
import open3d as o3d
import sklearn.neighbors as skln
from scipy.io import loadmat
import multiprocessing as mp
import matplotlib.pyplot as plt

from tqdm import tqdm
from functools import partial

parser = argparse.ArgumentParser('Golden-section search for the best CD')
parser.add_argument('density_path', type=str, help='the density volume filepath or model ckpt')
parser.add_argument('data_dir', type=str, help='training data directory')
parser.add_argument('scan', type=int)
parser.add_argument('gt_data_dir', type=str, help='ground-truth data directory')
parser.add_argument('--min_thresh', type=float, default=
None
# 1.
                    )
parser.add_argument('--max_thresh', type=float, default=
None
# 100.
                    )
parser.add_argument('--downsample_density', type=float, default=0.2)
parser.add_argument('--patch_size', type=float, default=60)
parser.add_argument('--max_dist', type=float, default=20)
parser.add_argument('--visualize_threshold', type=float, default=10)

args = parser.parse_args()


def grid2world(points, gsz):
    return points * 2. / gsz + (1. / gsz - 1.)


def sample_single_tri(input_):
    n1, n2, v1, v2, tri_vert = input_
    c = np.mgrid[:n1 + 1, :n2 + 1]
    c += 0.5
    c[0] /= max(n1, 1e-7)
    c[1] /= max(n2, 1e-7)
    c = np.transpose(c, (1, 2, 0))
    k = c[c.sum(axis=-1) < 1]  # m2
    q = v1 * k[:, :1] + v2 * k[:, 1:] + tri_vert
    return q


def _eval_mesh(vertices, triangles):
    mp.freeze_support()

    thresh = args.downsample_density
    tri_vert = vertices[triangles]

    v1 = tri_vert[:, 1] - tri_vert[:, 0]
    v2 = tri_vert[:, 2] - tri_vert[:, 0]
    l1 = np.linalg.norm(v1, axis=-1, keepdims=True)
    l2 = np.linalg.norm(v2, axis=-1, keepdims=True)
    area2 = np.linalg.norm(np.cross(v1, v2), axis=-1, keepdims=True)
    non_zero_area = (area2 > 0)[:, 0]
    l1, l2, area2, v1, v2, tri_vert = [
        arr[non_zero_area] for arr in [l1, l2, area2, v1, v2, tri_vert]
    ]
    thr = thresh * np.sqrt(l1 * l2 / area2)
    n1 = np.floor(l1 / thr)
    n2 = np.floor(l2 / thr)

    with mp.Pool() as mp_pool:
        new_pts = mp_pool.map(
            sample_single_tri,
            ((n1[i, 0], n2[i, 0], v1[i:i + 1], v2[i:i + 1], tri_vert[i:i + 1, 0]) for i in range(len(n1))),
            chunksize=1024)

    new_pts = np.concatenate(new_pts, axis=0)
    data_pcd = np.concatenate([vertices, new_pts], axis=0)

    shuffle_rng = np.random.default_rng()
    shuffle_rng.shuffle(data_pcd, axis=0)

    nn_engine = skln.NearestNeighbors(n_neighbors=1, radius=thresh, algorithm='kd_tree', n_jobs=-1)
    nn_engine.fit(data_pcd)
    rnn_idxs = nn_engine.radius_neighbors(data_pcd, radius=thresh, return_distance=False)
    mask = np.ones(data_pcd.shape[0], dtype=np.bool_)
    for curr, idxs in enumerate(rnn_idxs):
        if mask[curr]:
            mask[idxs] = 0
            mask[curr] = 1
    data_down = data_pcd[mask]

    obs_mask_file = loadmat(f'{args.dataset_dir}/ObsMask/ObsMask{args.scan}_10.mat')
    ObsMask, BB, Res = [obs_mask_file[attr] for attr in ['ObsMask', 'BB', 'Res']]
    BB = BB.astype(np.float32)

    patch = args.patch_size
    inbound = ((data_down >= BB[:1] - patch) & (data_down < BB[1:] + patch * 2)).sum(axis=-1) == 3
    data_in = data_down[inbound]

    data_grid = np.around((data_in - BB[:1]) / Res).astype(np.int32)
    grid_inbound = ((data_grid >= 0) & (data_grid < np.expand_dims(ObsMask.shape, 0))).sum(axis=-1) == 3
    data_grid_in = data_grid[grid_inbound]
    in_obs = ObsMask[data_grid_in[:, 0], data_grid_in[:, 1], data_grid_in[:, 2]].astype(np.bool_)
    data_in_obs = data_in[grid_inbound][in_obs]

    stl_pcd = o3d.io.read_point_cloud(f'{args.dataset_dir}/Points/stl/stl{args.scan:03}_total.ply')
    stl = np.asarray(stl_pcd.points)

    nn_engine.fit(stl)
    dist_d2s, idx_d2s = nn_engine.kneighbors(data_in_obs, n_neighbors=1, return_distance=True)
    max_dist = args.max_dist
    mean_d2s = dist_d2s[dist_d2s < max_dist].mean()

    ground_plane = loadmat(f'{args.dataset_dir}/ObsMask/Plane{args.scan}.mat')['P']

    stl_hom = np.concatenate([stl, np.ones_like(stl[:, :1])], -1)
    above = (ground_plane.reshape((1, 4)) * stl_hom).sum(-1) > 0
    stl_above = stl[above]

    nn_engine.fit(data_in)
    dist_s2d, idx_s2d = nn_engine.kneighbors(stl_above, n_neighbors=1, return_distance=True)
    mean_s2d = dist_s2d[dist_s2d < max_dist].mean()

    over_all = (mean_d2s + mean_s2d) / 2
    return mean_d2s, mean_s2d, over_all


def eval_mesh(density_volume, scale_mat, thresh):
    vertices, triangles = mcubes.marching_cubes(density_volume, thresh)
    vertices = grid2world(vertices, np.array(density_volume.shape))
    vertices = vertices * scale_mat[0, 0] + scale_mat[:3, 3][None]
    mean_d2s, mean_s2d, over_all = _eval_mesh(vertices, triangles)
    return mean_d2s, mean_s2d, over_all


if __name__ == '__main__':
    # load the density volume
    print("Density filepath:", args.density_path)
    print("Data dir:", args.data_dir)
    print("DTU scan", args.scan)
    if args.density_path.endwith('.npy'):
        density_volume = np.load(args.density_path)
    elif args.density_path.endwith('.npz'):
        ckpt = svox2.SparseGrid.load(args.model_path, device='cpu')
        density = ckpt.density_data.data.numpy().reshape([-1])
        links = ckpt.links.numpy()
        density_volume = np.zeros_like(links).astype(np.float32)
        mask = links >= 0
        density_volume[mask] = density[links[mask]]
    else:
        raise ValueError('`args.density_path` should be .npy density volume or .npz model ckpt')
    max_dens = np.max(density_volume)
    min_dens = np.min(density_volume)
    print(f'Density range: max: {max_dens}\n min: {min_dens}')

    if min_dens < 0.:
        min_dens = 0.

    density_volume[density_volume < 0.] = 0.
    camera_dict = np.load(os.path.join(args.data_dir, 'cameras_sphere.npz'))
    scale_mat = camera_dict['scale_mat_0'].astype(np.float32)

    if max_dens > 1000:
        min_thresh = 5.
        max_thresh = 500.
    elif max_dens > 100:
        min_thresh = 5.
        max_thresh = 200.
    else:
        min_thresh = min_dens if min_dens > 0 else max_dens / 10
        max_thresh = min_thresh + 5 * (max_dens - min_thresh) / 10

    if args.min_thresh is not None:
        min_thresh = args.min_thresh
    if args.max_thresh is not None:
        max_thresh = args.max_thresh
    print(f'min_thresh: {min_thresh}; max_thresh: {max_thresh}')

    EVAL_MESH = partial(eval_mesh, density_volume, scale_mat)

    # Golden-section search for the best CD
    num_max_iter = 50
    t = 0.618
    record_mean_d2s, record_mean_s2d, record_over_all, record_thresh = \
        [np.zeros([num_max_iter + 1, 2]) for _ in range(4)]
    record_thresh[0, 0] = min_thresh + (1 - t) * (max_thresh - min_thresh)
    record_thresh[0, 1] = min_thresh + t * (max_thresh - min_thresh)
    record_mean_d2s[0, 0], record_mean_s2d[0, 0], record_over_all[0, 0] = EVAL_MESH(record_thresh[0, 0])
    record_mean_d2s[0, 1], record_mean_s2d[0, 1], record_over_all[0, 1] = EVAL_MESH(record_thresh[0, 1])

    for i in tqdm(range(num_max_iter)):
        if np.abs(record_over_all[i, 0] - record_over_all[i, 1]) < 1e-3:
            break
        if record_over_all[i, 0] < record_over_all[i, 1]:
            max_thresh = record_thresh[i, 1]
            record_thresh[i + 1, 1] = record_thresh[i, 0]
            record_mean_d2s[i + 1, 1], record_mean_s2d[i + 1, 1], record_over_all[i + 1, 1] = \
                record_mean_d2s[i, 0], record_mean_s2d[i, 0], record_over_all[i, 0]
            record_thresh[i + 1, 0] = min_thresh + (1 - t) * (max_thresh - min_thresh)
            record_mean_d2s[i + 1, 0], record_mean_s2d[i + 1, 0], record_over_all[i + 1, 0] = EVAL_MESH(
                record_thresh[i + 1, 0])
        else:
            min_thresh = record_thresh[i, 0]
            record_thresh[i + 1, 0] = record_thresh[i, 1]
            record_mean_d2s[i + 1, 0], record_mean_s2d[i + 1, 0], record_over_all[i + 1, 0] = \
                record_mean_d2s[i, 1], record_mean_s2d[i, 1], record_over_all[i, 1]
            record_thresh[i + 1, 1] = min_thresh + t * (max_thresh - min_thresh)
            record_mean_d2s[i + 1, 1], record_mean_s2d[i + 1, 1], record_over_all[i + 1, 1] = EVAL_MESH(
                record_thresh[i + 1, 1])
    record_thresh = record_thresh.reshape([-1])
    record_mean_d2s = record_mean_d2s.reshape([-1])
    record_mean_s2d = record_mean_s2d.reshape([-1])
    record_over_all = record_over_all.reshape([-1])
    valid_id = record_thresh > 0.
    record_thresh = record_thresh[valid_id]
    record_mean_d2s = record_mean_d2s[valid_id]
    record_mean_s2d = record_mean_s2d[valid_id]
    record_over_all = record_over_all[valid_id]

    arg_idx = np.argsort(record_thresh)
    record_thresh = record_thresh[arg_idx]
    record_mean_d2s = record_mean_d2s[arg_idx]
    record_mean_s2d = record_mean_s2d[arg_idx]
    record_over_all = record_over_all[arg_idx]

    min_idx = np.argmin(record_over_all)

    print("Thresholds: ", record_thresh)
    print("Metric: ", record_over_all)
    print(f"Best thresh and metric: {record_thresh[min_idx]}; {record_over_all[min_idx]}")

    np.save(os.path.join(os.path.dirname(args.density_path), 'threshs.npy'), record_thresh)
    np.save(os.path.join(os.path.dirname(args.density_path),
                         f'{args.density_path.split(os.sep)[-2]}_mean_d2s.npy'), record_mean_d2s)
    np.save(os.path.join(os.path.dirname(args.density_path),
                         f'{args.density_path.split(os.sep)[-2]}_mean_s2d.npy'), record_mean_s2d)
    np.save(os.path.join(os.path.dirname(args.density_path),
                         f'{args.density_path.split(os.sep)[-2]}_over_all.npy'), record_over_all)
