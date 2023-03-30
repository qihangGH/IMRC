import functools
import os
from os import path

from absl import app
from absl import flags
from tqdm import tqdm

import flax
from flax.metrics import tensorboard
from flax.training import checkpoints
from flax.core.frozen_dict import freeze
import jax
from jax import random
import numpy as np
import jax.numpy as jnp

from jaxnerf.nerf import datasets
from jaxnerf.nerf import models
from jaxnerf.nerf import utils
from jaxnerf.nerf import model_utils

FLAGS = flags.FLAGS

utils.define_flags()


def grid2world(points, gsz, radius):
    return points * 2. * radius / gsz + radius * (1. / gsz - 1.)


def main(unused_argv):
    # `center` is [0., 0., 0.] by default
    radius = jnp.array([float(i) for i in FLAGS.radius.split(',')], dtype=jnp.float32)
    reso = [int(i) for i in FLAGS.reso.split(',')]
    reso_jnp = jnp.array(reso)
    ii, jj, kk = jnp.arange(reso[0]), jnp.arange(reso[1]), jnp.arange(reso[2])
    i, j, k = jnp.meshgrid(ii, jj, kk, indexing='ij')
    grid_coord = jnp.stack([i, j, k], axis=-1)
    world_coord = grid2world(grid_coord, reso_jnp, radius).reshape([-1, 3])
    dirs = jnp.ones_like(world_coord)
    dirs /= jnp.linalg.norm(dirs[0])

    rng = random.PRNGKey(20200823)

    if FLAGS.config is not None:
        utils.update_flags(FLAGS)

    dataset = datasets.get_dataset("test", FLAGS)
    rng, key = random.split(rng)
    model, init_variables = models.get_model(key, dataset.peek(), FLAGS)
    optimizer = flax.optim.Adam(FLAGS.lr_init).create(init_variables)
    state = utils.TrainState(optimizer=optimizer)
    del optimizer, init_variables

    samples_enc = model_utils.posenc(
        world_coord[:10],
        model.min_deg_point,
        model.max_deg_point,
        model.legacy_posenc_order
    )
    mlp = model_utils.MLP(
        net_depth=model.net_depth,
        net_width=model.net_width,
        net_depth_condition=model.net_depth_condition,
        net_width_condition=model.net_width_condition,
        net_activation=model.net_activation,
        skip_layer=model.skip_layer,
        num_rgb_channels=model.num_rgb_channels,
        num_sigma_channels=model.num_sigma_channels
    )
    variables = mlp.init(jax.random.PRNGKey(1), samples_enc[:, None])

    state = checkpoints.restore_checkpoint(FLAGS.train_dir, state, step=FLAGS.step)
    variables = variables.unfreeze()
    if FLAGS.export_with_fine is False:
        suffix = 'coarse'
        variables['params'] = state.optimizer.target['params']['MLP_0']
    else:
        suffix = 'fine'
        variables['params'] = state.optimizer.target['params']['MLP_1']
    variables = freeze(variables)

    chunk_size = FLAGS.export_chunk_size
    iter_times = len(world_coord) // chunk_size
    if len(world_coord) % chunk_size != 0:
        iter_times += 1
    sigma = []
    for i in tqdm(range(iter_times)):
        samples_enc = model_utils.posenc(
            world_coord[i * chunk_size:(i + 1) * chunk_size],
            model.min_deg_point,
            model.max_deg_point,
            model.legacy_posenc_order
        )
        viewdirs_enc = model_utils.posenc(
            dirs[i * chunk_size:(i + 1) * chunk_size],
            0,
            model.deg_view,
            model.legacy_posenc_order
        )
        _, raw_sigma = mlp.apply(variables, samples_enc[:, None], viewdirs_enc[:, None])
        sigma.append(np.squeeze(np.asarray(raw_sigma)))
    density = np.hstack(sigma).reshape(reso)
    print(density.max(), density.min())
    density[density < 0.] = 0.
    if FLAGS.dataset == 'llff':
        density = density[:, ::-1].copy()

    if FLAGS.step is not None:
        suffix = repr(FLAGS.step) + '_' + suffix
    np.save(os.path.join(FLAGS.train_dir, f'density_volume_{suffix}_{reso[0]}.npy'), density)


if __name__ == "__main__":
    app.run(main)
