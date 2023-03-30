_base_ = '../default.py'

expname = 'dvgo_lego'
basedir = './logs/nerf_synthetic'

data = dict(
    datadir='path/to/data/nerf_synthetic/lego',
    dataset_type='blender',
    white_bkgd=False,
    inverse_y=True
)

