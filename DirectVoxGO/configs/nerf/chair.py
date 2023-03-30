_base_ = '../default.py'

expname = 'dvgo_chair'
basedir = './logs/nerf_synthetic'

data = dict(
    datadir='path/to/data/nerf_synthetic/chair',
    dataset_type='blender',
    white_bkgd=False,
    inverse_y=True
)

