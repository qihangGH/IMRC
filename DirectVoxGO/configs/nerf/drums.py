_base_ = '../default.py'

expname = 'dvgo_drums'
# expname = 'drums_dis_0.01'
basedir = './logs/nerf_synthetic'

data = dict(
    datadir='path/to/data/nerf_synthetic/drums',
    dataset_type='blender',
    white_bkgd=False,
    inverse_y=True
)

