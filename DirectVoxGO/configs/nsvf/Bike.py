_base_ = '../default.py'

expname = 'dvgo_Bike'
basedir = './logs/nsvf_synthetic'

data = dict(
    datadir='./data/Synthetic_NSVF/Bike',
    dataset_type='nsvf',
    inverse_y=True,
    white_bkgd=True,
)

