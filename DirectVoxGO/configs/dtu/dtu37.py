_base_ = '../default.py'

expname = 'dvgo_dtu37'
basedir = './logs/dtu'

data = dict(
    datadir=r'path/to/public_data/dtu_scan37',
    dataset_type='dtu',
    white_bkgd=False,
    inverse_y=True
)
