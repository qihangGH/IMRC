_base_ = '../default.py'

expname = 'dvgo_dtu40'
basedir = './logs/dtu'

data = dict(
    datadir=r'path/to/public_data/dtu_scan40',
    dataset_type='dtu',
    white_bkgd=False,
    inverse_y=True
)
