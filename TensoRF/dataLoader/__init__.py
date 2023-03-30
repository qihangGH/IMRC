from .llff import LLFFDataset
from .blender import BlenderDataset
from .nsvf import NSVF
from .tankstemple import TanksTempleDataset
from .your_own_data import YourOwnDataset
from .dtu import DTUDataset
from .dtu_mvs import DTUMVSDataset

dataset_dict = {'blender': BlenderDataset,
                'llff': LLFFDataset,
                'tankstemple': TanksTempleDataset,
                'nsvf': NSVF,
                'dtu': DTUDataset,
                'dtu_mvs': DTUMVSDataset,
                'own_data': YourOwnDataset}
