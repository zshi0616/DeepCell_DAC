from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import deepcell
import torch
import os
from config import get_parse_args
import deepcell.top_model
import deepcell.top_trainer 
from torch_geometric.data import Data

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
DATA_DIR = './data/lcm'

class OrderedData(Data):
    def __init__(self): 
        super().__init__()

if __name__ == '__main__':
    args = get_parse_args()
    aig_path = './npz/iccad_aig.npz'
    pm_path_list = [
        './npz/gf180.npz', './npz/gscl45.npz', './npz/sky130.npz'
    ]
    
    num_epochs = args.num_epochs
    
    print('[INFO] Parse Dataset')
    dataset = deepcell.NpzParser_Pair(args, DATA_DIR, aig_path, pm_path_list)
    print('Done!!!')
    
