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
from deepcell.pg_model import Model as PolarGate
import deepgate as dg

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
DATA_DIR = './data/lcm'

if __name__ == '__main__':
    args = get_parse_args()
    num_epochs = args.num_epochs
    
    print('[INFO] Parse Dataset')
    dataset = deepcell.NpzParser_Pair(args, DATA_DIR, random_sample=0.05)
    train_dataset, val_dataset = dataset.get_dataset()
    print('[INFO] Create Model and Trainer')
    model = deepcell.top_model.TopModel(
        args, 
        dc_ckpt='./ckpt/dc_0413.pth', 
        dg_ckpt='./ckpt/dg_0417.pth'
    )
    
    trainer = deepcell.top_trainer.TopTrainer(args, model, distributed=args.distributed, device=args.device)
    if args.resume:
        trainer.resume()
    trainer.set_training_args(lr=1e-4, lr_step=50, loss_weight=[1, 0, 1])
    print('[INFO] Stage 1 Training ...')
    trainer.train(num_epochs, train_dataset, val_dataset)
    
    