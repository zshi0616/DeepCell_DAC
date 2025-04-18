from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import deepcell
import torch
import os
from config import get_parse_args
from deepcell.dg_model import Model as DGModel
from deepcell.dg_trainer import Trainer as DGTrainer

import deepgate as dg

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
DATA_DIR = './data/lcm_sample'
checkpoint = './ckpt/dg2_default.pth'

if __name__ == '__main__':
    args = get_parse_args()
    
    print('[INFO] Create Model and Trainer')
    model = dg.Model()
    model.load(checkpoint)
    print('[INFO] Parse Dataset')
    dataset = deepcell.NpzParser_Pair(args, DATA_DIR, random_sample=0.1)
    train_dataset, val_dataset = dataset.get_dataset()
    print('[INFO] Dataset Size Train: {:} / Test: {:}'.format(len(train_dataset), len(val_dataset)))
    
    trainer = DGTrainer(args, model, distributed=args.distributed, training_id=args.exp_id)
    if args.resume:
        trainer.resume()
    # trainer.set_training_args(loss_weight=[1.0, 0.0, 1.0], lr=1e-4, lr_step=80)
    # print('[INFO] Stage 1 Training ...')
    # trainer.train(10, train_dataset, val_dataset)
    
    trainer.set_training_args(loss_weight=[3.0, 1.0, 0.0], lr=1e-4, lr_step=20)
    print('[INFO] Stage 1 Training ...')
    trainer.train(args.num_epochs, train_dataset, val_dataset)
    
    
    