from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import deepcell
import torch
import os
from config import get_parse_args
import deepcell.top_model
from deepcell.eco_model import TopModel
from deepcell.eco_trainer import ECOTrainer

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
DATA_DIR = './data/dg_pair'

if __name__ == '__main__':
    args = get_parse_args()
    args.batch_size = 1
    train_dataset = '/home/zyshi21/studio/DeepMap_Dataset/npz/test.npz'
    num_epochs = args.num_epochs
    
    print('[INFO] Parse Dataset')
    dataset = deepcell.NpzParser_Pair(DATA_DIR, train_dataset)
    train_dataset, val_dataset = dataset.get_dataset()
    print('[INFO] Create Model and Trainer')
    model = TopModel(
        args, 
        dc_ckpt='./ckpt/dc.pth', 
        dg_ckpt='./ckpt/dg_1113.pth'
    )
    
    trainer = ECOTrainer(args, model, distributed=args.distributed, device='cpu')
    if args.resume:
        trainer.resume()
    trainer.set_training_args(lr=1e-4, lr_step=50)
    print('[INFO] Stage 1 Training ...')
    trainer.train(num_epochs, train_dataset, val_dataset)
    
    