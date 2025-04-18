import os
import argparse

def get_parse_args():
    parser = argparse.ArgumentParser(description='Pytorch training script of DeepGate.')

    # basic experiment setting
    parser.add_argument('--exp_id', default='train')
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--load_model', default='',
                             help='path to pretrained model')
    parser.add_argument('--resume', action='store_true',
                             help='resume an experiment. '
                                  'Reloaded the optimizer parameter and '
                                  'set load_model to model_last.pth '
                                  'in the exp dir if load_model is empty.')

    # experiment 
    parser.add_argument('--disable_encode', action='store_true', default=False)
    
    # GNN
    parser.add_argument('--gnn', default='dg', choices=['dg', 'gat', 'gcn', 'polargate'],)

    # system
    parser.add_argument('--gpus', default='-1', 
                             help='-1 for CPU, use comma for multiple gpus')
    parser.add_argument("--local-rank", default=0, type=int)
    parser.add_argument('--num_workers', type=int, default=4,
                             help='dataloader threads. 0 for single-thread.')
    parser.add_argument('--not_cuda_benchmark', action='store_true',
                             help='disable when the input size is not fixed.')
    parser.add_argument('--random-seed', type=int, default=208, 
                             help='random seed')

    # log
    parser.add_argument('--print_iter', type=int, default=0, 
                             help='disable progress bar and print to screen.')
    parser.add_argument('--hide_data_time', action='store_true',
                             help='not display time during training.')
    parser.add_argument('--save_all', action='store_true',
                             help='save model to disk every 5 epochs.')
    parser.add_argument('--save_intervals', type=int, default=5,
                             help='number of epochs to run validation.')
    parser.add_argument('--metric', default='loss', 
                             help='main metric to save best model')

    # dataset settings
    parser.add_argument('--data_dir', default='./data/train',
                             type=str, help='the path to the dataset')
    parser.add_argument('--max_token_size', default=4096, type=int,
                             help='max token size for each circuit pair')

    # train and val
    parser.add_argument('--lr', type=float, default=1.0e-4, 
                             help='learning rate for batch size 32.')
    parser.add_argument('--weight_decay', type=float, default=1e-10, 
                             help='weight decay (default: 1e-10)')
    parser.add_argument('--lr_step', type=str, default='40',
                             help='drop learning rate by 10.')
    parser.add_argument('--num_epochs', type=int, default=40,
                             help='total training epochs.')
    parser.add_argument('--batch_size', type=int, default=4,
                             help='batch size')
    parser.add_argument('--trainval_split', default=0.9, type=float,
                             help='the splitting setting for training dataset and validation dataset.')
    
    # Model 
    parser.add_argument('--dim_hidden', type=int, default=128)
    parser.add_argument('--tf_head', type=int, default=8)
    parser.add_argument('--tf_layer', type=int, default=4)
    parser.add_argument('--mask_ratio', type=float, default=0.15)
    parser.add_argument('--k_hop', type=int, default=4)
    parser.add_argument('--linformer', action='store_true', default=False,
                             help='use linformer instead of transformer')
    parser.add_argument('--mask_aig', action='store_true', default=False,
                                help='mask aig instead of nodes')
    
    args = parser.parse_args()

    args.gpus_str = args.gpus
    args.gpus = [int(gpu) for gpu in args.gpus.split(',')]
    args.lr_step = [int(i) for i in args.lr_step.split(',')]
    
    if len(args.gpus) == 1:
        args.distributed = False
        if args.gpus[0] == -1:
            args.device = 'cpu'
        else:
            args.device = 'cuda:{}'.format(args.gpus[0])
    else:
        args.distributed = True
        args.device = 'distributed'
        


    # update data settings
    args.gate_to_index = {'PI': 0, 'AND': 1, 'NOT': 2}
    args.num_gate_types = len(args.gate_to_index)
    args.dim_node_feature = len(args.gate_to_index)

    if args.debug > 0:
        args.num_workers = 0
        # args.batch_size = 1
        args.gpus = [args.gpus[0]]

    # dir
    args.root_dir = os.path.join(os.path.dirname(__file__), '.')
    args.exp_dir = os.path.join(args.root_dir, 'exp')
    args.save_dir = os.path.join(args.exp_dir, args.exp_id)
    args.debug_dir = os.path.join(args.save_dir, 'debug')
    print('The output will be saved to ', args.save_dir)

    if args.resume and args.load_model == '':
        model_path = args.save_dir
        args.load_model = os.path.join(model_path, 'model_last.pth')
    elif args.load_model != '':
        model_path = args.save_dir
        args.load_model = os.path.join(model_path, args.load_model)

    args.local_rank = 0


    return args

def update_dir(args, exp_id):
    # dir
    args.root_dir = os.path.join(os.path.dirname(__file__), '..')
    args.exp_dir = os.path.join(args.root_dir, 'exp', args.task)
    args.save_dir = os.path.join(args.exp_dir, args.exp_id)
    args.debug_dir = os.path.join(args.save_dir, 'debug')
    print('The output will be saved to ', args.save_dir)

    if args.resume and args.load_model == '':
        model_path = args.save_dir
        args.load_model = os.path.join(model_path, 'model_last.pth')
    elif args.load_model != '':
        model_path = args.save_dir
        args.load_model = os.path.join(model_path, args.load_model)
    
    return args