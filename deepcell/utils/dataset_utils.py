from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch_geometric.data import Data
from .data_utils import construct_node_feature
from .dag_utils import return_order_info
        
class OrderedData(Data):
    def __init__(self, edge_index=None, x=None, y=None, \
                 tt_pair_index=None, tt_dis=None, \
                 forward_level=None, forward_index=None, backward_level=None, backward_index=None):
        super().__init__()
        self.edge_index = edge_index
        self.tt_pair_index = tt_pair_index
        self.x = x
        self.y = y
        self.tt_dis = tt_dis
        self.forward_level = forward_level
        self.forward_index = forward_index
        self.backward_level = backward_level
        self.backward_index = backward_index
    
    def __inc__(self, key, value, *args, **kwargs):
        if 'index' in key or 'face' in key:
            return self.num_nodes
        if key == 'aig_batch': 
            return 1
        else:
            return 0

    def __cat_dim__(self, key, value, *args, **kwargs):
        if 'forward_index' in key or 'backward_index' in key:
            return 0
        elif 'edge_index' in key:
            return 1
        elif key == 'tt_pair_index' or key == 'connect_pair_index':
            return 1
        else:
            return 0

def parse_pyg_mlpgate(x, edge_index, tt_dis, tt_pair_index, is_pi, \
                        prob, no_edges, connect_label, connect_pair_index, \
                        backward_level, forward_index, forward_level, \
                        no_nodes, backward_index, \
                        no_label = False
                        ):
    
    # x_torch = construct_node_feature(x, num_gate_types)
    # print(x_torch)
    x_torch = torch.LongTensor(x)
    # print(x_torch)

    if no_label:
        tt_pair_index = None
        tt_dis = None
        connect_pair_index = None
        connect_label = None
    else:
        tt_pair_index = torch.tensor(tt_pair_index, dtype=torch.long)
        tt_dis = torch.tensor(tt_dis)
        connect_pair_index = torch.tensor(connect_pair_index, dtype=torch.long).contiguous()
        connect_label = torch.tensor(connect_label)
        if len(connect_pair_index) == 0 or connect_pair_index.shape[1] == 0:
            connect_pair_index = connect_pair_index.reshape((2, 0))

    edge_index = torch.tensor(edge_index, dtype=torch.long).contiguous()
    
    forward_level = torch.tensor(forward_level)
    backward_level = torch.tensor(backward_level)

    forward_index = torch.tensor(forward_index)
    backward_index = torch.tensor(backward_index)

    graph = OrderedData(x=x_torch, edge_index=edge_index, y=None,
                        tt_pair_index=tt_pair_index, tt_dis=tt_dis, 
                        forward_level=forward_level, forward_index=forward_index, 
                        backward_level=backward_level, backward_index=backward_index)
    graph.use_edge_attr = False
    
    if not no_label:
        graph.connect_label = connect_label
        graph.connect_pair_index = connect_pair_index

    graph.gate = torch.tensor(x[:, 1:2], dtype=torch.float)
    graph.prob = torch.tensor(prob).reshape((len(x)))

    return graph

