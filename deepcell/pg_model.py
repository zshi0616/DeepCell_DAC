from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import os
from torch import nn
from torch.nn import LSTM, GRU
from .utils.dag_utils import subgraph, custom_backward_subgraph
from .utils.utils import generate_hs_init

from .arch.mlp import MLP
from .arch.mlp_aggr import MlpAggr
from .arch.tfmlp import TFMlpAggr
from .arch.gcn_conv import AggConv
from .arch.gat_conv import AGNNConv
from .arch.pg_layer import create_spectral_features, MLP, PolarGateConv, restPolarGateConv
import torch.nn.functional as F


class Model(nn.Module):
    '''
    Recurrent Graph Neural Networks for Circuits.
    '''
    def __init__(self, 
                 num_rounds = 1, 
                 dim_hidden = 128, 
                 enable_encode = True,
                 enable_reverse = False, 
                 lamb=5, 
                 norm_emb = False
                ):
        super(Model, self).__init__()
        
        # configuration
        self.num_rounds = num_rounds
        self.enable_encode = enable_encode
        self.enable_reverse = enable_reverse        # TODO: enable reverse
        self.lamb = lamb

        # dimensions
        self.dim_hidden = dim_hidden
        self.dim_mlp = 32
        
        # edge 
        self.pos_edge_index = None
        self.neg_edge_index = None
        
        # Network
        self.conv1 = PolarGateConv(dim_hidden, dim_hidden // 2, first_aggr=True)
        self.convs = torch.nn.ModuleList()
        self.convs.append(
            restPolarGateConv(dim_hidden // 2, dim_hidden // 2, first_aggr=False,
                            norm_emb=norm_emb))
        self.weight = torch.nn.Linear(self.dim_hidden, self.dim_hidden)
        self.init_feature_ln = nn.Linear(self.dim_hidden // 2, self.dim_hidden)
        self.readout_prob = MLP(self.dim_hidden, self.dim_hidden, 1, num_layer=3, p_drop=0.2, norm_layer='batchnorm',
                                act_layer='relu')

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.weight.reset_parameters()

    def get_x_edge_index(self, init_emb, edge_index_s):
        self.pos_edge_index = edge_index_s[edge_index_s[:, 2] > 0][:, :2].t()
        self.neg_edge_index = edge_index_s[edge_index_s[:, 2] < 0][:, :2].t()
        if init_emb is None:
            init_emb = create_spectral_features(
                pos_edge_index=self.pos_edge_index,
                neg_edge_index=self.neg_edge_index,
                node_num=self.node_num,
                dim=self.in_dim
            ).to(self.device)
        else:
            init_emb = init_emb
        self.x = self.init_feature_ln(init_emb)

    def forward(self, G):
        init_emb = G.x
        edge_index_s = G.edge_index
        self.get_x_edge_index(init_emb, edge_index_s)
        z = torch.tanh(self.conv1(
            self.x, self.pos_edge_index, self.neg_edge_index))
        for conv in self.convs:
            z = torch.tanh(conv(z, self.pos_edge_index, self.neg_edge_index))
        z = torch.tanh(self.weight(z))

        prob = self.readout_prob(z)
        prob = F.sigmoid(prob)

        return z, prob
        
    
    def load(self, model_path):
        checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
        state_dict_ = checkpoint['state_dict']
        state_dict = {}
        for k in state_dict_:
            if k.startswith('module') and not k.startswith('module_list'):
                state_dict[k[7:]] = state_dict_[k]
            else:
                state_dict[k] = state_dict_[k]
        model_state_dict = self.state_dict()
        
        for k in state_dict:
            if k in model_state_dict:
                if state_dict[k].shape != model_state_dict[k].shape:
                    print('Skip loading parameter {}, required shape{}, loaded shape{}.'.format(
                        k, model_state_dict[k].shape, state_dict[k].shape))
                    state_dict[k] = model_state_dict[k]
            else:
                print('Drop parameter {}.'.format(k))
        for k in model_state_dict:
            if not (k in state_dict):
                print('No param {}.'.format(k))
                state_dict[k] = model_state_dict[k]
        self.load_state_dict(state_dict, strict=False)
        
    def load_pretrained(self, pretrained_model_path = ''):
        if pretrained_model_path == '':
            pretrained_model_path = os.path.join(os.path.dirname(__file__), 'pretrained', 'model.pth')
        self.load(pretrained_model_path)
