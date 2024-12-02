from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import os
import random
from torch import nn
from torch.nn import LSTM, GRU
from .utils.dag_utils import subgraph, custom_backward_subgraph
from .utils.utils import generate_hs_init

from .arch.mlp import MLP
from .arch.mlp_aggr import MlpAggr
from .arch.tfmlp import TFMlpAggr
from .arch.gcn_conv import AggConv

# from .dc_model import Model as DeepCell
from .dc_eco_model import Model as DeepCell
from .dg_model import Model as DeepGate

class TopModel(nn.Module):
    def __init__(self, 
                 args, 
                 dc_ckpt, 
                 dg_ckpt
                ):
        super(TopModel, self).__init__()
        self.args = args
        self.mask_ratio = args.mask_ratio
        
        # DeepCell 
        self.deepcell = DeepCell(dim_hidden=args.dim_hidden)
        self.deepcell.load(dc_ckpt)
        
        # DeepGate 
        self.deepgate = DeepGate(dim_hidden=args.dim_hidden)
        self.deepgate.load(dg_ckpt)
        
        # Transformer
        tf_layer = nn.TransformerEncoderLayer(d_model=args.dim_hidden * 2, nhead=args.tf_head, batch_first=True)
        self.mask_tf = nn.TransformerEncoder(tf_layer, num_layers=args.tf_layer)
        
        # Token masking
        self.mask_token = nn.Parameter(torch.randn(1, args.dim_hidden))  # learnable mask token
        
        # Finetune: Drive 
        self.drive_mlp = MLP(
            args.dim_hidden * 4, args.dim_hidden * 2, 1, 
            num_layer=4, p_drop=0.2, norm_layer='batchnorm', act_layer='relu', 
            sigmoid=True
        )
        
    def mask_single_nodes(self, G, k_hops=[4, 10]): 
        """
        Randomly mask a ratio of tokens and extract its k_hop
        Args:
            G: Input graph
            tokens: Input tokens (batch_size, seq_len, dim_hidden)
            k_hops: Range of hops to extract
        Returns:
            masked_tokens: Tokens with some positions replaced by mask token
            mask_indices: Indices of masked tokens
        """
        seq_len = len(G.x)
        k_hop = 0
        cand_nodes = []
        for i in range(seq_len):
            if G.forward_level[i] >= k_hops[1]:
                cand_nodes.append(i)
        if len(cand_nodes) == 0:
            cand_nodes = list(range(seq_len))
        idx = random.choice(cand_nodes)
        k_hop = min(random.randint(k_hops[0], k_hops[1]), G.forward_level[idx])
        mask_flag = torch.zeros(seq_len, dtype=torch.bool)
        mask_flag[idx] = True
        device = next(self.parameters()).device
        
        # Extract k-hop subgraph
        current_nodes = torch.tensor([idx]).to(device)
        mask_indices = torch.tensor([idx]).to(device)
        for hop in range(k_hop):
            fanin_nodes, _ = subgraph(current_nodes, G.edge_index, dim=1)
            fanin_nodes = torch.unique(fanin_nodes[0])
            current_nodes = fanin_nodes
            mask_flag[fanin_nodes] = True
            mask_indices = torch.cat([mask_indices.to(device), fanin_nodes.to(device)])
        mask_indices = torch.unique(mask_indices)
        
        # Driving Cone 
        boundary_nodes, _ = subgraph(current_nodes, G.edge_index, dim=1)
        drive_cone = boundary_nodes[0].clone()
        while current_nodes.shape[0] > 0:
            fanin_nodes, _ = subgraph(current_nodes, G.edge_index, dim=1)
            fanin_nodes = torch.unique(fanin_nodes[0])
            drive_cone = torch.cat([drive_cone, fanin_nodes])
            current_nodes = fanin_nodes
        drive_cone = torch.unique(drive_cone)
        drive_map = torch.zeros(seq_len, dtype=torch.bool)
        drive_map[drive_cone] = 1
        drive_map = drive_map.bool().to(device)
        
        return idx, mask_indices, drive_map 
    
    def forward(self, G):
        self.device = next(self.parameters()).device
        assert self.args.batch_size == 1, "Batch size must be 1 for ECO"
        mask_node, mask_indices, gt_pm_drive = self.mask_single_nodes(G)
        m_masked = torch.ones(len(G.x)).to(self.device)
        m_masked[mask_indices] = 0
        
        # Get PM and AIG tokens
        pm_hs, pm_hf = self.deepcell(G, mcm_mask=m_masked, mcm_mask_token_hf=self.mask_token)
        nomask_pm_hs, nomask_pm_hf = self.deepcell(G)
        aig_hs, aig_hf = self.deepgate(G)
        aig_hs = aig_hs.detach()
        aig_hf = aig_hf.detach()
        nomask_pm_tokens = torch.cat([nomask_pm_hs, nomask_pm_hf], dim=1)
        pm_tokens = torch.cat([pm_hs, pm_hf], dim=1)
        aig_tokens = torch.cat([aig_hs, aig_hf], dim=1)

        # Reconstruction: Mask Circuit Modeling 
        all_tokens = torch.cat([pm_tokens, aig_tokens], dim=0)
        predicted_tokens = self.mask_tf(all_tokens)
        pred_pm_tokens = predicted_tokens[:pm_tokens.shape[0], :]
        
        # Predict PM probability
        pm_prob = self.deepcell.pred_prob(pred_pm_tokens[:, :self.args.dim_hidden] + pm_hf)
        
        # Predict PM driving 
        drive_input = torch.cat([pred_pm_tokens, pred_pm_tokens[mask_node].repeat(pred_pm_tokens.shape[0], 1)], dim=1)
        pred_pm_drive = self.drive_mlp(drive_input)
        
        return pred_pm_tokens, mask_indices, nomask_pm_tokens, pm_prob, pred_pm_drive, gt_pm_drive
    
   
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
        print('Model loaded from {}'.format(model_path))
        