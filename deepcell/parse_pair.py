from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Optional, Callable, List
import os.path as osp

import numpy as np 
import torch
import shutil
import os
import copy
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader

from .utils.data_utils import read_npz_file
from .utils.aiger_utils import aig_to_xdata
from .utils.circuit_utils import get_fanin_fanout, read_file, add_node_index, feature_gen_connect
from .utils.dataset_utils import parse_pyg_mlpgate

class NpzParser_Pair():
    '''
        Parse the npz file into an inmemory torch_geometric.data.Data object
    '''
    def __init__(self, args, data_dir, aig_path="", pm_path_list=[], \
                 random_shuffle=True, trainval_split=0.9, random_sample=1.0): 
        self.args = args
        self.data_dir = data_dir
        dataset = self.inmemory_dataset(args, data_dir, aig_path, pm_path_list, debug=args.debug)
        if random_shuffle:
            perm = torch.randperm(len(dataset))
            dataset = dataset[perm]
        max_length = int(len(dataset) * random_sample)
        training_cutoff = int(max_length * trainval_split)
        self.train_dataset = dataset[:training_cutoff]
        self.val_dataset = dataset[training_cutoff:max_length]
        # self.train_dataset = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers)
        # self.val_dataset = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        
    def get_dataset(self):
        return self.train_dataset, self.val_dataset
    
    class inmemory_dataset(InMemoryDataset):
        def __init__(self, args, root, aig_path, pm_path_list, debug=False, transform=None, pre_transform=None, pre_filter=None):
            self.name = 'inmemory'
            self.root = root
            self.debug = debug
            self.aig_path = aig_path
            self.pm_path_list = pm_path_list
            self.max_size = args.max_token_size
            super().__init__(root, transform, pre_transform, pre_filter)
            self.data, self.slices = torch.load(self.processed_paths[0])
        
        
        @property
        def raw_dir(self):
            return self.root

        @property
        def processed_dir(self):
            name = 'inmemory'
            if self.max_size > 0:
                name += '_max_{}'.format(self.max_size)
            if self.debug:
                name += '_debug'
            print('[INFO] Processed dataset name: ', name)
            return osp.join(self.root, name)

        @property
        def raw_file_names(self) -> List[str]:
            return [self.aig_path]
        
        @property
        def processed_file_names(self) -> str:
            return ['data.pt']

        def download(self):
            pass
        
        def process(self):
            data_list = []
            tot_pairs = 0
            assert os.path.exists(self.aig_path)
            print('[INFO] Parse AIG circuits')
            aigs = read_npz_file(self.aig_path)['circuits'].item()
            
            for pm_path in self.pm_path_list:
                assert os.path.exists(pm_path)
                lib = pm_path.split('/')[-1].replace('.npz', '').replace('iccad_', '')
                print('[INFO] Parse PM circuits in stdlib: ', lib)
                pms = read_npz_file(pm_path)['circuits'].item()
            
                for pm_idx, pm_name in enumerate(pms):
                    print('Parse circuit: {}, {:} / {:} = {:.2f}%'.format(pm_name, pm_idx+1, len(pms), (pm_idx+1) / len(pms) * 100))
                    pm = pms[pm_name]
                    aig_name = pm_name + '.aig'
                    if aig_name not in aigs:
                        print('AIG circuit not found: ', pm_name)
                        continue
                    aig = aigs[aig_name]
                    
                    if self.max_size > 0 and len(aig['x']) + len(pm['x']) > self.max_size:
                        print(f'Skipping {pm_name} due to size limit')
                        continue
                                        
                    x = pm["x"]
                    edge_index = pm["edge_index"]
                    is_pi = pm["is_pi"]
                    no_edges = pm["no_edges"]
                    no_nodes = pm["no_nodes"]
                    prob = pm["prob"]
                    backward_level = pm["backward_level"]
                    forward_index = pm["forward_index"]
                    forward_level = pm["forward_level"]
                    backward_index = pm["backward_index"]
                    tt_dis = pm['tt_dis']
                    tt_pair_index = pm['tt_pair_index']
                    connect_label = pm['connect_label']
                    connect_pair_index = pm['connect_pair_index']

                    graph = parse_pyg_mlpgate(
                        x, edge_index, tt_dis, tt_pair_index, is_pi,
                        prob, no_edges, connect_label, connect_pair_index,
                        backward_level, forward_index, forward_level,
                        no_nodes, backward_index, 
                        no_label=False
                    )
                    
                    graph.aig_x = torch.tensor(aig["x"])
                    graph.aig_edge_index = torch.tensor(aig["edge_index"], dtype=torch.long).contiguous()
                    graph.aig_prob = torch.tensor(aig["prob"])
                    graph.aig_forward_index = torch.tensor(aig["forward_index"])
                    graph.aig_forward_level = torch.tensor(aig["forward_level"])
                    graph.aig_backward_index = torch.tensor(aig["backward_index"])
                    graph.aig_backward_level = torch.tensor(aig["backward_level"])
                    graph.aig_batch = torch.zeros(len(graph.aig_x), dtype=torch.long)
                    graph.aig_gate = torch.zeros((len(graph.aig_x), 1), dtype=torch.float)
                    graph.aig_tt = torch.tensor(aig["tt_dis"])
                    graph.aig_tt_pair_index = torch.tensor(aig["tt_pair_index"])
                    for idx in range(len(aig["x"])):
                        if aig["x"][idx][1] == 1:
                            graph.aig_gate[idx] = 1
                        elif aig["x"][idx][2] == 1:
                            graph.aig_gate[idx] = 2
                    aig_connect_label = aig['connect_label']
                    aig_connect_pair_index = aig['connect_pair_index']
                    if len(aig_connect_pair_index) == 0 or aig_connect_pair_index.shape[1] == 0:
                        aig_connect_pair_index = aig_connect_pair_index.reshape((2, 0))
                    # Random sample 2*len(aig_x)
                    if aig_connect_pair_index.shape[1] > len(aig['x']) * 2:
                        perm = torch.randperm(len(aig_connect_pair_index))
                        aig_connect_pair_index = aig_connect_pair_index[:, perm[:len(aig['x']) * 2]]
                        aig_connect_label = aig_connect_label[perm[:len(aig['x']) * 2]]
                    
                    graph.aig_connect_label = aig_connect_label
                    graph.aig_connect_pair_index = aig_connect_pair_index
                    
                    graph.name = '{}_{}'.format(lib, pm_name)
                    data_list.append(graph)
                    
                    if self.debug and len(data_list) > 100:
                        break
                if self.debug and len(data_list) > 100:
                    break
                
            data, slices = self.collate(data_list)
            torch.save((data, slices), self.processed_paths[0])
            print('[INFO] Inmemory dataset save: ', self.processed_paths[0])
            print('Total Circuits: {:} Total pairs: {:}'.format(len(data_list), tot_pairs))

        def __repr__(self) -> str:
            return f'{self.name}({len(self)})'

class AigParser():
    def __init__(self):
        pass
    
    def read_aiger(self, aig_path):
        circuit_name = os.path.basename(aig_path).split('.')[0]
        # tmp_aag_path = os.path.join(self.tmp_dir, '{}.aag'.format(circuit_name))
        x_data, edge_index = aig_to_xdata(aig_path)
        # os.remove(tmp_aag_path)
        # Construct graph object 
        x_data = np.array(x_data)
        edge_index = np.array(edge_index)
        tt_dis = []
        tt_pair_index = []
        prob = [0] * len(x_data)
        rc_pair_index = []
        is_rc = []
        graph = parse_pyg_mlpgate(
            x_data, edge_index, tt_dis, tt_pair_index, prob, rc_pair_index, is_rc
        )
        graph.name = circuit_name
        graph.PIs = graph.forward_index[(graph['forward_level'] == 0) & (graph['backward_level'] != 0)]
        graph.POs = graph.backward_index[(graph['backward_level'] == 0) & (graph['forward_level'] != 0)]
        graph.no_connect = graph.forward_index[(graph['forward_level'] == 0) & (graph['backward_level'] == 0)]
        
        return graph        
        
class BenchParser():
    def __init__(self, gate_to_index={'PI': 0, 'AND': 1, 'NOT': 2}):
        self.gate_to_index = gate_to_index
        pass
    
    def read_bench(self, bench_path):
        circuit_name = os.path.basename(bench_path).split('.')[0]
        x_data = read_file(bench_path)
        x_data, num_nodes, _ = add_node_index(x_data)
        x_data, edge_index = feature_gen_connect(x_data, self.gate_to_index)
        for idx in range(len(x_data)):
            x_data[idx] = [idx, int(x_data[idx][1])]
        # os.remove(tmp_aag_path)
        # Construct graph object 
        x_data = np.array(x_data)
        edge_index = np.array(edge_index)
        tt_dis = []
        tt_pair_index = []
        prob = [0] * len(x_data)
        rc_pair_index = []
        is_rc = []
        graph = parse_pyg_mlpgate(
            x_data, edge_index, tt_dis, tt_pair_index, prob, rc_pair_index, is_rc
        )
        graph.name = circuit_name
        graph.PIs = graph.forward_index[(graph['forward_level'] == 0) & (graph['backward_level'] != 0)]
        graph.POs = graph.backward_index[(graph['backward_level'] == 0) & (graph['forward_level'] != 0)]
        graph.no_connect = graph.forward_index[(graph['forward_level'] == 0) & (graph['backward_level'] == 0)]
        
        return graph       
