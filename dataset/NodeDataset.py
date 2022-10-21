import sys
import random
import json
import math
import platform
from monai.data import CacheDataset
import numpy as np


class NodeDataset(CacheDataset):
    def __init__(self, split_path, section, transforms, val_frac = 0, seed = 100, cache_num = sys.maxsize, cache_rate=1.0, num_workers=0, node_idx = None, n_nodes = None):    
        #if execute test is False, training and test split are used both fro traning. 
        self.section = section
        self.text_labels = ['negative', 'positive']
        self.seed = seed
        self.val_frac = val_frac
        self.node_idx = node_idx
        self.n_nodes = n_nodes
        data = self._generate_data_list(split_path)
        super().__init__(data, transforms, cache_num=cache_num, cache_rate=cache_rate, num_workers=num_workers)
        
     
    #split data in train, val and test sets in a reproducible way
    def _generate_data_list(self, split_path):
        with open(split_path) as fp:
           path=json.load(fp)
        data = list()
        
        if self.section == 'test':
            data = path['test']
        elif self.section == 'train' or self.section == 'val':
            datalist_pos = path['train']['pos']
            datalist_neg = path['train']['neg']
            random.seed(self.seed)
            random.shuffle(datalist_pos) # shuffles the ordering of datalist (deterministic given the chosen seed)
            random.shuffle(datalist_neg) # shuffles the ordering of datalist (deterministic given the chosen seed)
            
            val_items = math.ceil(self.val_frac * (len(datalist_pos)+len(datalist_neg)))
            train_items = len(datalist_pos) + len(datalist_pos) - val_items 
            assert (len(datalist_pos) + len(datalist_pos)) == (train_items + val_items), "Error dataset split"
            if self.section == 'train':
                data = datalist_pos[math.ceil(val_items/2):] + datalist_neg[math.floor(val_items/2):]
                random.shuffle(data)
                if self.node_idx is not None:
                    nodes_datasets = np.array_split(data,self.n_nodes)
                    data = nodes_datasets[self.node_idx]
            elif self.section == 'val':
                data = datalist_pos[:math.ceil(val_items/2)]+ datalist_neg[:math.floor(val_items/2)]
                

        else: 
            raise ValueError(
                    f"Unsupported section: {self.section}, "
                    "available options are ['train', 'val', 'test']."
                )
        
        if platform.system() != 'Windows':
            for sample in data:
                for key in sample.keys():
                    if isinstance(sample[key], str):
                        sample[key] = sample[key].replace('\\', '/')
        return data  
