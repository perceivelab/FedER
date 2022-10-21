import sys
import random
import json
import math
import platform
from monai.data import CacheDataset

class ImagesBufferDataset(CacheDataset):
    def __init__(self, split_path, section, transforms, buffer_size = None,val_frac = 0, seed = 100, cache_num = sys.maxsize, cache_rate=1.0, num_workers=0):    
        #if execute test is False, training and test split are used both fro traning. 
        self.section = section
        self.text_labels = ['negative', 'positive']
        self.seed = seed
        self.val_frac = val_frac
        self.buffer_size = buffer_size
        
        data = self._generate_data_list(split_path)
        super().__init__(data, transforms, cache_num=cache_num, cache_rate=cache_rate, num_workers=num_workers)
        
     
    #split data in train, val and test sets in a reproducible way
    def _generate_data_list(self, split_path):
        with open(split_path) as fp:
           path=json.load(fp)
        data = list()
        
        if self.section == 'test':
            data = path['test']
        elif self.section == 'train':
            datalist_pos = path['train']['pos']
            datalist_neg = path['train']['neg']
            random.seed(self.seed)
            random.shuffle(datalist_pos) # shuffles the ordering of datalist (deterministic given the chosen seed)
            random.shuffle(datalist_neg) # shuffles the ordering of datalist (deterministic given the chosen seed)
            data = datalist_pos+ datalist_neg
            if self.buffer_size is not None:
                random.shuffle(data)
                data = data[:self.buffer_size]
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
