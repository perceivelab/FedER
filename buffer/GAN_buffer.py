import torch
from buffer.buffer import Buffer
from gan.GAN import GAN
from monai import transforms
import random

class GANBuffer(Buffer):
    NAME = 'GAN_BUFFER'
    def __init__(self, buffer_size:int, transforms:transforms, buf_path:str, node_z_c):
        super(GANBuffer, self).__init__(buffer_size, transforms)
        self.node_z_c = node_z_c
        self.gan = GAN(buf_path,'cuda')
        self.buf_path = buf_path #GAN checkpoint
    
    def get_data(self, size):
        z_c_subset = None
        if self.node_z_c is not None:
            indices = torch.randperm(len(self.node_z_c[0]))[:size]
            z_c_subset = (self.node_z_c[0][indices],self.node_z_c[1][indices])
        
        images, labels = self.gan.get_imgs(size, z_c_subset)
        #apply transformations
        images = torch.stack([self.transforms(img) for img in images])
        return images, labels
  
    