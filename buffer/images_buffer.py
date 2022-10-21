from sklearn.utils import shuffle
import torch
from buffer.buffer import Buffer
from monai import transforms
from dataset.infiniteDataLoader import InfiniteDataLoader
from dataset.ImagesBufferDataset import ImagesBufferDataset

class ImagesBuffer(Buffer):
    NAME = 'IMAGES_BUFFER'
    def __init__(self, buffer_size:int, transforms:transforms, buf_path : str, keys : tuple, batch_size:int ):
        super(ImagesBuffer, self).__init__(buffer_size, transforms)
        self.loader = InfiniteDataLoader(ImagesBufferDataset(split_path = buf_path, section = 'train', transforms = transforms, buffer_size = buffer_size), batch_size = batch_size, shuffle = True)
        self.img_key = keys[0]
        self.lbl_key = keys[-1]
    
    def get_data(self, size):
        data = next(self.loader)
        images= data[self.img_key]
        labels = data[self.lbl_key]
        return images, labels
