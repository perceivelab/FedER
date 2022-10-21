from monai import transforms
from buffer.GAN_buffer import GANBuffer
from buffer.images_buffer import ImagesBuffer

def create_buffer(buffer_type:str, batch_size:int, transforms:transforms, buf_path:str = None, keys:tuple = None, buffer_size:int = -1, node_z_c = None):
    if buffer_type == "gan":
        return GANBuffer(buffer_size, transforms, buf_path, node_z_c = node_z_c)
    elif buffer_type == "images":
        return ImagesBuffer(buffer_size, transforms, buf_path, keys, batch_size = batch_size)
    else:
        raise ValueError(f"There is no {buffer_type} here.")
    

