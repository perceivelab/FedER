from abc import abstractmethod
from torchvision import transforms 


class Buffer():
    NAME = None
    def __init__(self, buffer_size:int, transforms: transforms) -> None:
        self.buffer_size = buffer_size
        self.transforms = transforms
    
    @abstractmethod
    def get_data(self, size):
        """
        Random samples a batch of size items
        :param size: the number of requested items
        """
        pass
