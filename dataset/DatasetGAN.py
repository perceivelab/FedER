from gan.GAN import GAN
from torch.utils.data import Dataset

class DatasetGAN(Dataset):
    def __init__(self, gan_paths, device, transforms, num_imgs = 1000):
        self.gan_paths = gan_paths
        self.transforms = transforms
        self.num_imgs = num_imgs
        self.gans = [GAN(ckpt,device) for ckpt in self.gan_paths]

    def __len__(self):
        return self.num_imgs

    def __getitem__(self, idx, z_c_subset = None):
        gan_idx = idx % len(self.gans)
        img, lbl = self.gans[gan_idx].get_imgs(1, z_c_subset)
        #apply transformations
        image = self.transforms(img[0])
        return image, lbl[0]
