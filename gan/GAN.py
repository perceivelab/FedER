import torch
import pickle
import torch.nn.functional as F

class GAN():

    def __init__(self, ckpt:str, device:torch.device):
        self.device = device
        self.name = 'gan'
        with open(ckpt, 'rb') as f:
            self.G = pickle.load(f)['G_ema'].to(self.device)
    

    def get_imgs(self,batch_size:int,z_c=None):
        assert batch_size > 0
        with torch.no_grad():
            if z_c is None:
                z = torch.randn([batch_size, self.G.z_dim],device=self.device)
                labels = torch.randint(0,2,(batch_size,),device=self.device)
                c = F.one_hot(labels,num_classes=2).to(self.device)
            else:
                z = z_c[0].to(self.device)
                c = z_c[1].to(self.device)
                labels = torch.argmax(c,dim=1)
            img = self.G(z, c)
            # LE IMG SONO IN RANGE -1 1.
        return img,labels