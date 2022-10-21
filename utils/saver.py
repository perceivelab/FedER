from pathlib import Path
from time import time
from datetime import datetime
import os
import torch
from torch.utils.tensorboard import SummaryWriter
import torchvision
import json
import csv
import wandb


class Saver(object):
    """
    Saver allows for saving and restore networks.
    """

    def __init__(self, output_folder: Path, experiment_name: str, wandb_mode: str):
        timestamp_str = datetime.fromtimestamp(
            time()).strftime('%Y-%m-%d_%H-%M-%S')
        self.path = os.path.join(output_folder, f'{experiment_name+"_"+timestamp_str}')
        print(self.path)
        self.writer = SummaryWriter(str(self.path))
        self.csv_file = os.path.join(self.path,'metrics.csv')
        # Create checkpoint sub-directory
        self.ckpt_path = os.path.join(self.path,'ckpt')
        os.makedirs(self.ckpt_path)
        #wandb
        wandb.init( project="fedER", entity="bomber-team",mode=wandb_mode, resume=True)
        

    def save_model(self, net, name: str, step: int):
        """
        Save model parameters in the checkpoint directory.
        """
        # Get state dict
        state_dict = net.state_dict()
        # Save
        torch.save(state_dict, str(self.ckpt_path) +
                   '/' + f'{name}_{step:05d}.pth')        
    
    def save_data(self,data,name:str):
        ''' Save generic data in experiment folder '''
        torch.save(data, self.path / f'{name}.pth')

    def close(self):
        '''
        Close Tensorboard connection
        '''
        self.writer.close()
        self.csv_file.close()

    def log_loss(self, name: str, value: float, iter_n: int):
        '''
        Log loss to TB and comet_ml and csv
        '''
        self.writer.add_scalar(name, value, iter_n)
        wandb.log({name: value, "STEP": iter_n})
        with open(self.csv_file, 'a', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([name, iter_n, value])


    def log_images(self, title: str, images_vector: torch.Tensor):
        '''
        Log images to WANDB
        '''

        img_grid = torchvision.utils.make_grid(
            images_vector, normalize=True, nrow=10)
        images = wandb.Image(img_grid)
        wandb.log({title: images})  

    
    def log_text(self,title:str,text:str,step:int):
        self.writer.add_text(title,text,step)

    def log_hparams(self,params_dict):
        #self.writer.add_hparams(params_dict)
        
        self.writer.add_text('hparams',str(params_dict),0)
        wandb.config.update(params_dict)
        with open(os.path.join(self.path, 'hparams.json'), 'w') as fp:
            json.dump(params_dict, fp)

    def log_cmd(self, cmd):
        self.writer.add_text(cmd, cmd, 0)

        with open(os.path.join(self.path, 'cmd.json'), 'w') as fp:
            json.dump(cmd, fp)
