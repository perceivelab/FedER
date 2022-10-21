from federated import setup
import logging
from train import train
from validation import valid 
import click
import copy


class Node():
    def __init__(self,idx,saver,private_dataset_loader,args,node_z_c, share_classifier, share_buffer):
        self.idx = idx
        self.saver = saver
        self.share_classifier = share_classifier
        self.share_buffer = share_buffer
        self.private_dataset_loader = private_dataset_loader
        _, model, buffer = setup(args, node_z_c)
        self.buffer = buffer
        self.model = model
        self.args = args
        self.logger = logging.getLogger(__name__)
        self.train_epochs_counter = 0

        #input received from other node
        self.in_buffer = None
        self.in_model = None
    
    def receive_data(self, model, buffer):
        self.in_buffer = buffer
        self.in_model = model

    def send_data(self):
        if self.share_classifier and self.share_buffer:
            return copy.deepcopy(self.model), copy.deepcopy(self.buffer)
        elif self.share_classifier and not self.share_buffer:
            return copy.deepcopy(self.model), None
        elif not self.share_classifier and self.share_buffer:
            return None, copy.deepcopy(self.buffer)
        else:
            return None, None

    def reset(self):
        if self.share_classifier:
            self.model = self.in_model
        self.in_model = None

    

    def train(self, num_epochs):
        click.echo(f'Fed. Training - Node {self.idx}')
        self.model = train(self.idx, self.args, self.logger, self.model, self.in_buffer,num_epochs, self.args.keys,self.saver,self.private_dataset_loader,self.train_epochs_counter)
        self.train_epochs_counter += num_epochs

    def test(self,test_loader,test_name,round):
        test_accuracy_dict, test_loss_dict, test_roc_metrics_dict, probs, labels = valid(self.args,self.logger,self.model,self.saver,'Test',test_loader,test_name, round,self.args.keys,self.idx)
        return test_accuracy_dict['accuracy_balanced'], probs, labels

    def save_ckpt(self, round):
        self.saver.save_model(self.model, f"node{self.idx}", round)

