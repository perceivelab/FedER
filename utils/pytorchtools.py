import numpy as np
import torch

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_metrics_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, model, val_loss=None, val_accuracy = None, val_auc = None):
        
        if not val_loss is None:
            score = -val_loss
        elif not val_accuracy is None:
            score = val_accuracy
        elif not val_auc is None:
            score = val_auc
        else:
            raise ValueError('All val_loss, val_accuracy and val_auc must not be None.')
            
        metric = score if val_loss is None else -score

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(metric, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(metric, model)
            self.counter = 0

    def save_checkpoint(self, metric, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation metric decreased ({self.val_metrics_min:.6f} --> {metric:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        
        self.val_metrics_min = metric
