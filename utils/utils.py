import os
import torch
import json
import random
import numpy as np

from typing import Any

from sklearn.metrics import balanced_accuracy_score, confusion_matrix, accuracy_score, hamming_loss
from sklearn.metrics import f1_score, jaccard_score, precision_score, recall_score

'''
from monai.optimizers import LearningRateFinder

def estimate_optim_lr(model, optimizer, loss, device, lower_lr, upper_lr, train_loader, val_loader, image_key, label_key):
    lr_finder = LearningRateFinder(model, optimizer, loss, device=device)
    lr_finder.range_test(train_loader, val_loader, end_lr=upper_lr, num_iter=20, image_extractor = lambda x: x[image_key], label_extractor = lambda x: x[label_key])
    steepest_lr, _ = lr_finder.get_steepest_gradient()
    #ax = plt.subplots(1, 1, figsize=(15, 15), facecolor="white")[1]
    #_ = lr_finder.plot(ax=ax)
    return steepest_lr
'''

def compute_metrics(labels, preds, multi_label = False):
    if not multi_label:
        metrics = {
            'precision' : precision_score(labels, preds, average='macro', zero_division = 0),
            'recall' : recall_score(labels, preds, average='macro', zero_division = 0),
            'f1_score' : f1_score(labels, preds, average='macro', zero_division = 0),
            'jaccard_score': jaccard_score(labels, preds, average='macro', zero_division = 0)
            }
    else:
        metrics = {
            'precision_macro' : precision_score(labels, preds, average='macro', zero_division = 0),
            'precision_micro' : precision_score(labels, preds, average='micro', zero_division = 0),
            'recall_macro' : recall_score(labels, preds, average='macro', zero_division = 0),
            'recall_micro' : recall_score(labels, preds, average='micro', zero_division = 0),
            'f1_score_macro' : f1_score(labels, preds, average='macro', zero_division = 0),
            'f1_score_micro' : f1_score(labels, preds, average='micro', zero_division = 0),
            'jaccard_score_macro': jaccard_score(labels, preds, average='macro', zero_division = 0),
            'jaccard_score_micro': jaccard_score(labels, preds, average='micro', zero_division = 0)
            }
    return metrics

def get_accuracy(preds, labels, accuracy_type='simple'):
    if len(labels.shape)>1: # if we have more than one dimension, we are doing multilabel classification
        if accuracy_type == 'simple':
            return {'accuracy_simple':accuracy_score(labels, preds)}
        else:
            acc = accuracy_score(labels, preds)
            hamm = hamming_loss(labels, preds)
            return {'accuracy_simple': acc, 'hamming_loss':hamm}
    if accuracy_type == 'simple':
        return {'accuracy_simple':(preds == labels).mean()}
    elif accuracy_type == 'balanced':
        return {'accuracy_balanced':balanced_accuracy_score(labels, preds)}
    elif accuracy_type == 'both':
        simple_accuracy = (preds == labels).mean()
        balanced_accuracy = balanced_accuracy_score(labels, preds)
        return {'accuracy_simple':simple_accuracy, 'accuracy_balanced':balanced_accuracy}

def save_model(args, logger, model, suffix='', global_step = None, val_accuracy_dict = None, test_accuracy_dict = None, val_roc_metrics_dict = None, test_roc_metrics_dict = None):
    model_to_save = model.module if hasattr(model, 'module') else model
    model_checkpoint = os.path.join(args.output_dir, f"{args.name}_{suffix}.bin")
    torch.save(model_to_save.state_dict(), model_checkpoint)
    logger.info(f"Saved model {suffix} to [DIR: {args.output_dir}]")
    
    info = {
        'global_step': global_step
        }
    
    if not val_accuracy_dict is None:
        for k in val_accuracy_dict.keys():
            info[f'val_{k}'] = val_accuracy_dict[k]
        
    if not test_accuracy_dict is None:
        for k in test_accuracy_dict.keys():
            info[f'test_{k}'] = test_accuracy_dict[k]
            
    if not val_roc_metrics_dict is None:
        for k in val_roc_metrics_dict.keys():
            info[f'val_roc_{k}'] = val_roc_metrics_dict[k]
            
    if not test_roc_metrics_dict is None:
        for i in range(len(test_accuracy_dict)):
            for k in test_roc_metrics_dict[i].keys():
                info[f'test_roc_{k}_node_{i}'] = test_roc_metrics_dict[i][k]
    
    with open(os.path.join(args.output_dir, f'{suffix}.json'), 'w') as fp:
        json.dump(info, fp)
        
    with open(os.path.join(args.output_dir, f'log_{suffix}.json'), 'a') as f:
        json.dump(info, f, indent = 0)
        

def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params/1000000


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

