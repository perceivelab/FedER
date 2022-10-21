# coding=utf-8
from __future__ import absolute_import, division, print_function

import logging
import argparse
import os
import json
import sys

from datetime import timedelta, datetime

import torch
import time

from train import train

from models.modeling import CNNClassifier
from utils.data_utils import get_loss_weights_train, get_buf_transforms
from utils.utils import count_parameters, set_seed
import matplotlib

from buffer import utilities

matplotlib.use('Agg')

logger = logging.getLogger(__name__)

def setup(args, node_z_c):
    # Prepare model and buffer
    
    if args.dataset in ['Tuberculosis']:
        args.in_channels = 1
        args.num_classes = 2

    elif args.dataset in ['SkinLesion']:
        args.in_channels = 3
        args.num_classes = 2
    
    model = CNNClassifier(args.img_size, in_channels=args.in_channels, num_classes=args.num_classes, model_type = args.model_type, pretrained = args.pretrained)
    model.float().to(args.device)
    if args.cl_pretrain_path!="" and os.path.exists(args.cl_pretrain_path):
        print(f"Load model: {model.load_state_dict(torch.load(args.cl_pretrain_path, map_location=args.device))}")

    if args.use_buffer:
        transforms = get_buf_transforms(args)
        buffer = utilities.create_buffer(args.buffer_type, args.buffer_batch_size , transforms, args.buffer_path, args.keys, buffer_size = args.buffer_size, node_z_c = node_z_c)
    else:
        buffer = None
        
    num_params = count_parameters(model)
    
    logger.info("Training parameters %s", args)
    logger.info("Total Parameter: \t%2.1fM" % num_params)
    print(num_params)

    return args, model, buffer

def federatedNode():
    parser = argparse.ArgumentParser()


    with open('config_file/bufferHalfBatchSize/Task0Node0.json') as f:
        args_dict = json.load(f)
    
    for k, v in args_dict.items():
        parser.add_argument('--' + k, default=v)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.cuda_id}" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    
    args.loss_weights = None
    if args.weighted_loss:
        args.loss_weights = get_loss_weights_train(args.split_path).to(args.device)
    
    timestamp_str = datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H-%M-%S')
    args.name = timestamp_str + args.name
    
    folders_output = args.output_dir.split('/')
    args.output_dir = ''
    for s in folders_output:
        args.output_dir = os.path.join(args.output_dir, s)
    
    if args.eval_auc:
        args.output_dir = os.path.join(args.output_dir, args.model_type, args.name)
    else: 
        args.output_dir = os.path.join(args.output_dir, args.model_type, args.name)
    
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    logger.warning("Device: %s, n_gpu: %s" %
                   (args.device, args.n_gpu))

    # Set seed
    
    set_seed(args)
    if args.dataset in ['Tuberculosis', 'SkinLesion']:
        args.keys = ('image','label')
    
    dict_args = vars(args).copy()
    dict_args['device'] = str(dict_args['device'])
    dict_args['loss_weights'] = dict_args['loss_weights'].tolist() if args.loss_weights is not None else None
    # Saving training info
    
    # Model & Tokenizer Setup
    args, model, buffer = setup(args)

    return model, buffer, args
    
    info = {
        'model_name': model.__class__.__name__,
        'KEYS': args.keys,
        'model_args': dict_args,
        'cmd': str(sys.argv)
        }
    
    with open(os.path.join(args.output_dir, args.name+'.json'), 'w') as fp:
        json.dump(info, fp)

    l = str(sys.argv)
    with open(os.path.join(args.output_dir, 'cmd.json'), 'w') as fp:
        json.dump(l, fp)

    # Training
    train(args, logger, model, buffer, args.keys, info)
        
if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    federatedNode()