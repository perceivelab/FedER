import logging
import os
import json
import torch

from torch.utils.data import DataLoader
from utils.transforms import Transpose_BWImage
from monai.transforms import (
    LoadImageD, 
    AddChannelD,
    AsChannelFirstD,
    Compose,
    RandFlipD,
    RandRotate90D,
    ScaleIntensityD,
    EnsureTyped, 
    ToDeviceD, ResizeD,
    RandFlip, 
    RandRotate90, 
    EnsureType,
    ToDevice,
    Resize,
    ScaleIntensity,
    LoadImage,
    AddChannel,
    AsChannelFirst)

logger = logging.getLogger(__name__)

def get_loss_weights(split_path):
    with open(split_path) as fp:
        d = json.load(fp)
        lbl = [el['label'] for el in d]
        num = len(lbl)
        pos = sum(lbl)
        neg = num - pos
        p_pos = num/pos
        p_neg = num/neg
        weights = [p_neg/max(p_pos, p_neg), p_pos/max(p_pos, p_neg)]
    return torch.tensor(weights)

def get_buf_transforms(args):
    buf_transforms = []
    if args.buffer_type == 'images':
        buf_transforms = buf_transforms + [LoadImage()]
        if 'Tuberculosis' in args.dataset:
            buf_transforms = buf_transforms + [AddChannel()]
        elif 'SkinLesion' in args.dataset :
            buf_transforms = buf_transforms + [AsChannelFirst()]
        buf_transforms = buf_transforms + [
            ScaleIntensity(),
            Resize(spatial_size=( args.img_size,args.img_size)),
            EnsureType(data_type="tensor")
        ] 
        if not args.data_on_CPU:
            buf_transforms = buf_transforms + [ToDevice(device=args.device)]
        
        buf_transforms = buf_transforms + [RandFlip(prob = 0.5, spatial_axis=0)]
        
        if 'SkinLesion' in args.dataset: 
            buf_transforms = buf_transforms + [RandRotate90(prob=0.5, spatial_axes=(0,1))]
    elif args.buffer_type =='gan':
        buf_transforms = buf_transforms + [
            ScaleIntensity(),
            Resize(spatial_size=(args.img_size,args.img_size)),
            EnsureType(data_type="tensor")
        ] 
        if not args.data_on_CPU:
            buf_transforms = buf_transforms + [ToDevice(device=args.device)]
        
        buf_transforms = buf_transforms + [RandFlip(prob = 0.5, spatial_axis=1)]
        
        if 'SkinLesion' in args.dataset: 
            buf_transforms = buf_transforms + [RandRotate90(prob=0.5, spatial_axes=(0,1))]
    
    return Compose(buf_transforms)

def get_loss_weights_train(split_path):
    with open(split_path) as fp:
        d = json.load(fp)
        pos = len(d['train']['pos'])
        neg = len(d['train']['neg'])
        num = pos+neg
        p_pos = num/pos
        p_neg = num/neg
        weights = [p_neg/max(p_pos, p_neg), p_pos/max(p_pos, p_neg)]
    return torch.tensor(weights)

def get_dataset_transforms(args, KEYS):

    train_list_transf = [LoadImageD(keys = KEYS[:-1])]
    if args.dataset in ['Tuberculosis']:
        train_list_transf = train_list_transf + [Transpose_BWImage(keys = KEYS[:-1]),AddChannelD(keys = KEYS[:-1])]
    elif args.dataset in ['SkinLesion']:
        train_list_transf = train_list_transf + [AsChannelFirstD(keys = KEYS[:-1])]
    train_list_transf = train_list_transf + [ScaleIntensityD(keys = KEYS[:-1]),
                                             ResizeD(keys = KEYS[:-1], spatial_size=(args.img_size,args.img_size)), 
                                             EnsureTyped(keys = KEYS[:-1], data_type="tensor")]
    if not args.data_on_CPU:
        train_list_transf = train_list_transf + [ToDeviceD(keys = KEYS[:-1], device=args.device)]
    train_list_transf = train_list_transf + [RandFlipD(keys = KEYS[:-1], prob = 0.5, spatial_axis=1)]
        
    if args.dataset in ['SkinLesion']: 
        train_list_transf = train_list_transf + [RandRotate90D(keys= KEYS[:-1], prob=0.5, spatial_axes=(0,1))]
    
    val_list_transf = [LoadImageD(keys = KEYS[:-1])]
    if args.dataset in ['Tuberculosis']:
        val_list_transf = val_list_transf + [Transpose_BWImage(keys = KEYS[:-1]), AddChannelD(keys = KEYS[:-1])]
    elif args.dataset in ['SkinLesion']:
        val_list_transf = val_list_transf + [AsChannelFirstD(keys = KEYS[:-1])]
    val_list_transf = val_list_transf + [ScaleIntensityD(keys = KEYS[:-1]), ResizeD(keys = KEYS[:-1], spatial_size=(args.img_size,args.img_size)), EnsureTyped(keys = KEYS[:-1], data_type="tensor")]
    if not args.data_on_CPU:
        val_list_transf = val_list_transf + [ToDeviceD(keys = KEYS[:-1], device=args.device)]
    
    train_transforms = Compose(train_list_transf)
    val_transforms = Compose(val_list_transf)

    return train_transforms , val_transforms

def get_loader(args, KEYS, section= None,node_idx = None, n_nodes = None):
    
    from dataset.NodeDataset import NodeDataset as DS
       
    train_transforms, val_transforms = get_dataset_transforms(args, KEYS)
    
    if section is None:
    
        dataset = {}
        dataset["train"] = DS(split_path = args.split_path, section = 'train', transforms = train_transforms, cache_rate = args.cache_rate, seed = args.seed,node_idx = node_idx, n_nodes = n_nodes)
        dataset["val"] = DS(split_path = args.split_path, section = 'val', transforms = val_transforms, cache_rate = args.cache_rate, seed = args.seed)
        dataset["test"] = DS(split_path = args.split_path, section = 'test', transforms = val_transforms, cache_rate = args.cache_rate, seed = args.seed)
    
    
        train_loader = DataLoader(dataset["train"],
                                  batch_size = args.train_batch_size,
                                  num_workers=args.num_workers,
                                  shuffle=True
                                  )
        val_loader = DataLoader(dataset["val"],
                                  batch_size = args.eval_batch_size,
                                  num_workers=args.num_workers,
                                  )
        test_loader = DataLoader(dataset["test"],
                                      batch_size = args.eval_batch_size,
                                      num_workers=args.num_workers,
                                      )
        
        return train_loader, val_loader, test_loader
    else:
        if section == 'train':
            dataset = DS(split_path = args.split_path, section = 'train', transforms = train_transforms, cache_rate = args.cache_rate, seed = args.seed,node_idx = node_idx, n_nodes = n_nodes)
            shuffle = True
        elif section in ['val']:
            dataset = DS(split_path = args.split_path, section = section, transforms = val_transforms, cache_rate = args.cache_rate, seed = args.seed,node_idx = node_idx, n_nodes = n_nodes)
            shuffle = False
        elif section in ['test']:
            dataset = DS(split_path = args.split_path, section = section, transforms = val_transforms, cache_rate = args.cache_rate, seed = args.seed,node_idx = node_idx, n_nodes = n_nodes)
            shuffle = False

        print(f'Section: {section}, Shuffle: {shuffle}')
        
        loader = DataLoader(dataset,
                                batch_size = args.train_batch_size,
                                num_workers=args.num_workers,
                                shuffle = shuffle 
                                )
        
        return loader

def get_loaderGAN(args, KEYS, gan_ckpt):
    from dataset.DatasetGAN import DatasetGAN as DS

    transforms = get_buf_transforms(args)
    dataset = DS(gan_paths=gan_ckpt, device=args.device, transforms=transforms, num_imgs= args.num_imgs_gan)
    shuffle = False
    loader = DataLoader(dataset,
                        batch_size = args.train_batch_size,
                        num_workers=args.num_workers,
                        shuffle = shuffle 
                        )
    return loader