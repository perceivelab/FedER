
from ast import Raise
import random
import itertools
import torch
from utils.data_utils import get_loss_weights_train
import json
import argparse
import torch.nn.functional as F

def get_epochs_num(epoch_policy, initial_num_epochs, round, n_rounds):
    if epoch_policy == 'constant':
        return initial_num_epochs
    elif epoch_policy == 'dec_linear':
        slot = initial_num_epochs // n_rounds
        epochs = initial_num_epochs - (slot * round)
        return epochs if epochs>1 else 1
    return initial_num_epochs

def pairwise(iterable):
    # pairwise('ABCDEFG') --> AB BC CD DE EF FG
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)

def get_routes(n_nodes):
    idxs = [i for i in range(n_nodes)]
    random.shuffle(idxs)
    routes = [x for x in pairwise(idxs)]
    last_route = (idxs[-1],idxs[0])
    routes.append(last_route)
    return routes


def generate_z(n_z, n_nodes):
    z_nodes = []
    for node in range(n_nodes):
        z = torch.randn([n_z, 512])
        labels = torch.randint(0,2,(n_z,))
        c = F.one_hot(labels,num_classes=2)
        z_nodes.append((z,c))
    return z_nodes

def args_from_json(json_path, n_nodes, n_rounds, experiment_name, outdir, dataset, setting, save_every, share_classifier, share_buffer, epoch_policy, num_epochs, cuda_id, num_imgs_gan, run_idx, seed, wandb_mode, model_type, mu, client_weight, cross_val, fold,cache_rate):
    parser = argparse.ArgumentParser()
    with open(json_path) as f:
        args_dict = json.load(f)
    for k, v in args_dict.items():
        parser.add_argument('--' + k, default=v)
    parser.add_argument('--n_nodes', default=n_nodes)
    parser.add_argument('--n_rounds', default=n_rounds)
    parser.add_argument('--experiment_name', default=experiment_name)
    parser.add_argument('--outdir', default=outdir)
    parser.add_argument('--setting', default=setting)
    parser.add_argument('--save_every', default=save_every)
    parser.add_argument('--share_buffer', default = share_buffer)
    parser.add_argument('--share_classifier', default=share_classifier)
    parser.add_argument('--epoch_policy', default=epoch_policy)
    parser.add_argument('--run_idx',default=run_idx, type=int)
    parser.add_argument('--seed',default=seed, type=int)
    parser.add_argument('--num_imgs_gan', default=num_imgs_gan, type = int)
    parser.add_argument('--wandb_mode', default=wandb_mode)
    parser.add_argument('--mu', default=mu, type = float)
    parser.add_argument('--client_weight', default = client_weight, type= bool)
    parser.add_argument('--cross_val', default = cross_val, type= bool)
    parser.add_argument('--fold', default = fold, type= int)
    parser.add_argument('--cache_rate', default = cache_rate, type= float)
    args = parser.parse_args()
    device = torch.device(f"cuda:{cuda_id}" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    args.loss_weights = None
    if args.weighted_loss:
        if not(cross_val):
            args.loss_weights = get_loss_weights_train(args.split_path).to(args.device)
        else:
            raise Exception('split path not yet defined. Impossible to calculate loss_weights')
    KEYS = ('image','label')
    args.keys = KEYS

    return args