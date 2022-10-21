import torch
import click
import sys

from node import Node
from utils.saver import Saver
from utils.data_utils import get_loader
from utils.fed_utils import get_epochs_num, get_routes, generate_z, args_from_json
from utils.utils import set_seed
import os

@click.command()
@click.option('--n_nodes', default=4, help='Number of nodes in the federation.')
@click.option('--save_every', default=5, help='Number of rounds between each ckpt')
@click.option('--dataset', type=click.Choice(['Tuberculosis', 'SkinLesion']), default="Tuberculosis", help='Dataset.')
@click.option('--epoch_policy',type=click.Choice(['constant', 'dec_linear']), default="constant", help='Setting for training epoch decreasing policy')
@click.option('--num_epochs', type = int, default = 10, help = 'Max number of epochs per round.')
@click.option('--share_classifier', default=True, type =bool, help='Enable/Disable sharing of the classifier between nodes')
@click.option('--share_buffer', default=True, type =bool, help='Enable/Disable sharing of the classifier between nodes')
@click.option('--setting',type=click.Choice(['IID', 'non-IID']), default="non-IID", help='Setting for the experiments (IID vs non-IID)')
@click.option('--n_rounds', type=int, default=10, help='Number of rounds to run the federation.')
@click.option('--model_type', type=str, default=None, help='Model type to use when setting is FedAvg or FedProx or FedBN')
@click.option('--experiment_name',default=None,help='Name of the experiment')
@click.option('--outdir',default='runs',help='Output directory')
@click.option('--learning_rate', default = 1e-6, type = float)
@click.option('--buffer_size', type = int, default = 512, help='Number of images in buffer')
@click.option('--cuda_id', type = int, default = 0)
@click.option('--num_imgs_gan', type = int, default=1000, help='Number of images generated from the GAN. Only used when the setting Central Node and dataset TuberculosisGAN or Tuberculosis_Mixed')
@click.option('--run_idx',default=-1,help='Run index for multiple runs of the same experiment')
@click.option('--seed',default=42,help='Random seed',type=int)
@click.option('--wandb_mode',default='online')
@click.option('--mu', default = 0.0, type = float, help='mu parameter of FedProx')
@click.option('--client_weight', type = bool, default = False, help = 'used only for FedProx, Avg, BN when compute server model\'s weight')
@click.option('--cross_val', type=bool, default = False)
@click.option('--fold', type=int, default = -1)
@click.option('--cache_rate', type=float, default=1.0)

def main(n_nodes, n_rounds, experiment_name, outdir, dataset, setting, save_every, share_classifier, share_buffer, epoch_policy, num_epochs, learning_rate, buffer_size, num_imgs_gan, cuda_id, run_idx, seed, wandb_mode, model_type, mu, client_weight, cross_val, fold, cache_rate):
    
    KEYS = ('image','label')
    set_seed(seed)
    dic_hyperparams ={'n_nodes': n_nodes, 
                    'n_rounds': n_rounds, 
                    'experiment_name': experiment_name, 
                    'outdir': outdir, 
                    'dataset': dataset, 
                    'setting': setting, 
                    'save_every': save_every, 
                    'share_classifier': share_classifier, 
                    'share_buffer': share_buffer,
                    'epoch_policy': epoch_policy, 
                    'num_epochs': num_epochs, 
                    'num_imgs_gan': num_imgs_gan,
                    'run_idx':run_idx,
                    'cuda_id': cuda_id,
                    'seed': seed,
                    'wandb_mode': wandb_mode, 
                    'model_type':model_type, 
                    'mu': mu, 
                    'client_weight':client_weight, 
                    'cross_val': cross_val,
                    'fold': fold, 
                    'cache_rate': cache_rate}

    if setting == 'non-IID':
        if dataset == "Tuberculosis":
            n_nodes = 2
            args_montgomery = args_from_json('config_file/montgomery.json', **dic_hyperparams)
            args_shenzhen= args_from_json('config_file/shenzhen.json', **dic_hyperparams)
            if cross_val:
                args_montgomery.split_path = os.path.join('data', 'Tuberculosis', 'Tuberculosis5fold', f'mont_fold{fold}.json')
                args_shenzhen.split_path = os.path.join('data', 'Tuberculosis', 'Tuberculosis5fold', f'shenzhen_fold{fold}.json')
            assert dataset == args_montgomery.dataset
            args_montgomery.num_epochs = num_epochs
            args_montgomery.learning_rate = learning_rate
            args_montgomery.buffer_size = buffer_size
            dic_hyperparams['node_montgomery_args'] = vars(args_montgomery).copy() 
            dic_hyperparams['node_montgomery_args']['device'] = str(dic_hyperparams['node_montgomery_args']['device'])
            mont_train_loader,_,mont_test_loader = get_loader(args_montgomery,KEYS)
            
            assert dataset == args_shenzhen.dataset
            dic_hyperparams['node_shenzhen_args'] = vars(args_shenzhen).copy() 
            dic_hyperparams['node_shenzhen_args']['device'] = str(dic_hyperparams['node_shenzhen_args']['device'])
            args_shenzhen.num_epochs = num_epochs
            args_shenzhen.buffer_size = buffer_size
            args_shenzhen.learning_rate = learning_rate
            shenzhen_train_loader,_,shenzhen_test_loader = get_loader(args_shenzhen,KEYS)
            
            train_loaders = [shenzhen_train_loader,mont_train_loader]
            test_loaders = {'shenzhen':shenzhen_test_loader,'montgomery':mont_test_loader}
            nodes_args = [args_shenzhen, args_montgomery]
        elif dataset == 'SkinLesion':
            n_nodes = 3
            #BCN dataset
            args_bcn = args_from_json('config_file/bcn.json', **dic_hyperparams)
            args_ham = args_from_json('config_file/ham.json', **dic_hyperparams)
            args_msk4 = args_from_json('config_file/msk4.json', **dic_hyperparams)
            if cross_val:
                args_bcn.split_path = os.path.join('data', 'Melanoma', 'Skin5fold', f'bcn_fold{fold}.json')
                args_ham.split_path = os.path.join('data', 'Melanoma', 'Skin5fold', f'ham_fold{fold}.json')
                args_msk4.split_path = os.path.join('data', 'Melanoma', 'Skin5fold', f'msk4_fold{fold}.json')
            assert dataset == args_bcn.dataset
            args_bcn.num_epochs = num_epochs
            args_bcn.learning_rate = learning_rate
            args_bcn.buffer_size = buffer_size
            dic_hyperparams['node_bcn_args'] = vars(args_bcn).copy() 
            dic_hyperparams['node_bcn_args']['device'] = str(dic_hyperparams['node_bcn_args']['device'])
            bcn_train_loader,_,bcn_test_loader = get_loader(args_bcn,KEYS)
            #HAM dataset
            assert dataset == args_ham.dataset
            args_ham.num_epochs = num_epochs
            args_ham.learning_rate = learning_rate
            args_ham.buffer_size = buffer_size
            dic_hyperparams['node_ham_args'] = vars(args_ham).copy() 
            dic_hyperparams['node_ham_args']['device'] = str(dic_hyperparams['node_ham_args']['device'])
            ham_train_loader,_,ham_test_loader = get_loader(args_ham,KEYS)
            #MSK4 dataset
            assert dataset == args_msk4.dataset
            args_msk4.num_epochs = num_epochs
            args_msk4.learning_rate = learning_rate
            args_msk4.buffer_size = buffer_size
            dic_hyperparams['node_msk4_args'] = vars(args_msk4).copy() 
            dic_hyperparams['node_msk4_args']['device'] = str(dic_hyperparams['node_msk4_args']['device'])
            msk4_train_loader,_,msk4_test_loader = get_loader(args_msk4,KEYS)

            train_loaders = [bcn_train_loader, ham_train_loader, msk4_train_loader]
            test_loaders = {'bcn': bcn_test_loader, 'ham': ham_test_loader,'msk4': msk4_test_loader}
            nodes_args = [args_bcn, args_ham, args_msk4]
    
    if experiment_name is None:
        text_params = [setting,epoch_policy,str(num_epochs) + 'epochs', str(n_rounds) + 'rounds', 'lr', str(learning_rate), 'buffer' + str(buffer_size)]
        if not share_classifier:
            text_params.append('noshareClassifier')
        if not share_buffer:
            text_params.append('noshareBuffer')
        if cross_val:
            text_params.append(f'crossVal_fold{fold}')
        experiment_name = os.path.join(outdir,dataset,setting, '_'.join(text_params) )
    
    dic_hyperparams['args'] = n_nodes
    dic_hyperparams['experiment_name'] = experiment_name


    list_z_c = generate_z(buffer_size, n_nodes)
    saver = Saver(outdir, experiment_name,wandb_mode)
    nodes = [Node(idx,saver,train_loaders[idx],nodes_args[idx], list_z_c[idx], share_classifier, share_buffer) for idx in range(n_nodes)]

    saver.log_hparams(dic_hyperparams)
    cmd = str(sys.argv)
    saver.log_cmd(cmd)

    routes = None
    for round in range(n_rounds):
        click.echo(f"Round {round} started.")

        # TRAINING PHASE
        click.echo("TRAINING PHASE STARTED")
        for node in nodes:
            round_epochs = get_epochs_num(epoch_policy, node.args.num_epochs, round, n_rounds)
            node.train(round_epochs)
        click.echo("TRAINING PHASE FINISHED")

        # TEST PHASE
        click.echo("TESTING PHASE STARTED")
        
        
        for test_name, test_loader in test_loaders.items():
            all_acc = []
            all_probs = []
            labels = None
            for node in nodes:
                node_test_acc, probs, labels = node.test(test_loader,test_name,round)
                all_acc.append(node_test_acc)
                all_probs.append(probs)
            all_acc = torch.tensor(all_acc)
            saver.log_loss(f'Controller/{test_name}/Acc/Mean', all_acc.mean().item(), round)
            saver.log_loss(f'Controller/{test_name}/Acc/STD', all_acc.std().item(), round)

        click.echo("TESTING PHASE FINISHED")

        #CKPT PHASE
        
        if (round%save_every) == 0:
            for node in nodes:
                node.save_ckpt(round)
        
        
        
        click.echo("DATA EXCHANGE PHASE STARTED")
        routes = get_routes(len(nodes))
        for route in routes:
            dest_node_idx = route[1]
            source_node_idx = route[0]
            click.secho(f"Node {source_node_idx} -> Node {dest_node_idx}",fg="yellow")
            nodes[dest_node_idx].receive_data(*nodes[source_node_idx].send_data())
        click.echo("DATA EXCHANGE PHASE FINISHED")

        # RESET PHASE
        click.echo("RESETTING NODES")
        for node in nodes:
            node.reset()

        click.echo(f"Round {round} finished.")

if __name__ == "__main__":
    main()
