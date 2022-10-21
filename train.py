# coding=utf-8
from __future__ import absolute_import, division, print_function

import os
import torch
import json

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from utils.scheduler import WarmupLinearSchedule, WarmupCosineSchedule
from validation import valid
from utils.data_utils import get_loader
from utils.averageMeter import AverageMeter
from utils.utils import compute_metrics, get_accuracy, save_model, set_seed
from utils.pytorchtools import EarlyStopping
#from utils.utils import estimate_optim_lr
from monai.optimizers import LearningRateFinder
from torch.nn import CrossEntropyLoss
import matplotlib
matplotlib.use('Agg')


def train(node_idx, args, logger, model, buffer,num_epochs, KEYS, saver, train_loader, node_epochs = 0):
    """ Train the model """
    '''
    os.makedirs(args.output_dir, exist_ok=True)
    folders_logs = args.output_dir.split(os.path.sep)[1:]
    if folders_logs[0]== 'output':
        folders_logs = folders_logs[1:]
    sub_path_logs = ''
    for s in folders_logs:
        sub_path_logs = os.path.join(sub_path_logs, s)
    '''
    #writer = SummaryWriter(log_dir=os.path.join("logs", sub_path_logs))

    
    # Prepare dataset
    #train_loader, val_loader, test_loader = get_loader(args, KEYS)
    
    if args.loss_type == 'CrossEntropy':
        loss_fct = CrossEntropyLoss(weight = args.loss_weights if torch.is_tensor(args.loss_weights) else None)
        loss_fct.to(args.device)
        
    if args.lr_finder:
        lr = args.lower_lr
    else:
        lr = args.learning_rate
    # Prepare optimizer and scheduler
    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=lr,
                                    momentum=0.9,
                                    weight_decay=args.weight_decay)
    elif args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(),
                                lr=lr,
                                weight_decay=args.weight_decay)
    elif args.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(),
                                lr=lr,
                                weight_decay=args.weight_decay)
    elif args.optimizer == 'RMSprop':
        optimizer = torch.optim.RMSprop(model.parameters(),
                                lr=lr,
                                momentum=0.9,
                                weight_decay=args.weight_decay)
    if args.lr_finder:
        lr_finder = LearningRateFinder(model, optimizer, loss_fct, device=args.device)
        lr_finder.range_test(train_loader, val_loader, end_lr=args.upper_lr, num_iter=20)
        steepest_lr, _ = lr_finder.get_steepest_gradient()
        ax = lr_finder.plot()
        img = ax.figure
        saver.writer.add_figure('lr_finder', img, global_step=1)
        
        with open(os.path.join(args.output_dir, 'steepest_lr.json'), 'w') as fp:
            json.dump(str(steepest_lr), fp)
        
        optimizer.param_groups[0]['lr'] = steepest_lr
    '''
    steepest_lr = estimate_optim_lr(model = model,  optimizer = optimizer, loss = model.loss_fct, device = args.device, lower_lr = args.learning_rate, upper_lr = args.learning_rate * 1e-3, train_loader = train_loader, val_loader = val_loader, image_key = KEYS[0], label_key = KEYS[-1])
    
    print('*'*30, steepest_lr)
    exit()
    '''
    t_total = num_epochs
    
    if args.use_scheduler and args.decay_type == "cosine":
        scheduler = WarmupCosineSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total, cycles = args.cycles_scheduler)
    elif args.use_scheduler and args.decay_type == "linear":
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
        
    if args.early_stopping:
        early_stopping = EarlyStopping(patience = args.es_patience, delta = args.es_delta, path = os.path.join(args.output_dir, 'checkpoint_EarlyStopping.pt'))
        
    # Train!
    logger.info("***** Running training *****")
    logger.info("  Total optimization epochs = %d", num_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size)
    model.train()
    model.zero_grad()

    losses = AverageMeter()
    metrics = {
        'precision': AverageMeter(),
        'recall': AverageMeter(),
        'f1_score': AverageMeter(),
        'jaccard_score': AverageMeter()
        }
    
    best_acc, best_simple_acc, best_auc = 0, 0, 0
    
    for epoch in range(num_epochs):
        model.train()
        epoch_iterator = tqdm(train_loader,
                                  desc="Training (X / X epochs) (loss=X.X)",
                                  bar_format="{l_bar}{r_bar}",
                                  dynamic_ncols=True,
                                  )
        sum_train_accuracy = 0
        for step, batch in enumerate(epoch_iterator):
            # private data
            x = batch[KEYS[0]].to(args.device)
            y = batch[KEYS[-1]].to(args.device).long()
            if epoch == 0 and step == 0:
                saver.log_images(f"Real Images Node {node_idx}", x)
            if buffer is not None:
                # buffer data
                buf_images, buf_labels = buffer.get_data(x.shape[0])
                if epoch == 0 and step == 0:
                    saver.log_images(f"Buffer Images Node {node_idx}", buf_images)
                buf_images = buf_images.to(args.device)
                buf_labels = buf_labels.to(args.device).long()
                # concat private and buffer images
                x = torch.cat([x,buf_images])
                y = torch.cat([y,buf_labels])

            x = x.to(dtype=torch.float, device=args.device)
            
            with torch.set_grad_enabled(True):
                logits = model(x)
            
            loss = loss_fct(logits.view(-1, args.num_classes), y)
            
            pred_labels = logits.argmax(-1)
            
            if args.accuracy == 'both':
                accuracy_type = 'balanced'
            else:
                accuracy_type= args.accuracy
                
            batch_accuracy = get_accuracy(pred_labels.cpu().numpy(), y.cpu().numpy(), accuracy_type = accuracy_type)
            sum_train_accuracy += batch_accuracy['accuracy_'+accuracy_type]

            metrics_dict = compute_metrics(y.cpu().numpy(), pred_labels.cpu().numpy(), False)   
            for k in metrics_dict.keys():
                metrics[k].update(metrics_dict[k])

            loss.backward()
                
            losses.update(loss.item())
                
            optimizer.step()
                
            if args.use_scheduler:
                scheduler.step()
                
            optimizer.zero_grad()
            
            epoch_iterator.set_description("Training (%d / %d Steps) (loss=%2.5f)" % (epoch, num_epochs, losses.val))
            
                            
        
        epoch_train_accuracy = sum_train_accuracy / len(train_loader)
        saver.log_loss(f"Train/node_{node_idx}/accuracy_{accuracy_type}", epoch_train_accuracy,  node_epochs+epoch)
        saver.log_loss(f"Train/node_{node_idx}/loss", losses.avg,  node_epochs + epoch)
            
        for k in metrics.keys():
            saver.log_loss(f'Train/node_{node_idx}/{k}', metrics[k].avg,  node_epochs + epoch )
            
        losses.reset()
        for k in metrics.keys():
            metrics[k].reset()

    
    logger.info("Best Accuracy: \t%f" % best_acc)
    logger.info("End Training!")
    return model


def train_fedbn(model, data_loader, optimizer, loss_fun, device, KEYS):
    model.train()
    loss_all = 0
    total = 0
    correct = 0
    for batch in data_loader:

        optimizer.zero_grad()

        data = batch[KEYS[0]].to(device)
        target = batch[KEYS[-1]].to(device).long()

        output = model(data)
        loss = loss_fun(output, target)
        loss_all += loss.item()
        total += target.size(0)
        pred = output.data.max(1)[1]
        correct += pred.eq(target.view(-1)).sum().item()

        loss.backward()
        optimizer.step()

    return loss_all / len(data_loader), correct/total


def train_prox(args, model, server_model, data_loader, optimizer, loss_fun, device, KEYS):
    model.train()
    loss_all = 0
    total = 0
    correct = 0
    for step, batch in enumerate(data_loader):
        optimizer.zero_grad()

        data = batch[KEYS[0]].to(device)
        target = batch[KEYS[-1]].to(device).long()
        output = model(data)
        loss = loss_fun(output, target)
        if step>0:
            w_diff = torch.tensor(0., device=device)
            for w, w_t in zip(server_model.parameters(), model.parameters()):
                w_diff += torch.pow(torch.norm(w - w_t), 2)
                
            w_diff = torch.sqrt(w_diff)
            loss += args.mu / 2. * w_diff
                        
        loss.backward()
        optimizer.step()

        loss_all += loss.item()
        total += target.size(0)
        pred = output.data.max(1)[1]
        correct += pred.eq(target.view(-1)).sum().item()

    return loss_all / len(data_loader), correct/total
     

def test_fedbn(model, data_loader, loss_fun, device, KEYS):
    model.eval()
    loss_all = 0
    total = 0
    correct = 0
    with torch.no_grad():
        for batch in data_loader:
            data = batch[KEYS[0]].to(device)
            target = batch[KEYS[-1]].to(device).long()
            output = model(data)
            loss = loss_fun(output, target)
            loss_all += loss.item()
            total += target.size(0)
            pred = output.data.max(1)[1]
            correct += pred.eq(target.view(-1)).sum().item()

        return loss_all / len(data_loader), correct/total


################# Key Function ########################
def communication(setting, server_model, models, client_weights, client_num):
    with torch.no_grad():
        # aggregate params
        if setting.lower() == 'fedbn':
            for key in server_model.state_dict().keys():
                if 'bn' not in key:
                    temp = torch.zeros_like(server_model.state_dict()[key], dtype=torch.float32)
                    for client_idx in range(client_num):
                        temp += client_weights[client_idx] * models[client_idx].state_dict()[key]
                    server_model.state_dict()[key].data.copy_(temp)
                    for client_idx in range(client_num):
                        models[client_idx].state_dict()[key].data.copy_(server_model.state_dict()[key])
        else:
            for key in server_model.state_dict().keys():
                # num_batches_tracked is a non trainable LongTensor and
                # num_batches_tracked are the same for all clients for the given datasets
                if 'num_batches_tracked' in key:
                    server_model.state_dict()[key].data.copy_(models[0].state_dict()[key])
                else:
                    temp = torch.zeros_like(server_model.state_dict()[key])
                    for client_idx in range(len(client_weights)):
                        temp += client_weights[client_idx] * models[client_idx].state_dict()[key]
                    server_model.state_dict()[key].data.copy_(temp)
                    for client_idx in range(len(client_weights)):
                        models[client_idx].state_dict()[key].data.copy_(server_model.state_dict()[key])
    return server_model, models