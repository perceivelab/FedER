# coding=utf-8
from __future__ import absolute_import, division, print_function

import numpy as np
import torch.nn.functional as F
import torch

from tqdm import tqdm

from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, RocCurveDisplay
from utils.plot_conf_matrix import plot_confusion_matrix
from utils.averageMeter import AverageMeter
from utils.utils import compute_metrics, get_accuracy
from torch.nn import CrossEntropyLoss
import matplotlib
matplotlib.use('Agg')


def valid(args, logger, model, saver, phase, test_loader,test_dataset, epoch, KEYS, node_id = None):
    # Validation!
    figure = None
    eval_losses = AverageMeter()

    logger.info(f"***** Running {phase} *****")
    logger.info("  Num steps = %d", len(test_loader))
    logger.info("  Batch size = %d", args.eval_batch_size)

    if args.loss_type == 'CrossEntropy':
        loss_fct = CrossEntropyLoss(weight = args.loss_weights if torch.is_tensor(args.loss_weights) else None)
    
    
    model.eval()
    all_preds, all_label, all_logits = [], [], []
    epoch_iterator = tqdm(test_loader,
                          desc=f"{phase}... (loss=X.X)",
                          bar_format="{l_bar}{r_bar}",
                          dynamic_ncols=True)
    #loss_fct = torch.nn.CrossEntropyLoss(weight = args.loss_weights.to(args.device) if torch.is_tensor(args.loss_weights) else None)
    for step, batch in enumerate(epoch_iterator):
        x = batch[KEYS[0]].to(args.device)
        y = batch[KEYS[-1]].to(args.device).long()
        
        with torch.no_grad():
            x.to(dtype=torch.float, device=args.device)
            logits = model(x)
            
            eval_loss = loss_fct(logits.to(args.device), y.to(args.device))
            eval_losses.update(eval_loss.item())
            
            preds = torch.argmax(logits, dim=-1)
            
        if len(all_preds) == 0:
            all_preds.append(preds.detach().cpu().numpy())
            all_label.append(y.detach().cpu().numpy())
            all_logits.append(logits.detach().cpu().numpy())
        else:
            all_preds[0] = np.append(
                all_preds[0], preds.detach().cpu().numpy(), axis=0
            )
            all_label[0] = np.append(
                all_label[0], y.detach().cpu().numpy(), axis=0
            )
            all_logits[0] = np.append(all_logits[0], logits.detach().cpu().numpy(), axis = 0)
        epoch_iterator.set_description(f"{phase}... (loss=%2.5f)" % eval_losses.val)

    all_preds, all_label, all_logits = all_preds[0], all_label[0], all_logits[0]
    
    accuracy_dict = get_accuracy(all_preds, all_label, accuracy_type = args.accuracy)
        
    conf_matrix = confusion_matrix(all_label, all_preds)
    class_names = np.arange(model.num_classes)
    figure = plot_confusion_matrix(conf_matrix, class_names=class_names)
        
    metrics = compute_metrics(all_label, all_preds, False)
    
    '''ROC metrics'''
    try:
        all_probs = F.softmax(torch.from_numpy(all_logits), dim = 1)
        all_probs = all_probs.numpy()
        auc = roc_auc_score(all_label, all_probs[:,1])
    except ValueError:
        auc = 0
        pass
    fpr, tpr, thresholds = roc_curve(all_label, all_probs[:,1])

    logger.info("\n")
    logger.info(f"{phase} Results")
    logger.info("Global Steps: %d" % epoch)
    logger.info(f"{phase} Loss: %2.5f" % eval_losses.avg)
    
    for k in accuracy_dict.keys():
        logger.info(f"{phase} {k}: %2.5f" % accuracy_dict[k])
                       
    
    
    for k in accuracy_dict.keys():
        if node_id is not None:
            tag = f"{phase}/node_{node_id}/{k}"
        else:
            tag = f"{phase}/{k}"
        saver.log_loss(tag + '/'+test_dataset, accuracy_dict[k], epoch)
    if figure is not None:
        if node_id is not None:
            tag = f'{phase}/node_{node_id}/conf_matrix'
        else:
            tag = f'{phase}/conf_matrix'
        saver.writer.add_figure(tag, figure, global_step=epoch)
    
    if node_id is not None:
        tag = f"{phase}/node_{node_id}/loss"
    else:
        tag = f"{phase}/loss"
    saver.log_loss(tag + '/'+test_dataset, eval_losses.avg, epoch)
    
    if node_id is not None:
        tag = f"{phase}/node_{node_id}/auc"
    else:
        tag = f"{phase}/auc"
    saver.log_loss(tag + '/'+test_dataset, auc, epoch)
    roc_display = RocCurveDisplay.from_predictions(all_label, all_probs[:,1])
    if node_id is not None:
        tag = f"{phase}/node_{node_id}/roc"
    else:
        tag = f"{phase}/roc"
    saver.writer.add_figure(tag, roc_display.figure_, global_step=epoch)
    
    for m in metrics.keys():
        if node_id is not None:
            tag = f"metrics/{phase}/node_{node_id}/{m}"
        else:
            tag = f"metrics/{phase}/{m}"
        saver.log_loss(tag + '/'+test_dataset, metrics[m], epoch)
    
    #eval_losses.reset()
    
    return accuracy_dict , {'eval_loss': eval_losses.val}, {'auc': auc, 'fpr': fpr.tolist(), 'tpr': tpr.tolist(), 'thresholds': thresholds.tolist()},all_probs, all_label