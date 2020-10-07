#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 19:35:24 2020

@author: Amoko
"""

from config import MyConfig
from dataloader import get_dataloader
cfg = MyConfig()

import os
os.environ['CUDA_VISIBLE_DEVICES'] = cfg.gpu
import warnings
warnings.filterwarnings('ignore')
import time
import pickle
import torch
torch.backends.cudnn.benchmark = True
import torch.nn as nn
import torch.optim as optim
# distributed
import torch.distributed as dist
import torch.multiprocessing as mp
# amp
from torch.cuda.amp import autocast, GradScaler
from torchsummary import summary
from efficientnet_pytorch import EfficientNet

#---------------------------- red loss ---------------------------- 
def get_top_name(label):
    if label[:4] in ['commie', 'pinko']:
        top_name = 'ban'
    else:
        top_name = 'pass'
    return top_name

def get_alpha_sample_weight(y_pred, y_true, epoch):
    L = len(y_pred)
    alpha =  [1] * L
    for i in range(L):
        # place your rule here
        pass
    return alpha

def get_loss_percent(y_true, loss):
    ban_loss_list = [0] * len(y_true)
    pass_loss_list = [0] * len(y_true)
    for i,label in enumerate(y_true):
        top_true = get_top_name(label)
        if top_true == 'ban':
            ban_loss_list[i] = loss[i]
        else:
            pass_loss_list[i] = loss[i]

    ban_sample_loss_percent = sum(ban_loss_list)/sum(loss)
    pass_sample_loss_percent = sum(pass_loss_list)/sum(loss)
    return ban_sample_loss_percent, pass_sample_loss_percent
#---------------------------- red loss ---------------------------- 

def train(model, mycriterion, opt, device, train_loader, epoch, local_rank, scaler):
    model.train()
    train_loss = 0
    correct = 0
    samples_num = len(train_loader.dataset) // cfg.NUM_GPUS
    interval = samples_num // (10 * cfg.BATCH_SIZE)
    class_names = train_loader.dataset.classes
    if cfg.distributed:
        train_loader.sampler.set_epoch(epoch)
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        with autocast(enabled=cfg.amp):
            outputs = model(inputs)        
            if cfg.USE_RED_LOSS:
                loss = mycriterion(outputs, labels)
                #--------------------------get alpha--------------------------
                index_pred = torch.max(outputs, dim=1)[1].tolist()
                index_true = labels.tolist()
                y_pred = [class_names[e] for e in index_pred]
                y_true = [class_names[e] for e in index_true]
                #print('pred:', index_pred[:5], y_pred[:5])
                #print('true:', index_true[:5], y_true[:5])
                alpha = get_alpha_sample_weight(y_pred, y_true, epoch)
                alpha = torch.Tensor(alpha).to(device)
                #print('sample_weight:', alpha[:5])
                #--------------------------get alpha--------------------------
                ban_percent_before, _ = get_loss_percent(y_true, loss)
                loss = alpha * loss
                ban_percent_after, _ = get_loss_percent(y_true, loss)
                loss = loss.mean()
            else:
                loss = mycriterion(outputs, labels)
        
        opt.zero_grad()
        if cfg.amp:
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            opt.step()
        
        if batch_idx % interval == 0 and local_rank == 0:
            if cfg.USE_RED_LOSS:
                details = 'ban_percent: {:.2f}% -> {:.2f}%'.format(
                    100 * ban_percent_before, 100 * ban_percent_after)
            else:
                details = ''
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\t batch_loss: {:.6f}'
                    .format(epoch, batch_idx * cfg.BATCH_SIZE, samples_num,
                    100. * batch_idx / len(train_loader), loss.item()),
                    details)

  
        train_loss += loss.item() # sum up batch loss
        pred = outputs.argmax(dim=1, keepdim=True) # get the index of the max log-probability
        correct += pred.eq(labels.view_as(pred)).sum().item()
        
    if cfg.distributed:
        correct_sum = reduce_tensor(torch.as_tensor(correct).to(device))
        correct = correct_sum.item()
    train_loss /= batch_idx
    train_acc = correct / len(train_loader.dataset)
    if local_rank == 0:
        print('train_set: average_batch_loss: {:.4f}, accuracy: {}/{} ({:.2f}%)'.format(
            train_loss, correct, len(train_loader.dataset), 100. * train_acc))
  
def validation(model, criterion, device, val_loader, local_rank):
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(val_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item() # sum up batch loss
            pred = outputs.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(labels.view_as(pred)).sum().item()
            total += labels.size(0)
    val_loss /= batch_idx
    val_acc = correct / total
    if cfg.distributed:
        correct_sum = reduce_tensor(torch.as_tensor(correct).to(device))
        total_sum = reduce_tensor(torch.as_tensor(total).to(device))
        #print(correct_sum, total_sum)
        correct = correct_sum.item()
        total = total_sum.item()
        val_acc = correct / total
    if local_rank == 0:
        print('val_set:   average_batch_loss: {:.4f}, accuracy: {}/{} ({:.2f}%)'.format(
            val_loss, correct, total, 100. * val_acc))
    return val_acc

def reduce_tensor(tensor):
    t1 = tensor.clone()
    dist.all_reduce(t1)
    return t1

def main_worker(local_rank):
    if cfg.distributed:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl',init_method='env://', 
            world_size=cfg.NUM_GPUS, rank=local_rank)

    # 0 data
    train_loader, val_loader = get_dataloader(cfg)
    class_names = train_loader.dataset.classes
    with open(cfg.path_class, 'wb') as fp:
        pickle.dump(class_names, fp)
   
    # 1 model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EfficientNet.from_pretrained('efficientnet-b3')
    conv_output = model._fc.in_features # 512 for resnet18, 1536 for b3, 1792 for b4
    model._fc = nn.Linear(conv_output, cfg.NUM_CLASSES)
    if cfg.train_mode == 'freeze_conv':
        for param in model.parameters():
            param.requires_grad = False # save computation
        model._fc.requires_grad_(True)
    else:
        map_location = 'cuda:{}'.format(local_rank)
        print(map_location)
        checkpoint = torch.load(cfg.path_model_saved, map_location=map_location)
        model.load_state_dict(checkpoint)
        print('load weights of ', cfg.path_model_saved)
        for param in model.parameters():
            param.requires_grad = True # unlock con
    model = model.to(device)
    if local_rank == 0:
        print('train samples:', len(train_loader.dataset))
        print('val samples:', len(val_loader.dataset))
        print(summary(model, (3, 224, 224)))
    if cfg.distributed:
        model = nn.parallel.DistributedDataParallel(model,device_ids=[local_rank])

    print(local_rank, 'model load over')
    # 2 loss, opt
    criterion = nn.CrossEntropyLoss()  
    if cfg.USE_RED_LOSS:
        mycriterion = nn.CrossEntropyLoss(reduction='none')
    else:
        mycriterion = nn.CrossEntropyLoss()
    opt = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.LR)
    if cfg.amp:
        scaler = GradScaler()
    else:
        scaler = None

    # 3 train
    print('start,', time.ctime())
    best_val_acc = 0
    for epoch in range(1, cfg.NUM_EPOCHS + 1):
        train(model, mycriterion, opt, device, train_loader, epoch, local_rank, scaler)
        val_acc = validation(model, criterion, device, val_loader, local_rank)
        if local_rank != 0:
            continue
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            path = '{}.{:.4f}'.format(cfg.path_model_prefix, best_val_acc)
            if cfg.distributed:
                torch.save(model.module.state_dict(), path)
            else:
                torch.save(model.state_dict(), path)
        print('best_val_acc: %0.4f' % (best_val_acc))
        print(time.ctime())
        print('-' * 60)
    print('end,', time.ctime())

if __name__ == '__main__':
    print('dataset:', cfg.path_dataset)
    print('train_mode:', cfg.train_mode)
    print('NUM_GPUS:', cfg.NUM_GPUS)
    print('distributed:', cfg.distributed)
    print('amp:', cfg.amp)
    if cfg.distributed:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '40404'
        print('start mp', time.ctime())
        mp.spawn(main_worker, nprocs=cfg.NUM_GPUS)
    else:
        main_worker(0)


         
