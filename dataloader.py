#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 19:48:52 2020

@author: Amoko
"""

import os
import torch
from torchvision import datasets, transforms


def get_transforms():
    train_transforms = transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
    val_transforms = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
    return train_transforms, val_transforms

def get_dataset(path_dataset, train_transforms, val_transforms):
    path_train = os.path.join(path_dataset, 'train')
    path_val = os.path.join(path_dataset, 'val')
    train_dataset = datasets.ImageFolder(path_train, train_transforms)
    val_dataset = datasets.ImageFolder(path_val, val_transforms)
    return train_dataset, val_dataset

def get_dataloader(cfg):
    path_dataset = cfg.path_dataset
    BATCH_SIZE, NUM_WORKERS = cfg.BATCH_SIZE, cfg.NUM_WORKERS

    train_transforms, val_transforms = get_transforms()
    train_dataset, val_dataset = get_dataset(path_dataset, 
                                             train_transforms, val_transforms)
   
    if cfg.NUM_GPUS > 1:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
            sampler=train_sampler)
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
            sampler=val_sampler)
    else: 
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    return train_loader, val_loader
