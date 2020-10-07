#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 19:48:52 2020

@author: Amoko
"""

import os

class MyConfig():
    def __init__(self):
        # 1 path
        path_dataset = '/data/amoko/data/0814class18N13S4P1_LG9split'
        dataset_name = os.path.basename(path_dataset)
        model_name = 'Efficientnet3'
        self.path_class = '{}_class.pkl'.format(dataset_name)
        self.path_dataset = path_dataset
        self.path_model_prefix = '{}.{}'.format(model_name, dataset_name)   
        
        # 2 train
        self.USE_RED_LOSS = True
        self.NUM_CLASSES = 18
        self.NUM_WORKERS = 8
        self.NUM_EPOCHS = 1000
        self.gpu = '0,2,3'
        #self.gpu = '2'
        self.NUM_GPUS = len(self.gpu.split(','))
        self.distributed = self.NUM_GPUS > 1
        self.amp = True
        train_mode = 'freeze_conv'
        #train_mode = 'ultimate'
        self.train_mode = train_mode
        if self.train_mode == 'freeze_conv':
            self.BATCH_SIZE = 256
            self.LR = 1e-3
        else:
            self.BATCH_SIZE = 64
            self.LR = 1e-4
        if self.amp:
            self.BATCH_SIZE = self.BATCH_SIZE * 2    
        # only for ultimate
        ACC = 0.8902
        self.path_model_saved = '{}.{:.4f}'.format(self.path_model_prefix, ACC)
