#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 16:00:38 2020

@author: Amoko
"""

import os
import shutil
import random

def get_random_subset(path_dir, NUM):
    path_dir2 = path_dir + '_sub' + str(NUM)
    subs = os.listdir(path_dir)
    
    count = 0
    for sub in subs:
        src_dir = os.path.join(path_dir, sub)
        dst_dir = os.path.join(path_dir2, sub)
        if not os.path.isdir(src_dir):
            continue
        filenames = os.listdir(src_dir)
        count += len(filenames)
        print(sub, len(filenames))
        filenames = random.sample(filenames, NUM)
        
        # make copy
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)
        for cargo in filenames:
            cargo = os.path.join(src_dir, cargo)
            shutil.copy(cargo, dst_dir)
    
    print(count)
    print(len(filenames)*len(subs))    

if __name__ == '__main__':
    path_dir = '0819class18N13S4P1_LG10'
    NUM = 100
    get_random_subset(path_dir, NUM)

