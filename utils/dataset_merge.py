#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 16:00:38 2020

@author: Amoko
"""

import os
import time
import shutil

def merge_one(seru, no18):
    filenames_seru = os.listdir(seru)
    filenames_no18 = os.listdir(no18)
    print(seru, no18)
    l1 = len(filenames_seru)
    l2 = len(filenames_no18)
    for name in filenames_no18:
        if name not in filenames_seru:
            src = os.path.join(no18, name)
            shutil.copy(src, seru)

    filenames_seru = os.listdir(seru)
    print(seru, l1, l2, len(filenames_seru))

def merge_all(seru_all, no18_all):
    subs = os.listdir(no18_all)
    for sub in subs:
        seru = os.path.join(seru_all, sub)
        no18 = os.path.join(no18_all, sub)
        merge_one(seru, no18)

if __name__ == '__main__':
    print('start.', time.ctime())
    if 0:
        seru = '0819class18N13S4P1_LG10_sub100A/porn_human'
        no18 = '0819class18N13S4P1_LG10_sub100B/porn_human'
        merge_one(seru, no18)
    else:
        seru_all = '0819class18N13S4P1_LG10_sub100A'
        no18_all = '0819class18N13S4P1_LG10_sub100B'
        merge_all(seru_all, no18_all)

    print('end.', time.ctime())
        
