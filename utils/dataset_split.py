# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 16:00:38 2020

@author: Amoko
"""
import os
import shutil

def split():
    root_src = '0819class18N13S4P1_LG10'
    root_dst = root_src + 'split'

    # 1 mkdir dst_part
    if not os.path.exists(root_dst):
        os.mkdir(root_dst)
    path = os.path.join(root_dst, 'train')
    os.mkdir(path)
    path = os.path.join(root_dst, 'val')
    os.mkdir(path)

    # 2 division
    subs = os.listdir(root_src)
    for sub in subs:
        src = os.path.join(root_src, sub)
        filenames = os.listdir(src)
        L = len(filenames)
        print(src, L)
        for i in range(L):
            fn = os.path.join(src, filenames[i])
            if i % 5 == 0:
                dst = os.path.join(root_dst, 'val', sub)
            else:
                dst = os.path.join(root_dst, 'train', sub)
            if not os.path.exists(dst):
                os.mkdir(dst)
            shutil.copy(fn, dst)


if __name__ == '__main__':
    split()


