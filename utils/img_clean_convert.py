# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 16:00:38 2020

@author: Amoko
"""

import os
import time
import hashlib
from PIL import Image
import multiprocessing as mp

def get_md5_fn(fn):
    md5 = hashlib.md5(open(fn, 'rb').read()).hexdigest()
    return md5

def get_md5_bytes(b):
    md5 = hashlib.md5(b).hexdigest()
    return md5

def replace_with_jpg(src):
    try:
        img = Image.open(src).convert('RGB')
        md5 = get_md5_bytes(img.tobytes())
        #print(get_md5_fn(src), md5)
        dst = os.path.join(os.path.dirname(src), md5 + '.jpg')
        os.remove(src)
        img.save(dst)
    except:
        print('bad', src)
        os.remove(src)

def remove_bad_img(fn):
    try:
        img = Image.open(fn).convert('RGB')
        img = img.resize((224, 224))
    except:
        print('bad', fn)
        os.remove(fn)

if __name__ == '__main__':
    print(time.ctime())
    
    the_root = '0819class18N13S4P1_LG10'
    just_remove_bad = False

    for root, dirs, files in os.walk(the_root):
        filenames = []
        for file in files:
            filenames.append(os.path.join(root, file))
        if filenames == []:
            continue
        print(root, len(filenames))

    pool = mp.Pool(processes = 35)
    if just_remove_bad:
        print('start remove bad img')
        pool.map(remove_bad_img, filenames)
    else:
        print('start replace with jpg')
        pool.map(replace_with_jpg, filenames)

    print(time.ctime())
