#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 16:00:38 2020

@author: Amoko
"""

import os

def check_data():
    path = '0503class25N17S7P1_LG3'
    path = '0819class18N13S4P1_LG10'
    print(path)
    
    subs = os.listdir(path)
    print(len(subs))
    
    a = []
    for sub in subs:
        L = len(os.listdir(os.path.join(path, sub)))
        a.append([sub, L])
    #a.sort(key=lambda x:[x[0][:10], x[1]])
    a.sort(key=lambda x:x[1])
    for e in a:
        print(e[0]+ ',', e[1])

    print([e[0] for e in a])
    print([e[1] for e in a])   
 
    num = sum([e[1] for e in a])
    num_disu = sum([e[1] for e in a if e[0][:4]=='disu'])
    num_neut = sum([e[1] for e in a if e[0][:4]=='neut'])
    num_porn = sum([e[1] for e in a if e[0][:4]=='porn'])
    num_sexy = sum([e[1] for e in a if e[0][:4]=='sexy'])
    print(num, num_neut,  num_sexy, num_disu, num_porn)

if __name__ == '__main__':
    check_data()
