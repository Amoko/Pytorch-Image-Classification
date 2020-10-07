#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 18:07:22 2020

@author: Amoko
"""

import os
os.environ['CUDA_VISIBLE_DEVICES']= '0'
import csv
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from efficientnet_pytorch import EfficientNet

class MyImageFolder(datasets.ImageFolder):
    def __init__(self, root, transform=None):
        super(MyImageFolder, self).__init__(root, transform=transform)

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target, path

def get_dataloader(path_val):
    val_transforms = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
    
    #val_dataset = datasets.ImageFolder(path_val, val_transforms)
    val_dataset = MyImageFolder(path_val, val_transforms)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=256, shuffle=False, num_workers=8)
    return val_loader

def load_model(model_path, device):
    model = EfficientNet.from_name('efficientnet-b3')
    conv_output = model._fc.in_features
    model._fc = nn.Linear(conv_output, 18)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

if __name__ == '__main__':
    data_path = '/data/houyj/data/0819class18N13S4P1_LG10'
    
    model_path = 'load/Efficientnet3.0819class18N13S4P1_LG10split.0.9797'

    data_name = os.path.basename(data_path)
    model_name = os.path.basename(model_path)
    if not os.path.exists(data_name):
        os.makedirs(data_name)
    csv_save_path = os.path.join(data_name, model_name + '.csv')
    

    dataloader = get_dataloader(data_path)
    class_names = dataloader.dataset.classes
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_path, device)
    print('start', time.ctime())
    print('samples number:', len(dataloader.dataset))
    start = time.time()
    with open(csv_save_path, 'w') as fp:
        writer = csv.writer(fp)
        for batch_idx, (inputs, labels, img_paths) in enumerate(dataloader):
            if batch_idx % 1000 == 0:
                print(batch_idx * len(labels), time.ctime())
                #break
            inputs, labels = inputs.to(device), labels.to(device)
            with torch.no_grad():
                outputs = model(inputs)
                outputs = F.softmax(outputs, dim=1)
            pred_scores, pred_idxs = outputs.max(dim=1)
            for i in range(len(labels)):
                true_name = class_names[labels[i].item()]
                pred_name = class_names[pred_idxs[i].item()]
                score = pred_scores[i].item()
                line = [img_paths[i], true_name, pred_name, score]
                writer.writerow(line)
    end = time.time()
    print('average cost:', (end - start)/len(dataloader.dataset))
    print('end', time.ctime()) 
        
        
        
        
        
