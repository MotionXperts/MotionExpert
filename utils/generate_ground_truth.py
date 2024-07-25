import pickle as pkl
import numpy as np
import os
import json
path = '/home/weihsin/datasets/FigureSkate/HumanML3D_g/global_human_test.pkl'
path = '/home/weihsin/datasets/Loop/test_Loop.pkl'
path1 = '/home/weihsin/datasets/VQA/test_local.pkl'
path2 = '/home/weihsin/datasets/VQA/train_local.pkl'
with open(path1, 'rb') as f:
    data1 = pkl.load(f)
with open(path1, 'rb') as f:
    data2 = pkl.load(f)
    
filepath = 'humalMLgroundtruth.json'
dictitory = {}

for i in range(len(data1)):
    print(data1[i]['video_name'])
    print(data1[i]['labels'])
    # dict_keys(['video_name', 'labels'])
    video_name = data1[i]['video_name']
    labels   = data1[i]['labels']
    dictitory[video_name] = labels

for i in range(len(data2)):
    print(data2[i]['video_name'])
    print(data2[i]['labels'])
    # dict_keys(['video_name', 'labels'])
    video_name = data2[i]['video_name']
    labels   = data2[i]['labels']
    dictitory[video_name] = labels

with open(filepath, 'w') as f:
    json.dump 
    json.dump(dictitory, f, indent=4)