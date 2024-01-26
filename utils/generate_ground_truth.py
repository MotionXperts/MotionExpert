import pickle as pkl
import numpy as np
import os
import json
path = '/home/weihsin/datasets/FigureSkate/HumanML3D_g/global_human_test.pkl'
with open(path, 'rb') as f:
    data = pkl.load(f)

filepath = 'humalMLgroundtruth.json'
dictitory = {}

for i in range(len(data)):
    print(data[i]['video_name'])
    # dict_keys(['video_name', 'labels'])
    video_name = data[i]['video_name']
    labels   = data[i]['labels']
    dictitory[video_name] = labels

with open(filepath, 'w') as f:
    # wirte dictionary to json file
    json.dump(dictitory, f)