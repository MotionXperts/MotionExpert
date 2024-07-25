import pickle
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision.io import read_video
# from VideoAlignment.dataset.data_augment import create_data_augment
import os
import torch.distributed as dist

USER = os.environ['USER']

bonelink = [(0, 1), (0, 2), (0, 3), (1, 4), (2,5), (3,6), (4, 7), (5, 8), (6, 9), (7, 10), 
            (8, 11), (9, 12), (9, 13), (9, 14), (12, 15), (13, 16), (14, 17), (16, 18), (17, 19), (18,  20), (19, 21)]

def generate_data(current_keypoints ):
    joint_coordinate = np.zeros((3,len(current_keypoints),22))
    bone_coordinate = np.zeros((3,len(current_keypoints),22))
    for i in range(len(current_keypoints)):
        for j in range(0,len(current_keypoints[i]),3):
            joint_coordinate[:, i, j//3] = current_keypoints[i,j:j+3]

    for v1, v2 in bonelink:
        bone_coordinate[:, :, v2] = joint_coordinate[:, :, v1] - joint_coordinate[:, :, v2]

    coordinates = np.concatenate((joint_coordinate, bone_coordinate), axis=0)
    return coordinates

class DatasetLoader(Dataset):
    def __init__(self, pretrain, pkl_file):
        with open(pkl_file, 'rb') as f:
            self.data_list = pickle.load(f)
        self.samples = []
        max_len = 0  
        
        if pretrain:
            self.standard = generate_data(self.data_list[0]['features'])
        if not pretrain:
            self.standard = generate_data(self.data_list[0]['features'])
            self.data_list = self.data_list[1:]   
        for item in self.data_list:
            features =  generate_data(item['features'])
            max_len = max(max_len, len(features[0]))
            video_name = item['video_name']
            for label in item['labels']:
                if pretrain : 
                    label = "Motion Description : " + label
                else : 
                    label = "Motion Instruction : " + label
                self.samples.append((features, label, video_name)) 

        self.max_len = max_len  

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        ## features: 6 x frame x 22
        features, label, video_name = self.samples[idx]
        keypoints_mask = torch.ones(22)       
        current_len = torch.tensor(len(features[0]))

        return  video_name, torch.FloatTensor(features), torch.FloatTensor(keypoints_mask), torch.FloatTensor(self.standard), current_len, label
