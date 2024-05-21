import pickle
import torch
import numpy as np
from torch.utils.data import Dataset

bonelink = [(0, 1), (0, 2), (0, 3), (1, 4), (2,5), (3,6), (4, 7), (5, 8), (6, 9), (7, 10), 
            (8, 11), (9, 12), (9, 13), (9, 14), (12, 15), (13, 16), (14, 17), (16, 18), (17, 19), (18,  20), (19, 21)]

def generate_data(current_keypoints ):
    joint_coordinate = np.zeros((3,len(current_keypoints),22))
    bone_coordinate = np.zeros((3,len(current_keypoints),22))

    for i in range(len(current_keypoints)):
        for j in range(0,len(current_keypoints[i]),3):
            joint_coordinate[:, i, j//3] = current_keypoints[i,j:j+3]

    for v1, v2 in bonelink:
        bone_coordinate[:, :, v1] = joint_coordinate[:, :, v1] - joint_coordinate[:, :, v2]

    coordinates = np.concatenate((joint_coordinate, bone_coordinate), axis=0)
    return coordinates

class HumanMLDataset(Dataset):
    def __init__(self, pkl_file, finetune,transform=None,split='train'):
        with open(pkl_file, 'rb') as f:
            self.data_list = pickle.load(f)
        self.samples = []
        max_len = 0  

        for item in self.data_list:
            features =  generate_data(item['features'])
            max_len = max(max_len, len(features[0]))
            video_name = item['video_name']
            for label in item['labels']:
                if(finetune == False) : 
                    label = "Motion Description : " + label
                else :
                    label = "Motion Instruction : " + label
                self.samples.append((features, label, video_name))
                if split != 'train':
                    break

        self.max_len = max_len  
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        features, label, video_name = self.samples[idx]
        padded_features = np.zeros((6,self.max_len, 22)) 
        keypoints_mask = np.ones(22)       
        current_len = len(features[0])
        video_mask = np.ones(self.max_len)
        video_mask[current_len:] = 0
        padded_features[:,:current_len, :] = features
        sample = {
            "video_name": video_name,
            "keypoints": torch.FloatTensor(padded_features),
            "keypoints_mask": torch.FloatTensor(keypoints_mask),
            "video_mask": torch.FloatTensor(video_mask),
            "label": label,
        }
        return sample