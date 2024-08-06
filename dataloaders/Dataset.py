import pickle
import torch
import numpy as np
from torch.utils.data import Dataset
# from VideoAlignment.dataset.data_augment import create_data_augment
import os,sys

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
    def __init__(self,cfg, pretrain, pkl_file):
        self.cfg = cfg
        with open(pkl_file, 'rb') as f:
            self.data_list = pickle.load(f)
        self.samples = []
        max_len = 0  
        
        if pretrain:
            self.standard = generate_data(self.data_list[0]['features'])
        if not pretrain:
            self.standard = generate_data(self.data_list[0]['features'])
            if 'train' in pkl_file and 'standard' in self.data_list[0]['name']:
                print(f'Standard motion {self.data_list[0]["name"]} found, skipping')
                self.data_list = self.data_list[1:] ## RGB subtract the standard already, no need to append standard   
        print('Data list length:', len(self.data_list))
        for item in self.data_list:
            features =  generate_data(item['features'])
            trimmed_start = item['trimmed_start']
            if not item['standard_longer']:
                start_frame = item['start_frame'] + trimmed_start
                end_frame = item['end_frame'] + trimmed_start
            else:
                start_frame = trimmed_start
                end_frame = trimmed_start + len(item['subtraction'])
            max_len = max(max_len, len(features[0]))
            video_name = item['video_name']
            labels = item['augmented_labels']
            if hasattr(self.cfg.TASK,'DIFFERENCE_TYPE') and self.cfg.TASK.DIFFERENCE_TYPE== 'RGB':
                subtraction = item['subtraction']
                features = features[:,start_frame:end_frame] 
                subtraction = subtraction[:features.shape[1]]
                assert features.shape[1] == subtraction.shape[0],f"""
                {generate_data(item['features']).shape}
                features.shape[2] = {features.shape[1]} in \n 
                {item['video_file']},\n
                original features.shape[2] = {item['features'].shape[1]}, \b
                subtraction.shape[0] = {subtraction.shape[0]}, \n 
                original_start_frame = {item['start_frame']}, original_end_frame = {item['end_frame']},\n
                trimmed_start = {item['trimmed_start']}, trimmed_end = {item['trimmed_end']}, \n
                standard_longer? {item['standard_longer']},
                """
            else:
                subtraction = torch.empty(0)
            labels.append(item['revised_label'])
            for label in labels:
                if pretrain : 
                    label = "Motion Description : " + label
                else : 
                    label = "Motion Instruction : " + label
                self.samples.append((features, label, video_name,subtraction)) 

        print('Sample length:', len(self.samples))
        self.max_len = max_len  

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        ## features: 6 x frame x 22
        features, label, video_name, subtraction = self.samples[idx]
        keypoints_mask = torch.ones(22)       
        current_len = torch.tensor(len(features[0]))

        
        

        return  video_name, torch.FloatTensor(features), torch.FloatTensor(keypoints_mask), torch.FloatTensor(self.standard), current_len, label, subtraction
