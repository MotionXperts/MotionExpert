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
        standard_path = '/home/andrewchen/Error_Localize/standard_features.pkl'
        with open(standard_path, 'rb') as f:
            standard_features_file = pickle.load(f)
            self.standard_features_list = [generate_data(item['features']) for item in standard_features_file]
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
            if 'Axel' in item['original_video_file']:
                std_features = self.standard_features_list[0]
            elif 'Axel_com' in item['original_video_file']:
                std_features = self.standard_features_list[1]
            elif 'Loop' in item['original_video_file']:
                std_features = self.standard_features_list[2]
            else:
                std_features = self.standard_features_list[3]
            trimmed_start = item['trimmed_start']
            if not item['standard_longer']:
                start_frame = item['start_frame'] + trimmed_start
                end_frame = item['end_frame'] + trimmed_start
                item['std_start_frame'] = 0
                item['std_end_frame'] = end_frame - start_frame
            else:
                item['std_start_frame'] = item['start_frame']
                item['std_end_frame'] = item['end_frame'] - 1
                start_frame = trimmed_start
                end_frame = trimmed_start + (item['std_end_frame'] - item['std_start_frame'])
            max_len = max(max_len, len(features[0]))
            video_name = item['video_name']
            if 'train' in pkl_file:
                labels = item['augmented_labels']
                labels.append(item['revised_label'])
            else:
                labels = [item['revised_label']]
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
            elif hasattr(self.cfg.TASK,'DIFFERENCE_TYPE') and self.cfg.TASK.DIFFERENCE_TYPE== 'Skeleton':
                features = features[:,start_frame:end_frame] 
                std_features = std_features[:,item['std_start_frame']:item['std_end_frame']]
                subtraction = item['subtraction']
                assert features.shape[1] ==  std_features.shape[1],f"""
                features.shape: {features.shape}
                std_features.shape: {std_features.shape}
                item["features"].shape: {item["features"].shape}
                trimmed_start: {item['trimmed_start']}
                features.shape[2] = {features.shape[1]} in \n 
                {item['video_file']},\n
                original features.shape[2] = {item['features'].shape[1]}, \b
                std_features.shape[1] = {std_features.shape[1]}, \n 
                original_start_frame = {item['start_frame']}, original_end_frame = {item['end_frame']},\n
                trimmed_start = {item['trimmed_start']}, trimmed_end = {item['trimmed_end']}, \n
                standard_longer? {item['standard_longer']},
                """
            else:
                subtraction = torch.empty(0)
            for label in labels:
                if pretrain : 
                    label = "Motion Description : " + label
                else : 
                    label = "Motion Instruction : " + label
                self.samples.append((features, label, video_name,subtraction, std_features)) 

        print('Sample length:', len(self.samples))
        self.max_len = max_len  

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        ## features: 6 x frame x 22
        features, label, video_name, subtraction, std_features = self.samples[idx]
        keypoints_mask = torch.ones(22)       
        current_len = torch.tensor(len(features[0]))

        
        
        # change self.standard to std_features
        return  video_name, torch.FloatTensor(features), torch.FloatTensor(keypoints_mask),  torch.FloatTensor(std_features), current_len, label, subtraction
