import pickle
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision.io import read_video
from VideoAlignment.dataset.data_augment import create_data_augment
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
        bone_coordinate[:, :, v1] = joint_coordinate[:, :, v1] - joint_coordinate[:, :, v2]

    coordinates = np.concatenate((joint_coordinate, bone_coordinate), axis=0)
    return coordinates

class Skating(Dataset):
    def __init__(self,cfg, pkl_file,transformation_policy='ORIGIN',split='train'):
        with open(pkl_file, 'rb') as f:
            self.data_list = pickle.load(f)
        self.samples = []
        self.cfg = cfg
        max_len = 0  

        self.standard = generate_data(self.data_list[0]['features'])
        if cfg is not None:
            self.standard_vid,_,_ = read_video(os.path.join(f'/home/{USER}/datasets/Axel_520_clip/processed_videos', f"{self.data_list[0]['video_name']}.mp4"), pts_unit='sec')
            self.data_list = self.data_list[1:]   

        for data_index,item in enumerate(self.data_list):
            features =  generate_data(item['features'])
            max_len = max(max_len, len(features[0]))
            video_name = item['video_name']
            for label in item['labels']:
                label = "Motion Instruction : " + label
                self.samples.append((features, label, video_name,data_index)) ## This will make cider score for the same input compute multiple times.
                # if split != 'train': ## But the loss will not be the same (Might not be very important as we are not in training time.)
                #     break

        self.max_len = max_len  
        self.transformation_policy = transformation_policy

        ## Branch 2 related
        if cfg is not None:
            self.data_preprocess,_ = create_data_augment(cfg,False)
            self.standard_vid = self.standard_vid.permute(0,3,1,2).float() / 255.0
            self.standard_vid = self.data_preprocess(self.standard_vid)
        else:
            self.standard_vid = torch.zeros(0)

        if dist.get_rank() == 0:
            print('\033[91m Warning: Bad implementation on vid_path in __getitem__. (Hardcoded path) \033[0m')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        ## features: 6 x frame x 22
        features, label, video_name,data_index = self.samples[idx]
        padded_features = torch.zeros((6,self.max_len, 22)) 
        keypoints_mask = torch.ones(22)       
        
        video_mask = torch.ones(self.max_len)
        video_mask[len(features[0]):] = 0
        # padded_features[:,:current_len, :] = features
        subtraction = torch.empty(1)
        
        ## Branch 2 related
        if self.cfg is not None:
            vid_path = os.path.join(f'/home/{USER}/datasets/Axel_520_clip/processed_videos', f'{video_name}.mp4')
            video, _ , metadata = read_video(vid_path, pts_unit='sec')
            video = video.permute(0,3,1,2).float() / 255.0
            video = self.data_preprocess(video)
            subtraction=self.data_list[data_index]['subtraction']

            ## this is used to avoid frame number difference between HYBRIK and torchvision.io
            if features.shape[1] > len(video):
                features = features[:,:len(video),:]
        else:
            video = torch.empty(1)

        current_len = torch.tensor(len(features[0]))
        
        if self.transformation_policy == 'ORIGIN':
            keypoints_mask = torch.ones(22*current_len)
        else :
            keypoints_mask = torch.ones(22)  

        return  video_name, \
                torch.FloatTensor(features), \
                torch.FloatTensor(keypoints_mask), \
                torch.FloatTensor(video_mask),  torch.FloatTensor(self.standard), current_len, label, video, self.standard_vid, subtraction
