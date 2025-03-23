import torch
from dataloaders.Dataset import DatasetLoader
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import numpy as np

def collate_fn(batch):
    video_name, keypoints, keypoints_mask, standard, seq_len, label, subtraction, labels = zip(*batch)
    def collect_video_from_batch(batch, idx=1):
        seq = []
        # convert to [number of frame , coordinates(6), joints(22)]
        for b in batch:
            seq.append(b[idx].permute(1,0,2)) 
        return seq

    keypoints = collect_video_from_batch(batch)

    padded_keypoints = pad_sequence(keypoints,batch_first=True,padding_value=0) # B , F , coordinates, nodes
    keypoints_mask = pad_sequence(keypoints_mask,batch_first=True,padding_value=0) 

    # convert to [Batch size, coordinates(6), number of frame, joints(22)]
    padded_keypoints = padded_keypoints.permute(0,2,1,3) 

    # standard = torch.stack(standard,dim=0)
    standard = collect_video_from_batch(batch, 3)
    padded_standard = pad_sequence(standard,batch_first=True,padding_value=0) # B , F , coordinates, nodes
    
    seq_len = torch.stack(seq_len,dim=0)
    subtraction =pad_sequence(subtraction,batch_first=True,padding_value=0) 
    # change standard to padded_standard
    return (video_name), padded_keypoints, keypoints_mask, (padded_standard), (seq_len), (label), subtraction, labels

def construct_dataloader(split,cfg,pkl_file):
    if split == 'train' : 
        batch_size = cfg.DATA.BATCH_SIZE
    elif split == 'test' :
        batch_size = cfg.DATA.BATCH_SIZE
        if(not cfg.TASK.PRETRAIN):
            batch_size = 1

    dataset = DatasetLoader(cfg,cfg.TASK.PRETRAIN,pkl_file)

    if split == 'train':
        # Distributed Training
        sampler = torch.utils.data.distributed.DistributedSampler(dataset,shuffle=True)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True, sampler=sampler,collate_fn=collate_fn)
    elif split == "test":
        # Distributed Training
        sampler = torch.utils.data.distributed.DistributedSampler(dataset,shuffle=False)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True,sampler=sampler,collate_fn=collate_fn, num_workers=0)

    return dataloader
