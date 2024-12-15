import torch
if __name__ == "__main__":
    import sys,os
    sys.path.append('/home/c1l1mo/projects/MotionExpert')
    ## path for VideoAlignment submodule
    sys.path.append(os.path.join(os.getcwd(),os.pardir,"VideoAlignment"))
from dataloaders.Dataset import DatasetLoader
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import numpy as np

def collate_fn(batch):
    video_name, keypoints, keypoints_mask, standard, seq_len, label, subtraction = zip(*batch)
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
    return (video_name), padded_keypoints, keypoints_mask, (padded_standard), (seq_len), (label), subtraction

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
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False,sampler=sampler,collate_fn=collate_fn)

    return dataloader

if __name__ == "__main__":
    from easydict import EasyDict as edict
    import torch.distributed as dist
    import yaml

    dist.init_process_group(backend='nccl', init_method='env://')
    rank = dist.get_rank()

    CONFIG = edict()
    with open("/home/c1l1mo/projects/VideoAlignment/result/scl_skating_long_50_512/config.yaml") as f:
        cfg = yaml.safe_load(f)
    CONFIG.update(cfg)
    data_loader = construct_dataloader('train','/home/weihsin/datasets/train_Axel_520_with_standard.pkl',False,8,alignment_cfg = CONFIG)
