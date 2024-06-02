import torch
if __name__ == "__main__":
    import sys,os
    sys.path.append('/home/c1l1mo/projects/MotionExpert')
    ## path for VideoAlignment submodule
    sys.path.append(os.path.join(os.getcwd(),os.pardir,"VideoAlignment"))
from dataloaders.HumanML import HumanMLDataset
from dataloaders.Aligned_finetune import Skating
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import numpy as np

def collate_fn(batch):
    video_name, keypoints, keypoints_mask, video_mask, standard,seq_len, label, video,standard_video = zip(*batch)
    def collect_video_from_batch(batch):
        seq = []
        masks = []
        for b in batch:
            seq.append(b[1].permute(1,0,2)) # convert to F , coordinates, nodes 
            mask = torch.ones(b[1].shape[1])
            masks.append(mask)
        return seq,masks
    keypoints,masks = collect_video_from_batch(batch)
    padded_sequences = pad_sequence(keypoints,batch_first=True,padding_value=0) # B , F , coordinates, nodes
    input_mask = pad_sequence(masks,batch_first=True,padding_value=0) # B ,(max)F
    keypoints_mask = pad_sequence(keypoints_mask,batch_first=True,padding_value=0) 
    padded_videos = pad_sequence(video,batch_first=True,padding_value=0) # B , T , C , H , W

    input_mask = input_mask

    padded_sequences = padded_sequences.permute(0,2,1,3) # convert to B , coordinates, F , nodes
    assert input_mask.size(0) == padded_sequences.size(0) and input_mask.size(1) == padded_sequences.size(2)
 
    standard = torch.stack(standard,dim=0)
    seq_len = torch.stack(seq_len,dim=0)
    standard_video = torch.stack(standard_video,dim=0)
    return (video_name),padded_sequences,keypoints_mask, input_mask, (standard), (seq_len), (label), padded_videos, standard_video

def construct_dataloader(split,cfg):
    is_pretraining = cfg.TASK.PRETRAIN
    alignment_cfg=cfg.alignment_cfg
    if split == 'train' : 
        pkl_file = cfg.DATA.TRAIN
        batch_size = cfg.DATA.BATCH_SIZE
    elif split == 'val' :
        pkl_file = cfg.DATA.TEST
        batch_size = 1

    if is_pretraining:
        if split == 'train':
            dataset = HumanMLDataset(pkl_file,transformation_policy = cfg.TRANSFORMATION.REDUCTION_POLICY,split='train')
            sampler = torch.utils.data.distributed.DistributedSampler(dataset,shuffle=True)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True,
                                            sampler=sampler,num_workers=1,collate_fn=collate_fn)
        elif split == "val":
            dataset = HumanMLDataset(pkl_file,transformation_policy = cfg.TRANSFORMATION.REDUCTION_POLICY,split='val')
            sampler = torch.utils.data.distributed.DistributedSampler(dataset,shuffle=False)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False,sampler=sampler,num_workers=1,collate_fn=collate_fn)
    else:
        if split == 'train':
            dataset = Skating(alignment_cfg,pkl_file,transformation_policy = cfg.TRANSFORMATION.REDUCTION_POLICY,split='train')
            sampler = torch.utils.data.distributed.DistributedSampler(dataset,shuffle=True)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True,
                                            sampler=sampler,num_workers=1,collate_fn=collate_fn)
        elif split == "val":
            dataset = Skating(alignment_cfg,pkl_file,transformation_policy = cfg.TRANSFORMATION.REDUCTION_POLICY,split='val')
            sampler = torch.utils.data.distributed.DistributedSampler(dataset,shuffle=False)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False,sampler=sampler,num_workers=0,collate_fn=collate_fn)
    return dataloader


if __name__ == "__main__":
    from easydict import EasyDict as edict
    import torch.distributed as dist
    import yaml
    # Initialize distributed process group
    dist.init_process_group(backend='nccl', init_method='env://')
    rank = dist.get_rank()

    CONFIG = edict()
    with open("/home/c1l1mo/projects/VideoAlignment/result/scl_skating_long_50_512/config.yaml") as f:
        cfg = yaml.safe_load(f)
    CONFIG.update(cfg)
    data_loader = construct_dataloader('train','/home/weihsin/datasets/train_Axel_520_with_standard.pkl',False,8,alignment_cfg = CONFIG)
    # Iterate over the DistributedDataLoader
    for sample in data_loader:
        # print(f"Rank {rank}: Padded Sequences:")
        # print(padded_sequences)
        # print(f"Rank {rank}: Input Mask:")
        # print(input_mask)
        # print()
        break