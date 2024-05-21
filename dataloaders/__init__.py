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
    def collect_video_from_batch(batch):
        seq = []
        masks = []
        for b in batch:
            seq.append(b['keypoints'].permute(1,0,2)) # convert to F , coordinates, nodes 
            mask = torch.ones(b['keypoints'].shape[1])
            masks.append(mask)
        return seq,masks
    keypoints,masks = collect_video_from_batch(batch)
    padded_sequences = pad_sequence(keypoints,batch_first=True,padding_value=0) # B , F , coordinates, nodes
    input_mask = pad_sequence(masks,batch_first=True,padding_value=0) # B ,(max)F

    input_mask = input_mask.unsqueeze(2).unsqueeze(3)
    rank = torch.distributed.get_rank()
    true_seq = padded_sequences.masked_fill(input_mask==0,0)
    for index,t in enumerate(true_seq):
        tmp = np.average(t,axis=(1,2))[np.average(t,axis=(1,2))!=0]
        msk = (np.array(input_mask[index]))[(np.array(input_mask[index]))!=0]
        print(tmp.shape,msk.shape,'\n')

    padded_sequences = padded_sequences.permute(0,2,1,3) # convert to B , coordinates, F , nodes
    assert input_mask.size(0) == padded_sequences.size(0) and input_mask.size(1) == padded_sequences.size(2)
    return padded_sequences,input_mask

def construct_dataloader(split,pkl_file,finetune,batch_size,transform=None,cfg=None):
    if not finetune:
        if split == 'train':
            dataset = HumanMLDataset(pkl_file,finetune,transform)
            sampler = torch.utils.data.distributed.DistributedSampler(dataset,shuffle=True)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True,
                                            sampler=sampler,num_workers=8)
        elif split == "val":
            dataset = HumanMLDataset(pkl_file,finetune,transform,split='val')
            sampler = torch.utils.data.distributed.DistributedSampler(dataset,shuffle=False)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False,sampler=sampler,num_workers=8)
    else:
        if split == 'train':
            dataset = Skating(cfg,pkl_file,transform,split='train')
            sampler = torch.utils.data.distributed.DistributedSampler(dataset,shuffle=True)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True,
                                            sampler=sampler,num_workers=1,collate_fn=collate_fn)
        elif split == "val":
            dataset = Skating(cfg,pkl_file,transform,split='val')
            sampler = torch.utils.data.distributed.DistributedSampler(dataset,shuffle=False)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False,sampler=sampler,num_workers=8,collate_fn=collate_fn)
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
    data_loader = construct_dataloader('train','/home/weihsin/datasets/VQA/test_local.pkl',True,8,cfg = CONFIG)
    # Iterate over the DistributedDataLoader
    for padded_sequences, input_mask in data_loader:
        # print(f"Rank {rank}: Padded Sequences:")
        # print(padded_sequences)
        # print(f"Rank {rank}: Input Mask:")
        # print(input_mask)
        # print()
        break