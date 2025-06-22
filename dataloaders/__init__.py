import torch, numpy as np
from dataloaders.Dataset import DatasetLoader
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
def covert_skeleton(skeleton_coords):
    # Convert to [number of frame , coordinates(6), joints(22)].
    return [skeleton.permute(1, 0, 2) for skeleton in skeleton_coords]

def collate_fn(batch):
    video_name, skeleton_coords, seq_len, frame_mask, label, labels, std_coords, subtraction = zip(*batch)
    skeleton_coords = covert_skeleton(skeleton_coords)

    # The dimension of padded_skeleton_coords is [B, F, coordinates, nodes].
    padded_skeleton_coords = pad_sequence(skeleton_coords, batch_first = True, padding_value = 0)
    frame_mask = pad_sequence(frame_mask, batch_first = True, padding_value = 0)

    # Convert to [Batch size, coordinates(6), number of frame, joints(22)].
    padded_skeleton_coords = padded_skeleton_coords.permute(0, 2, 1, 3)

    standard = covert_skeleton(std_coords)
    # The dimension of padded_standard is [B, F, coordinates, nodes].
    padded_standard = pad_sequence(standard, batch_first = True, padding_value = 0)
    padded_standard = padded_standard.permute(0, 2, 1, 3)

    subtraction = pad_sequence(subtraction, batch_first = True, padding_value = 0)

    seq_len = [s.item() if isinstance(s, torch.Tensor) else s for s in seq_len]
    # Change standard to padded_standard.
    return video_name, padded_skeleton_coords.float(), seq_len, frame_mask.float(), label, labels, padded_standard.float(), subtraction.float()

def construct_dataloader(split,cfg,pkl_file):
    if split == 'train' :
        batch_size = cfg.DATA.BATCH_SIZE
    elif split == 'test' :
        batch_size = cfg.DATA.BATCH_SIZE
        if(not cfg.TASK.PRETRAIN):
            # Batch Size must be set to 1 for visualization BertViz.
            batch_size = 1

    if split == 'train':
        # Distributed Training
        dataset = DatasetLoader(cfg, cfg.TASK.PRETRAIN, pkl_file, train=True)
        sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=True)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True, sampler=sampler, collate_fn=collate_fn)
    elif split == "test":
        # Distributed Training
        dataset = DatasetLoader(cfg, cfg.TASK.PRETRAIN, pkl_file, train=False)
        sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=False)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True, sampler=sampler, collate_fn=collate_fn, num_workers=0)

    return dataloader
