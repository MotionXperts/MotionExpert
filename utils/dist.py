import torch.distributed as dist

def is_root_proc():
    return dist.get_rank() == 0