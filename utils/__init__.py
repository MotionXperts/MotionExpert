import random
import numpy as np
import torch
import time

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def get_lr(optimizer):
    return [param_group["lr"] for param_group in optimizer.param_groups]

def time_elapsed(func):
    def wrapper(*args, **kwargs):
        # start_time = time.time()
        result = func(*args, **kwargs)
        # elapsed = time.time() - start_time
        # print(f"Elapsed time for {func.__name__}: {time.strftime('%H:%M:%S', time.gmtime(elapsed))}")
        return result
    return wrapper