import os
import torch
from natsort import natsorted
import torch.distributed as dist

def save_checkpoint(cfg, model, optimizer, epoch):
    if not cfg.TASK.PRETRAIN:
        path = os.path.join(cfg.LOGDIR, "checkpoints")
    else:
        path = os.path.join(cfg.LOGDIR, "pretrain_checkpoints")
    os.makedirs(path, exist_ok=True)
    ckpt_path = os.path.join(path, "checkpoint_epoch_{:05d}.pth".format(epoch))
    # Record the state
    checkpoint = {
        "epoch": epoch,
        "model_state": model.module.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }
    if not os.path.exists(ckpt_path):
        torch.save(checkpoint, ckpt_path)
        print(f"Saving epoch {epoch} checkpoint at {ckpt_path}")

def load_checkpoint(cfg,model,optimizer,name=None):
    continue_training = False
    logdir = cfg.LOGDIR 
    print("LOGDIR: ",logdir)

    if os.path.exists(os.path.join(logdir,'checkpoints')) and len(os.listdir(os.path.join(logdir,'checkpoints')))>0:
        continue_training = True
    if continue_training:
        checkpoint_dir = os.path.join(logdir, "checkpoints")
    else:
        checkpoint_dir = os.path.join(logdir, "pretrain_checkpoints")
        # os.makedirs(checkpoint_dir, exist_ok=True)
        # assert os.path.exists(checkpoint_dir), f"Checkpoint dir {checkpoint_dir} not found"
    if os.path.exists(checkpoint_dir):
        checkpoints = os.listdir(checkpoint_dir)
        if len(checkpoints) > 0:
            # Sort the files in checkpoint dir
            if name is not None:
                checkpoint_path = name
                checkpoint = torch.load(checkpoint_path)
            else:
                checkpoint_path = natsorted(checkpoints)[-1]
                checkpoint = torch.load(os.path.join(checkpoint_dir,checkpoint_path))
        
            newckpt = {'model_state':{}}
            for k,v in checkpoint["model_state"].items():
                if not 'projection' in k:
                    newckpt['model_state'][k] = v

            model.module.load_state_dict(newckpt["model_state"],strict=False)
            # Use for check the pretrain weight loaded correctly
            # print(newckpt["model_state"].keys())
            if continue_training or cfg.TASK.PRETRAIN:
                optimizer.load_state_dict(checkpoint["optimizer_state"])
            # Distributed Training
            if dist.get_rank() == 0:
                print(f"LOADING CHECKPOINT AT {checkpoint_path}")
            if continue_training:
                return checkpoint["epoch"] 
            return 0
    # Distributed Training
    if dist.get_rank() == 0:
        print("CHECKPOINT NOT FOUND")
    return 0 

def load_alignment_checkpoint(cfg,align_module):
    checkpoint_dir = os.path.join(cfg.alignment_cfg.LOGDIR, "checkpoints")
    assert os.path.exists(checkpoint_dir), f"Checkpoint dir {checkpoint_dir} not found"
    checkpoints = os.listdir(checkpoint_dir)
    checkpoint_path = natsorted(checkpoints)[-1]
    checkpoint= torch.load(os.path.join(checkpoint_dir,checkpoint_path))
    print("LOADING ALIGNMENT CHECKPOINT AT: ",checkpoint_path)

    newckpt = {'model_state':{}}
    for k,v in checkpoint["model_state"].items():
        if not 'projection' in k:
            newckpt['model_state'][k] = v
    # No need to load projections here
    align_module.load_state_dict(newckpt["model_state"],strict=False)

    return checkpoint['model_state'].items()
