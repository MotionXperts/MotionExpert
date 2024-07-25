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
    # Record the state.
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
    # logdir = '/home/c1l1mo/projects/MotionExpert/results/b2_para6' ## this line is for debugging
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
            ## sort the files in checkpoint dir
            if name is not None:
                checkpoint_path = name
                checkpoint = torch.load(checkpoint_path)
            else:
                checkpoint_path = natsorted(checkpoints)[-1]
                checkpoint = torch.load(os.path.join(checkpoint_dir,checkpoint_path))
            ## partially load, then print missing keys count
            model_keys = set(model.module.state_dict().keys())

            if not continue_training: ## the first time load from pretraining, discard transformation layer
                print("First time training")
                newckpt = {'model_state':{}}
                for k,v in checkpoint["model_state"].items():
                    if not 'transformation' in k and not 'projection' in k:
                        newckpt['model_state'][k] = v
            else: ## continue training from previous interruption, only discard projection layer
                newckpt = {'model_state':{}}
                for k,v in checkpoint["model_state"].items():
                    if not 'projection' in k:
                        newckpt['model_state'][k] = v

            loading_keys = set(newckpt["model_state"].keys())
            
            missing_keys = model_keys - loading_keys

            model.module.load_state_dict(newckpt["model_state"],strict=False)

            ## load alignment module weight
            if not continue_training and cfg.BRANCH ==2:
                ckpt_weight = load_alignment_checkpoint(cfg,model.module.align_module)
                align_module_keys = set([f'align_module.{x}' for x in model.module.align_module.state_dict().keys()])
                ## ensure the weights are correctly loaded.
                # for (k1,v1),(k2,v2) in zip(model.module.align_module.state_dict().items(),ckpt_weight):
                #     assert (torch.equal(v1,v2.to(v1.device))), f"Weight in {k1} and {k2} miss matched"
                missing_keys -= align_module_keys
                k = []
                for keys in missing_keys:
                    if not 'transformation' in keys:
                        k.append(keys)
            
            if continue_training or cfg.TASK.PRETRAIN:
                optimizer.load_state_dict(checkpoint["optimizer_state"])
            if dist.get_rank() == 0:
                print(f"LOADING CHECKPOINT AT {checkpoint_path}")
            if continue_training:
                return checkpoint["epoch"] 
            return 0
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
    ## no need to load projections here
    align_module.load_state_dict(newckpt["model_state"],strict=False)

    return checkpoint['model_state'].items()
