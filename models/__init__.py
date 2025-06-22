import os, torch, re
from natsort import natsorted
import torch.distributed as dist
import loralib as lora

def save_checkpoint(cfg, model, optimizer, epoch) :
    if not cfg.TASK.PRETRAIN :
        path = os.path.join(cfg.LOGDIR, "checkpoints")
    else :
        path = os.path.join(cfg.LOGDIR, "pretrain_checkpoints")
    os.makedirs(path, exist_ok = True)
    ckpt_path = os.path.join(path, "checkpoint_epoch_{:05d}.pth".format(epoch))

    checkpoint = {"epoch" : epoch,
                  "model_state" : model.module.state_dict(),
                  "lora_state" : lora.lora_state_dict(model.module),
                  "optimizer_state" : optimizer.state_dict()}

    if not os.path.exists(ckpt_path) :
        torch.save(checkpoint, ckpt_path)
        print(f"Saving epoch {epoch} checkpoint at {ckpt_path}")

def load_checkpoint(cfg, model, optimizer, name = None) :
    continue_training = False
    checkpoint_num = 0
    logdir = cfg.LOGDIR
    if dist.get_rank() == 0 :
        print("LOGDIR : ", logdir)

    # Load Model
    if (not cfg.TASK.PRETRAIN) :
        # Find pretrain path.
        pretrain_path = cfg.WEIGHT_PATH
        pretrain_checkpoint = torch.load(pretrain_path)

        # Load weights that were previously pre-trained.
        # pretrain_total_params = sum(v.numel() for v in pretrain_checkpoint["model_state"].values())
        # print("Pretrain Parameter Number", pretrain_total_params)
        model.module.load_state_dict(pretrain_checkpoint["model_state"], strict = False)
        '''
        print("Loaded pretrain parameters : ")
        for name in pretrainckpt["model_state"].keys() :
            print(name)
        print("############################")
        '''
        # Continue training.
        if (os.path.exists(os.path.join(logdir, 'checkpoints')) and 
            len(os.listdir(os.path.join(logdir, 'checkpoints'))) > 0) :
            # Find finetune path.
            checkpoint_dir = os.path.join(logdir, "checkpoints")
            checkpoints = os.listdir(checkpoint_dir)
            checkpoint_path = natsorted(checkpoints)[-1]
            print("Continue training - CHECKPOINT PATH : ", checkpoint_path)

            checkpoint = torch.load(os.path.join(checkpoint_dir, checkpoint_path))
            base_name = os.path.basename(checkpoint_path)
            checkpoint_num = int(re.search(r'\d+', base_name).group())
            # Load weights that were previously fine-tuned.
            model.module.load_state_dict(checkpoint["model_state"], strict = True)

        # If the checkpoint path is 'None', it indicates that the process is training rather than evaluation.
        if (cfg.EVAL.ckpt != "None") :
            # Evaluation
            checkpoint_path = cfg.EVAL.ckpt
            dist.barrier()
            print("checkpoint_path", checkpoint_path)
            checkpoint = torch.load(checkpoint_path)
            base_name = os.path.basename(checkpoint_path)
            try:
                checkpoint_num = int(re.search(r'\d+', base_name).group())
            except Exception as e:
                checkpoint_num = "demo"
            model.module.load_state_dict(checkpoint["model_state"], strict = False)

            # Load LoRA state
            new_state_dict = {f"module.{k}" : v for k, v in checkpoint['lora_state'].items()}
            model.load_state_dict(new_state_dict, strict = False)
            # model.load_state_dict(checkpoint['lora_state_none'], strict = True)

    # Set parameters become trainable.
    '''
    lora.mark_only_lora_as_trainable(model)
    for name, param in model.named_parameters() :
        if 't5' in name.lower() :
            # Enable gradient updates for T5-related parameters
            param.requires_grad = True
    '''
    lora_params = [param for name, param in model.named_parameters() if param.requires_grad and "lora" in name]
    lora_total_params = sum(p.numel() for p in lora.lora_state_dict(model).values())
    print("Lora Parameter Number", lora_total_params)
    '''
    base_params = [param for name, param in model.named_parameters() if param.requires_grad and "lora" not in name]
    '''
    model_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Trainable Parameter Number", model_trainable_params)

    # Print parameter names of LoRA.
    '''
    lora_params_names = [name for name, param in model.named_parameters() if param.requires_grad and "lora" in name]
    print("LoRA Parameter Names : ")
    for name in lora_params_names :
        print(name)
    '''
    # Print parameter names of Base Model parameter.
    '''
    print("##############################")
    base_params_names =  [name for name, param in model.named_parameters() if not param.requires_grad]
    print("Base Parameter Name")
    for name in base_params_names :
        print(name)
    '''
    print("Load checkpoint number", checkpoint_num)
    return checkpoint_num

def load_alignment_checkpoint(cfg, align_module) :
    checkpoint_dir = os.path.join(cfg.alignment_cfg.LOGDIR, "checkpoints")
    checkpoints = os.listdir(checkpoint_dir)
    checkpoint_path = natsorted(checkpoints)[-1]
    checkpoint = torch.load(os.path.join(checkpoint_dir, checkpoint_path))
    print("LOADING ALIGNMENT CHECKPOINT AT : ", checkpoint_path)

    newckpt = {'model_state' : {}}
    for k, v in checkpoint["model_state"].items() :
        if not 'projection' in k :
            newckpt['model_state'][k] = v
    # No need to load projections here.
    align_module.load_state_dict(newckpt["model_state"], strict = False)

    return checkpoint['model_state'].items()