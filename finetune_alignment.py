from models.AlignedT5 import AlignedT5
from models import load_checkpoint,save_checkpoint
from transformers import AutoTokenizer,get_linear_schedule_with_warmup
from utils.parser import parse_args
import torch.distributed as dist
import torch
from utils import seed_everything,get_lr
from datasets import construct_dataloader

def train():
    return

def main():
    args = parse_args()
    assert args.cfg_file, 'Config file not provided'
    model = AlignedT5(args.cfg_file)

    lr=1e-3
    warmup_steps=5000

    tokenizer = AutoTokenizer.from_pretrained('t5-base', use_fast=True)

    seed_everything(42)
    dist.init_process_group(backend='nccl', init_method='env://')
    torch.cuda.set_device(args.local_rank)
    model = model.cuda()
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank,find_unused_parameters=True)

    ## Wrong here, should change to finetune dataset
    train_dataloader = construct_dataloader('train',args.data,True,args.bs)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    schedular = get_linear_schedule_with_warmup(optimizer,num_warmup_steps=warmup_steps,num_training_steps=args.epochs*len(train_dataloader))


    start_epoch = load_checkpoint(args,model,optimizer)

    for epoch in range(start_epoch,args.epochs):
        ## sanity check first
        eval()

        train()



    return

if __name__ == "__main__":
    main()