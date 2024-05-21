import os,sys
os.environ['NUMEXPR_MAX_THREADS'] = '16'
os.environ['TOKENIZERS_PARALLELISM'] = "false"
import warnings
warnings.filterwarnings("ignore",category=UserWarning)
warnings.filterwarnings("ignore",category=FutureWarning)
import torch
from torch.nn import functional as nnf
from transformers import AutoTokenizer, AdamW, get_linear_schedule_with_warmup
from utils.parser import parse_args
from tqdm import tqdm
import numpy as np
from utils import seed_everything,get_lr
import pickle , sys , logging
from evaluate import eval

from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
import random
from dataloaders import construct_dataloader
from models.T5 import SimpleT5Model
from models import save_checkpoint,load_checkpoint
import traceback


logger = logging.getLogger(__name__)

def train(args,train_dataloader, model, optimizer,scheduler,tokenizer,scaler,summary_writer,epoch,logger):
    model.train()
    optimizer.zero_grad()
    loss_list = []

    if dist.get_rank() == 0:
        train_dataloader = tqdm(train_dataloader,total=len(train_dataloader), desc=args.prefix)
    for idx, batch in enumerate(train_dataloader):
        model.zero_grad()
        optimizer.zero_grad()
        video_names = batch['video_name']
        src_batch = batch['keypoints']
        keypoints_mask_batch = batch['keypoints_mask'].to(src_batch.device)
        video_mask_batch = batch['video_mask'].to(src_batch.device)
        label = batch['label']
        tgt_batch = tokenizer(label, return_tensors="pt", padding="max_length", truncation=True, max_length=50)['input_ids'].to(src_batch.device)

        tgt_input = tgt_batch[:, :-1]
        tgt_label = tgt_batch[:, 1:]

        with torch.cuda.amp.autocast():
            outputs = model(keypoints=src_batch.contiguous(), 
                            keypoints_mask=keypoints_mask_batch.contiguous(), 
                            video_mask= video_mask_batch.contiguous(),
                            decoder_input_ids=tgt_input.contiguous(),         # text
                            labels=tgt_label.contiguous())                        
            loss = outputs.loss
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        scheduler.step()

        loss[torch.isnan(loss)] = 0
        dist.all_reduce(loss, async_op=False)
        reduced_loss = loss / dist.get_world_size()
        loss_list.append(reduced_loss.detach().cpu())
        if dist.get_rank() == 0:
            train_dataloader.set_postfix({
                'loss': np.mean(loss_list),
                'lr': scheduler.optimizer.param_groups[0]['lr'],
            })
    if dist.get_rank() == 0:
        summary_writer.add_scalar('train/loss', np.mean(loss_list), epoch)
        summary_writer.add_scalar('train/learning_rate', get_lr(optimizer)[0], epoch)
        logger.info(f"Epoch {epoch} : Loss {np.mean(loss_list)}")
    
def main():
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained('t5-base', use_fast=True)
    model = SimpleT5Model()
    lr=1e-3
    warmup_steps=5000
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(lineno)d: %(message)s', datefmt='%Y-%m-%d %H:%M:%S',
                            filename=os.path.join(args.out_dir,'stdout.log'))
    
    ## maintain a name list in main process
    with open(args.test_data, 'rb') as f:
        data = pickle.load(f)
    name_list = []
    for d in data:
        name_list.append(d['video_name'])

    dist.init_process_group(backend='nccl', init_method='env://')

    if dist.get_rank() == 0:
        store = dist.TCPStore("127.0.0.1", 1234, dist.get_world_size(), True)
    else:
        store = dist.TCPStore("127.0.0.1", 1234, dist.get_world_size(), False)

    seed_everything(42)
    
    torch.cuda.set_device(args.local_rank)
    model = model.cuda()
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                    output_device=args.local_rank, find_unused_parameters=False)
    scaler = torch.cuda.amp.GradScaler()
    optimizer = AdamW(model.parameters(), lr=lr)
    summary_writer = SummaryWriter(os.path.join(args.out_dir, 'train_logs'))

    
    train_dataloader = construct_dataloader('train',args.data,args.finetune,args.bs)
    val_dataloader = construct_dataloader('val',args.test_data,args.finetune,21)

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=args.epochs * len(train_dataloader)
    )

    start_epoch = load_checkpoint(args,model,optimizer)
    
    try:
        ## sanity check
        eval(args,val_dataloader, model, tokenizer,start_epoch,summary_writer,sanity_check=True,
                            store=store,name_list=None,logger=logger)
        print(f"{args.local_rank} Sanity check passed")

        for epoch in range(start_epoch, args.epochs):
            if dist.get_rank() == 0:
                logger.info(f"Training epoch {epoch}")
            train_dataloader.sampler.set_epoch(epoch)
            train(args,train_dataloader, model, optimizer,scheduler,tokenizer,scaler,summary_writer,epoch,logger)
            if (epoch + 1) % 1 == 0:
                if dist.get_rank() == 0:
                    save_checkpoint(args,model,optimizer,epoch+1)
                if (epoch+1) % 1 == 0:
                    try:
                        eval(args,val_dataloader, model, tokenizer,epoch,summary_writer,sanity_check=False,
                                store=store,name_list=name_list,logger=logger)
                    except Exception as e:
                        print(traceback.format_exc())
                        print(f"Error {e} \n in evaluation at epoch {epoch}, continuing training.")
            dist.barrier()
    except Exception as e:
        print(traceback.format_exc())
        print(f"{e} occured, saving model before quitting.")
    finally:
        if dist.get_rank() == 0 and epoch != start_epoch:
            save_checkpoint(args,model,optimizer,epoch+1)
        dist.destroy_process_group()
    
if __name__ == '__main__':
    main()