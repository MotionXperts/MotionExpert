import os,sys
import torch.cuda
os.environ['NUMEXPR_MAX_THREADS'] = '2'
os.environ['TOKENIZERS_PARALLELISM'] = "false"
import warnings
warnings.filterwarnings("ignore",category=UserWarning)
warnings.filterwarnings("ignore",category=FutureWarning)
import torch
from transformers import AutoTokenizer, AdamW, get_linear_schedule_with_warmup
from utils.parser import parse_args,load_config
from tqdm import tqdm
import numpy as np
from pytorch_lightning.utilities.seed import seed_everything
import pickle , sys , logging
# Add videoalignment to sys path
sys.path.append(os.path.join('/home/weihsin/projects/MotionExpert','VideoAlignment'))
from evaluation import eval


from torch.utils.tensorboard import SummaryWriter
# Distributed Training
import torch.distributed as dist
from dataloaders import construct_dataloader
from models.T5 import SimpleT5Model
from models import save_checkpoint,load_checkpoint
import traceback
from datetime import timedelta

logger = logging.getLogger(__name__)

def train(cfg,train_dataloader, model, optimizer,scheduler,scaler,summary_writer,epoch,logger):
    model.train()
    optimizer.zero_grad()
    loss_list = []
    Tokenizer = AutoTokenizer.from_pretrained('t5-base', use_fast=True)
    if dist.get_rank() == 0:
        train_dataloader = tqdm(train_dataloader,total=len(train_dataloader), desc='Training')
    for index,batch in enumerate(train_dataloader):
        (video_name,src_batch,keypoints_mask_batch,standard,seq_len,label_batch) = batch
        model.zero_grad()
        optimizer.zero_grad()
        tgt_batch = Tokenizer(label_batch, return_tensors="pt", padding="max_length", truncation=True, max_length=50)['input_ids'].to(src_batch.device)
        tgt_input = tgt_batch[:, :-1]
        tgt_label = tgt_batch[:, 1:]
        with torch.cuda.amp.autocast():
            inputs = {  "video_name": video_name,
                        "input_embedding": src_batch.to(model.device),
                        "input_embedding_mask": keypoints_mask_batch.to(model.device),
                        "standard": standard.to(model.device),
                        "seq_len": seq_len.to(model.device),
                        "decoder_input_ids": tgt_input.to(model.device),
                        "labels": tgt_label.to(model.device)}
            '''
            ## branch 2
            if not ((hasattr(cfg,'BRANCH') and cfg.BRANCH == 1) or (cfg.TRANSFORMATION.REDUCTION_POLICY == 'TIME_POOL')):
                inputs['videos'] = videos.to(model.device)     
                inputs['standard_video'] = standard_video.to(model.device)
                inputs['subtraction'] = subtraction.to(model.device))
            '''
            outputs = model(**inputs)
            loss = outputs.loss
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        # Distributed Training
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
        logger.info(f"Epoch {epoch} : Loss {np.mean(loss_list)}")


def main():
    args = parse_args()
    cfg = load_config(args)

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(filename)s %(lineno)d: %(message)s', datefmt='%Y-%m-%d %H:%M:%S',
                            filename=os.path.join(cfg.LOGDIR,'stdout.log'))

    cfg.alignment_cfg = None

    if not cfg.TASK.PRETRAIN:
        assert hasattr(cfg,'BRANCH'), "BRANCH should be defined in config for finetuning."
        if cfg.BRANCH ==2:
            cfg.alignment_cfg = load_config(cfg.ALIGNMENT)

    
    model = SimpleT5Model(cfg)
    
    ## maintain a name list in main process
    with open(cfg.DATA.TEST, 'rb') as f:
        data = pickle.load(f)

    name_list = []
    for d in data:
        if d['video_name'] != 'standard':
            name_list.append(d['video_name'])    

    # Distributed Training
    dist.init_process_group(backend='nccl', init_method='env://')
    if dist.get_rank() == 0:
        store = dist.TCPStore("127.0.0.1", 8080, dist.get_world_size(), True,timedelta(seconds=30))
    else:
        store = dist.TCPStore("127.0.0.1", 8080, dist.get_world_size(), False,timedelta(seconds=30))
        
    seed_everything(42)
    
    # Distributed Training
    id = dist.get_rank()
    device = id % torch.cuda.device_count()
    model = model.to(device)
    torch.cuda.set_device(id)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device], output_device=device, find_unused_parameters=True)

    scaler = torch.cuda.amp.GradScaler()
    optimizer = AdamW(model.parameters(), lr=float(cfg.OPTIMIZER.LR))
    summary_writer = SummaryWriter(os.path.join(cfg.LOGDIR, 'train_logs'))

    train_dataloader =  construct_dataloader('train',cfg)
    test_dataloader  =  construct_dataloader('test' ,cfg)

    max_epoch = cfg.OPTIMIZER.MAX_EPOCH
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=cfg.OPTIMIZER.WARMUP_STEPS, num_training_steps=max_epoch * len(train_dataloader)
    )

    start_epoch = load_checkpoint(cfg,model,optimizer)
    try:
        # Sanity check
        eval(cfg, test_dataloader, model, start_epoch, summary_writer, True, store, name_list, logger)
        for epoch in range(start_epoch, max_epoch):

            # Distributed Training
            if dist.get_rank() == 0:
                logger.info(f"Training epoch {epoch}")
            
            train_dataloader.sampler.set_epoch(epoch)
            train(cfg,train_dataloader, model, optimizer,scheduler,scaler,summary_writer, epoch,logger)
            if (epoch+1) < 10 or (epoch+ 1) % 5 == 0:

                # Distributed Training
                if dist.get_rank() == 0:
                    os.makedirs(cfg.CKPTDIR,exist_ok=True)
                    save_checkpoint(cfg,model,optimizer,epoch+1)
                dist.barrier()

                try:
                    eval(cfg,test_dataloader, model,epoch,summary_writer,False, store=store,name_list=name_list,logger=logger)
                except Exception as e:
                    print(traceback.format_exc())
                    print(f"Error {e} \n in evaluation at epoch {epoch}, continuing training.")

            dist.barrier()

    except Exception as e:
        print(traceback.format_exc())
        print(f"{e} occured, saving model before quitting.")
    finally:
        if dist.get_rank() == 0 and epoch != start_epoch:
            save_checkpoint(cfg,model,optimizer,epoch+1)
        dist.destroy_process_group()
    
if __name__ == '__main__':
    main()