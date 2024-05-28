import os,sys
os.environ['NUMEXPR_MAX_THREADS'] = '16'
os.environ['TOKENIZERS_PARALLELISM'] = "false"
import warnings
warnings.filterwarnings("ignore",category=UserWarning)
warnings.filterwarnings("ignore",category=FutureWarning)
import torch
from torch.nn import functional as nnf
from transformers import T5ForConditionalGeneration, AutoConfig, AutoTokenizer, AdamW, get_linear_schedule_with_warmup
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import pickle , json, time , sys
from config import CONFIG
import torch.nn.functional as F
from typing import Optional

from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
import random
from dataloaders import construct_dataloader
import lightning as L
# from lightning.pytorch import seed_everything
from models.T5 import SimpleT5Model
from models.STAGCN import stagcn
from lightning.pytorch import seed_everything



## pytorch lightning tripled the memory usage :(
class STAGCN_T5(L.LightningModule):
    def __init__(self,T5,STAGCN,lr,batch_size,train_pickle_file,val_pickle_file,finetune=False,transform=False):
        super().__init__()
        self.T5 = T5
        self.STAGCN = STAGCN
        self.tokenizer = AutoTokenizer.from_pretrained('t5-base', use_fast=True)
        self.lr = lr
        self.batch_size = batch_size
        self.train_pickle_file = train_pickle_file
        self.val_pickle_file = val_pickle_file
        self.finetune = finetune
        self.transform = transform
    def forward(self,batch):
        video_name = batch['video_name']
        keypoints = batch['keypoints']
        keypoints_mask = batch['keypoints_mask']
        video_mask = batch['video_mask']
        label = batch['label']
        output = self.tokenizer(label, return_tensors="pt", padding="max_length", truncation=True, max_length=50)['input_ids']
        tgt_input = output[:, :-1]
        tgt_label = output[:, 1:]
        embedding, attention_node, attention_matrix = self.STAGCN(keypoints)
        return self.T5(inputs_embeds=embedding, attention_mask=video_mask,decoder_input_ids=tgt_input,labels=tgt_label)
    def training_step(self,batch,batch_idx):
        output = self.forward(batch)
        return output.loss
    def validation_step(self,batch,batch_idx):
        video_name = batch['video_name']
        keypoints = batch['keypoints']
        video_mask = batch['video_mask']
        label = batch['label']
        output = self.tokenizer(label, return_tensors="pt", padding="max_length", truncation=True, max_length=50)['input_ids']
        if not self.finetune:
            tgt_input = self.tokenizer("Motion Description : ", return_tensors="pt", padding="max_length", truncation=True, max_length=50)
            ## tgt_input = {"input_ids":tgt_input['input_ids'],"attention_mask":tgt_input['attention_mask']}
        else:
            tgt_input = self.tokenizer("Motion Instruction : ", return_tensors="pt", padding="max_length", truncation=True, max_length=50)

        embedding, attention_node, attention_matrix = self.STAGCN(keypoints)
        seqeunce = self.T5.generate(input_embeds=embedding, attention_mask=video_mask,
                            decoder_input_ids=tgt_input['input_ids'])
        print(self.tokenizer.decode(seqeunce[0], skip_special_tokens=True, clean_up_tokenization_spaces=True))
        return self.forward(batch).loss
    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.lr)
        schedular = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=5000, num_training_steps=10000)
        return [optimizer],[schedular]
    def train_dataloader(self):
        return construct_dataloader('train',self.train_pickle_file,self.tokenizer,self.finetune,self.batch_size,self.transform)
    def val_dataloader(self):
        return construct_dataloader('val',self.val_pickle_file,self.tokenizer,self.finetune,self.batch_size,self.transform)

def light_main(args,lr=1e-3,warmup_steps=5000):
    max_epochs = args.epochs

    ## setup output dir
    args.out_dir = os.path.join('results', args.out_dir)
    if os.path.exists(args.out_dir):
        pass
        # raise ValueError(f"Output directory {args.out_dir} already exists, use --args.out_dir to specify a new directory.")
    # os.makedirs(args.out_dir)

    ## construct models
    T5 = SimpleT5Model()
    STAGCN = stagcn()

    seed_everything(7,workers=True)
    model = STAGCN_T5(T5,STAGCN,lr,args.bs,args.data,args.test_data,args.finetune)
    trainer = L.Trainer(max_epochs=max_epochs,
                            devices="auto",sync_batchnorm=True,accelerator='cuda',strategy='fsdp',
                                    check_val_every_n_epoch=5,log_every_n_steps=1,
                                        default_root_dir=args.out_dir)
    trainer.fit(model)