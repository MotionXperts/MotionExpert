import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import torch
from torch.nn import functional as nnf
from transformers import T5ForConditionalGeneration, AutoConfig, AutoTokenizer, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
from transformers.data.data_collator import DataCollatorForSeq2Seq
import argparse
from tqdm import tqdm
import sys
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import pickle
from torch.nn.utils.rnn import pad_sequence
import json
import time
from cider import readJSON, readPickle, getGTCaptions, BLEUScore, CIDERScore
from config import CONFIG
################ 1228 weihsin ST-GCN
from net.utils.tgcn import ConvTemporalGraphical
from net.st_gcn import Model as st_gcn 
from net.utils.graph import Graph
from torch import Tensor
import torch.nn.functional as F
from typing import Optional
############### ST-GCN

class HumanMLDataset(Dataset):
    def __init__(self, pkl_file, tokenizer, transform=None):
        with open(pkl_file, 'rb') as f:
            self.data_list = pickle.load(f)
        self.samples = []
        max_len = 0  
        for item in self.data_list:
            features = item['features']
            max_len = max(max_len, len(features)) 
            video_name = item['video_name']
            for label in item['labels']:
                self.samples.append((features, label, video_name))
        self.max_len = max_len  
        self.tokenizer = tokenizer
        self.transform = transform
        
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        features, label, video_name = self.samples[idx]
        #print(label)

        padded_features = np.zeros((self.max_len, 66))
        keypoints_mask = np.zeros(self.max_len) 

        current_len = len(features)
        padded_features[:current_len, :] = features
        keypoints_mask[:current_len] = 1  
        tokenized_label = self.tokenizer(label, return_tensors="pt", padding="max_length", truncation=True, max_length=50)
        #print(tokenized_label)
        #print(self.tokenizer.decode(tokenized_label['input_ids'].squeeze(0)))

        sample = {
            "video_name": video_name,
            "keypoints": torch.FloatTensor(padded_features),
            "keypoints_mask": torch.FloatTensor(keypoints_mask),
            "label": label,
            "output": tokenized_label['input_ids'].squeeze(0),
        }
        return sample
    
class HumanMLDataset_val(Dataset):
    def __init__(self, pkl_file, tokenizer, transform=None):
        with open(pkl_file, 'rb') as f:
            self.data_list = pickle.load(f)
        self.tokenizer = tokenizer
        self.transform = transform
        self.max_len = max(len(item['features']) for item in self.data_list)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        item = self.data_list[idx]
        features, video_name = item['features'], item['video_name']

        padded_features = np.zeros((self.max_len, 66))
        keypoints_mask = np.zeros(self.max_len) 

        current_len = len(features)
        padded_features[:current_len, :] = features
        keypoints_mask[:current_len] = 1  

        sample = {
            "video_name": video_name,
            "keypoints": torch.FloatTensor(padded_features),
            "keypoints_mask": torch.FloatTensor(keypoints_mask),
        }
        return sample

class SimpleT5Model(nn.Module):

    def __init__(self, embed_size=384):
        super(SimpleT5Model, self).__init__()
        config = AutoConfig.from_pretrained('t5-large')
        self.t5 = T5ForConditionalGeneration.from_pretrained('t5-large', config=config)

        self.embedding = nn.Sequential(nn.Linear(66,embed_size), nn.ReLU(),nn.Linear(embed_size,1024))
        
        #### ST-GCN's GCN block ###
        #### SETTING ###
        self.out_channel = CONFIG.OUT_CHANNEL 
        self.graph = Graph(layout = 'SMPL')
        self.A     = torch.from_numpy(self.graph.A.astype(np.float32))
        self.STGCN = st_gcn(in_channels=3,edge_importance_weighting= True,graph_args= {'layout': 'SMPL', 'strategy': 'spatial'},num_class = 400)

    def _get_embeddings(self, src): ########################################################## Step 3
        src = self.STGCN(src)
        src = src.permute(0, 2, 1,3)
        return src

    def _get_encoder_feature(self, src):
        ori_src = src.clone()
        #print("Before : _get_encoder_feature_src",src.size()) 
        src  = self._get_embeddings(src)
        #print("After : _get_encoder_feature_src",src.size())
        
        src = src.reshape((src.size(0), src.size(1), src.size(2) * src.size(3)))
        self.embedding =    nn.Sequential(
                                nn.Linear(src.size(2),512), 
                                nn.ReLU(),
                                nn.Linear(512,512)).to(src.get_device())
        src = self.embedding(src)

        self.ori_embedding = nn.Sequential(
                                nn.Linear(66,512), 
                                nn.ReLU(),
                                nn.Linear(512,512)).to(src.get_device())

        ori_src = self.ori_embedding(ori_src.view(-1,ori_src.size(2))).view(ori_src.size(0), ori_src.size(1), -1)
        #print("ori_src size",ori_src.size())
        #print("src size",src.size(1))
        self.con1d = nn.Conv1d(in_channels=ori_src.size(1), out_channels=src.size(1), kernel_size=1,padding = 0).to(src.get_device())
        batch_size , seq_length , feature_dim = ori_src.size()
        ori_src = ori_src.permute(1,0,2)
        ori_src = ori_src.reshape(ori_src.size(0),-1)
        #print("ori_src size",ori_src.size())
        ori_src = self.con1d(ori_src)

        #print("ori_src size",ori_src.size())
        # ori_embedding convolution 
        ori_src = ori_src.reshape(src.size(1),batch_size,feature_dim)
        ori_src = ori_src.permute(1,0,2)
        # torch concate 
        src = torch.cat((src,ori_src),2)
        #print("after cat : src",src.size())
        return src    

    def forward(self, input_ids, attention_mask, decoder_input_ids=None, labels=None, use_embeds=True):
        if use_embeds:
            batch_size, seq_length, feature_dim = input_ids.shape
            
            #input_embeds = self.embedding(input_ids.view(-1,feature_dim)).view(batch_size, seq_length, -1)
            #output = self.t5(inputs_embeds=input_embeds, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids, labels=labels)
            ############################################# ST-GCN
            input_embeds = self._get_encoder_feature(input_ids)
            new_attentention_mask = attention_mask[:,::4].clone()
            attention_mask = new_attentention_mask
            output = self.t5(inputs_embeds=input_embeds, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids, labels=labels)
            ###############################################
        else:
            output = self.t5(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids, labels=labels)

        return output

def train(train_dataset, model, tokenizer, args, eval_dataset=None, lr=1e-3, warmup_steps=5000, output_dir=".", output_prefix=""):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = args.bs
    epochs = args.epochs

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model.to(device)
    model.train()
    optimizer = AdamW(model.parameters(), lr=lr)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=epochs * len(train_dataloader)
    )
    for epoch in range(epochs):
        print(f">>> Training epoch {epoch}")
        sys.stdout.flush()
        progress = tqdm(total=len(train_dataloader), desc=output_prefix)
        loss_list = []
        for idx, batch in enumerate(train_dataloader):
            model.zero_grad()
            video_names = batch['video_name']
            src_batch = batch['keypoints'].to(device)
            keypoints_mask_batch = batch['keypoints_mask'].to(device)
            tgt_batch = batch['output'].to(device)

            tgt_input = tgt_batch[:, :-1]
            tgt_labels = tgt_batch[:, 1:]

            outputs = model(input_ids=src_batch.contiguous(), 
                            attention_mask=keypoints_mask_batch.contiguous(), 
                            decoder_input_ids=tgt_input.contiguous(),
                            labels=tgt_labels.contiguous())

            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            loss_list.append(loss.item())
            progress.set_postfix({
                'loss': np.mean(loss_list),
                'lr': scheduler.optimizer.param_groups[0]['lr'],
            })
            progress.update()

        if eval_dataset is not None:
            print(f"Epoch {epoch}: Train Loss: {np.mean(loss_list):.4f}")
            torch.save(model.state_dict(), os.path.join(output_dir, f"{output_prefix}_epoch{epoch}.pt"))            
            model.eval()
            val_data_loader = DataLoader(eval_dataset, batch_size=2, shuffle=False)
            results = {}
            start_token_id = 3
            beam_size = 3
            with torch.no_grad():
                for batch in tqdm(val_data_loader):
                    video_names = batch['video_name']
                    src_batch = batch['keypoints'].to(device)
                    keypoints_mask_batch = batch['keypoints_mask'].to(device)

                    # input_embeds = model.embedding(src_batch)
                    ## FIXME: use ST-GCN
                    input_embeds = model._get_encoder_feature(src_batch)
                    decoder_input_ids = torch.tensor([[start_token_id]] * src_batch.shape[0]).to(device)
                    new_attentention_mask = keypoints_mask_batch[:,::4].clone()
                    keypoints_mask_batch = new_attentention_mask
                    generated_ids = model.t5.generate(
                        inputs_embeds=input_embeds,
                        attention_mask=keypoints_mask_batch,
                        decoder_input_ids=decoder_input_ids, 
                        max_length=50,
                        num_beams=beam_size,
                        repetition_penalty=2.5,
                        length_penalty=1.0,
                        early_stopping=True
                    )

                    for name, gen_id in zip(video_names, generated_ids):
                        decoded_text = tokenizer.decode(gen_id, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                        results[name] = decoded_text
            with open('STGCN_results_epoch'+str(epoch)+'.json', 'w') as f:
                json.dump(results, f)

            predictions = readJSON('STGCN_results_epoch'+str(epoch)+'.json')
            annotations = readPickle('/home/peihsin/projects/humanML/dataset/rm_test.pkl')
            gts = getGTCaptions(annotations)
            #Check predictions content is correct
            assert type(predictions) is dict
            assert set(predictions.keys()) == set(gts.keys())
            assert all([type(pred) is str for pred in predictions.values()])
            # CIDErScore
            cider_score = CIDERScore()(predictions, gts)
            bleu_score = BLEUScore()(predictions, gts)
            print(f"CIDEr: {cider_score}")
            print(f"BLEU: {bleu_score}")
        
        progress.close()

    return model


def main():
    parser = argparse.ArgumentParser()
    #parser.add_argument('--data', default='/home/peihsin/projects/humanML/dataset/train.pkl')
    parser.add_argument('--data', default='/home/peihsin/projects/humanML/dataset/rm_train.pkl')
    parser.add_argument('--out_dir', default='./models')
    parser.add_argument('--prefix', default='HumanML', help='prefix for saved filenames')
    parser.add_argument('--epochs', type=int, default=3000)
    parser.add_argument('--save_every', type=int, default=1)
    parser.add_argument('--bs', type=int, default=8)
    #parser.add_argument('--test_data', default='./test.pkl') 
    parser.add_argument('--test_data', default='/home/peihsin/projects/humanML/dataset/rm_test.pkl')
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained('t5-large', use_fast=True)
    dataset = HumanMLDataset(args.data, tokenizer)
    eval_dataset = HumanMLDataset_val(args.test_data, tokenizer)  
    model = SimpleT5Model()
    #print(model)
    #weight = './models/HumanML_epoch0.pt'
    #model.load_state_dict(torch.load(weight))

    train(dataset, model, tokenizer, args,eval_dataset=eval_dataset, output_dir=args.out_dir, output_prefix=args.prefix)


if __name__ == '__main__':
    main()