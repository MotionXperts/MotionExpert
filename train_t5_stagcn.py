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
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import pickle , json, time , sys
from torch.nn.utils.rnn import pad_sequence
from cider import readJSON, readPickle, getGTCaptions, BLEUScore, CIDERScore
from config import CONFIG
from torch import Tensor
import torch.nn.functional as F
from typing import Optional
#################################################### ST-GCN
#################################################### STAGCN
from net.st_agcn import STA_GCN as st_agcn
from net.Utils_attention.attention_branch import *
from net.Utils_attention.perception_branch import *
from net.Utils_attention.feature_extractor import *
from net.Utils_attention.graph_convolution import *
#################################################### STAGCN
bonelink = [(0, 1), (0, 2), (0, 3), (1, 4), (2,5), (3,6), (4, 7), (5, 8), (6, 9), (7, 10), (8, 11), (9, 12), (9, 13), (9, 14), (12, 15), (13, 16), (14, 17), (16, 18), (17, 19), (18,  20), (19, 21)]
def generate_data(current_keypoints ):
    joint_coordinate = np.zeros((3,len(current_keypoints),22))
    bone_coordinate = np.zeros((3,len(current_keypoints),22))
    for i in range(len(current_keypoints)):
        for j in range(0,len(current_keypoints[i]),3):
            joint_coordinate[:, i, j//3] = current_keypoints[i,j:j+3]
    for v1, v2 in bonelink:
        bone_coordinate[:, :, v1] = joint_coordinate[:, :, v1] - joint_coordinate[:, :, v2]
    coordinates = np.concatenate((joint_coordinate, bone_coordinate), axis=0)
    return coordinates

class HumanMLDataset(Dataset):
    def __init__(self, pkl_file, tokenizer, transform=None):
        with open(pkl_file, 'rb') as f:
            self.data_list = pickle.load(f)
        self.samples = []
        max_len = 0  

        for item in self.data_list:
            features =  generate_data(item['features'])
            max_len = max(max_len, len(features[0]))
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
        padded_features = np.zeros((6,self.max_len, 22)) 
        keypoints_mask = np.zeros((6,self.max_len))       

        current_len = len(features[0])
        padded_features[:,:current_len, :] = features
        keypoints_mask[:,:current_len] = 1  
        tokenized_label = self.tokenizer(label, return_tensors="pt", padding="max_length", truncation=True, max_length=50)
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
        features = generate_data(features)

        padded_features = np.zeros((6,self.max_len, 22)) # 6 469 22
        keypoints_mask = np.zeros((6,self.max_len))       # 469

        current_len = len(features[0])
        padded_features[:,:current_len, :] = features
        keypoints_mask[:,:current_len] = 1  

        sample = {
            "video_name": video_name,
            "keypoints": torch.FloatTensor(padded_features),
            "keypoints_mask": torch.FloatTensor(keypoints_mask),
        }
        return sample

class SimpleT5Model(nn.Module):

    def __init__(self, embed_size=384):
        super(SimpleT5Model, self).__init__()
        config = AutoConfig.from_pretrained('t5-base')
        self.t5 = T5ForConditionalGeneration.from_pretrained('t5-base', config=config)
        self.out_channel = CONFIG.OUT_CHANNEL 
        self.STAGCN  = st_agcn(num_class=1200, 
                                in_channels=6, 
                                residual=True, 
                                dropout=0.5, 
                                num_person=1, 
                                t_kernel_size=9,
                                layout='SMPL',
                                strategy='spatial',
                                hop_size=3,num_att_A=4 )
    
    def _get_encoder_feature(self, src):
        embedding, attention_node, attention_matrix  = self.STAGCN(src)
        return embedding, attention_node, attention_matrix    

    def forward(self, input_ids, attention_mask, decoder_input_ids=None, labels=None, use_embeds=True):
        if use_embeds:
            batch_size, channel,seq_length, feature_dim = input_ids.shape
            input_embeds, attention_node, attention_matrix  = self._get_encoder_feature(input_ids)
            new_attentention_mask = attention_mask[:,:,::4].clone()
            attention_mask = new_attentention_mask[:,0,:] # 8 6 118 --> 8 118
            output = self.t5(inputs_embeds=input_embeds, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids, labels=labels)
        else:
            output = self.t5(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids, labels=labels)
        
        return output

    def generate(self, input_ids, attention_mask, decoder_input_ids=None):
        batch_size, channel,seq_length, feature_dim = input_ids.shape
        input_embeds, attention_node, attention_matrix  = self._get_encoder_feature(input_ids)
        new_attentention_mask = attention_mask[:,:,::4].clone()
        attention_mask = new_attentention_mask[:,0,:]             # 8 6 118 --> 8 118
        beam_size = 5
        generated_ids = self.t5.generate( inputs_embeds=input_embeds, 
                                          attention_mask=attention_mask, 
                                          decoder_input_ids=decoder_input_ids, 
                                          max_length=50,
                                          num_beams=beam_size, 
                                          repetition_penalty=2.5,
                                          length_penalty=1.0,
                                          early_stopping=True)
                        
        return generated_ids , attention_node , attention_matrix

def train(train_dataset, model, tokenizer, args, eval_dataset=None, lr=1e-3, warmup_steps=5000):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = args.bs
    epochs = args.epochs

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
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
        progress = tqdm(total=len(train_dataloader), desc=args.prefix)
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
            torch.save(model.state_dict(), os.path.join(args.out_dir , f"{args.prefix}_epoch{epoch}.pt"))            
            model.eval()
            val_data_loader = DataLoader(eval_dataset, batch_size=2, shuffle=False)
            results = {}
            att_node_results = {}
            att_A_results = {}    
            with torch.no_grad():
                for batch in tqdm(val_data_loader):
                    video_names = batch['video_name']
                    src_batch = batch['keypoints'].to(device)
                    keypoints_mask_batch = batch['keypoints_mask'].to(device)
                    decoder_input_ids = tokenizer(["Description: "], # Instruction
                                                  return_tensors="pt", 
                                                  padding=True, 
                                                  truncation=True, 
                                                  add_special_tokens=False)['input_ids']
                    # decoder_input_ids = torch.tensor([[3]] * src_batch.shape[0]).to(device)
                    decoder_input_ids = decoder_input_ids.repeat(src_batch.shape[0], 1).to(device)
                    # decoder_input_ids = None
                    # print("decoder_input_ids",decoder_input_ids.shape)
                    generated_ids , att_node , att_A = model.generate(input_ids=src_batch.contiguous(), 
                                                                        attention_mask=keypoints_mask_batch.contiguous(),
                                                                        decoder_input_ids=decoder_input_ids)

                    for name, gen_id in zip(video_names, generated_ids):
                        decoded_text = tokenizer.decode(gen_id, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                        results[name] = decoded_text
                    for name, att_node in zip(video_names, att_node):
                        att_node_results[name] = att_node.cpu().numpy().tolist()
                    for name, att_A in zip(video_names, att_A):
                        att_A_results[name] = att_A.cpu().numpy().tolist()
                    
           
            with open(args.result_dir+'/results_epoch'+str(epoch)+'.json', 'w') as f:
                json.dump(results, f,indent = 1)
            with open(args.result_dir+'/att_node_results_epoch'+str(epoch)+'.json', 'w') as f:
                json.dump(att_node_results, f)
            with open(args.result_dir+'/att_A_results_epoch'+str(epoch)+'.json', 'w') as f:
                json.dump(att_A_results, f)

            predictions = readJSON(args.result_dir+'/results_epoch'+str(epoch)+'.json')
            annotations = readPickle(args.test_data)
           
            gts = getGTCaptions(annotations)
            # Check predictions content is correct
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
    parser.add_argument('--finetune', type=bool,default=False)
    parser.add_argument('--data', default='/home/weihsin/datasets/FigureSkate/HumanML3D_g/global_human_train.pkl')
    parser.add_argument('--out_dir', default='./models')
    parser.add_argument('--prefix', default='HumanML', help='prefix for saved filenames')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--save_every', type=int, default=1)
    parser.add_argument('--bs', type=int, default=8)
    parser.add_argument('--test_data', default='/home/weihsin/datasets/FigureSkate/HumanML3D_g/global_human_test.pkl')
    parser.add_argument('--result_dir', default = 'STAGCN_output')
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained('t5-base', use_fast=True)
    
    model = SimpleT5Model()
    if(args.finetune):
        # python train_t5_stagcn.py --finetune True > outputloss.txt  
        args.data    = '/home/weihsin/datasets/VQA/train_local.pkl'
        args.out_dir = './models_finetune'
        args.prefix  = 'Finetune'
        args.test_data  = '/home/weihsin/datasets/VQA/test_local.pkl'
        args.result_dir = 'STAGCN_output_finetune'
        weight           = './models/HumanML_epoch9.pt'
        model_state_dict = model.state_dict()
        state_dict = torch.load(weight)
        pretrained_dict_1 = {k: v for k, v in state_dict.items() if k in model_state_dict}
        model_state_dict.update(pretrained_dict_1)
        model.load_state_dict(model_state_dict)
    
    dataset = HumanMLDataset(args.data, tokenizer)
    eval_dataset = HumanMLDataset_val(args.test_data, tokenizer) 
    train(dataset, model, tokenizer, args,eval_dataset=eval_dataset)
    
if __name__ == '__main__':
    main()