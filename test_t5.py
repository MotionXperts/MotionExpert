import argparse
from tqdm import tqdm
import os
import sys
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import pickle
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, AutoConfig, T5ForConditionalGeneration
import torch
import json
from tqdm import tqdm

class HumanMLDataset(Dataset):
    def __init__(self, pkl_file, tokenizer, transform = None):
        with open(pkl_file, 'rb') as f:
            self.video_data_list = pickle.load(f)
        self.tokenizer = tokenizer
        self.transform = transform

    def __len__(self):
        return len(self.video_data_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        video_name = self.video_data_list[idx]['video_name']
        sentence = self.video_data_list[idx]['output_sentence'] if 'output_sentence' in self.video_data_list[idx] else self.video_data_list[idx]['label']
        tokenized_output = self.tokenizer(sentence, return_tensors="pt", padding="max_length", truncation=True, max_length=50)
        mask_output = tokenized_output['attention_mask'].squeeze(0)
        features_list = self.video_data_list[idx]['features']
        labels = self.video_data_list[idx]['labels'] if 'labels' in self.video_data_list[idx] else [self.video_data_list[idx]['label']]

        sample = {
          "keypoints": torch.FloatTensor(features_list), 
          "video_name": video_name, 
          "labels": labels,
          "output": tokenized_output['input_ids'].squeeze(0),
          "output_mask": mask_output
        }
        return sample
    
    def collate_fn(self, samples):
        PAD_IDX = 0
        video_names, labels_list, src_batch, tgt_batch = [], [], [], []
        samples.sort(key = lambda x: len(x["keypoints"]), reverse = True)
        to_len = len(samples[0]["keypoints"])
        mask = np.zeros((len(samples), to_len))

        for idx, sample in enumerate(samples):
            video_names.append(sample["video_name"])
            labels_list.append(sample["labels"])
            padded_tensor = torch.zeros((to_len, 66))
            padded_tensor[:len(sample["keypoints"]), :] = sample["keypoints"]
            src_batch.append(padded_tensor)

            tgt_batch.append(sample["output"])
            output_len = len(sample["keypoints"])
            mask[idx][:output_len] = np.ones(output_len)

        src_batch = torch.stack(src_batch)
        tgt_batch = torch.stack(tgt_batch)
        mask = torch.tensor(mask).bool()
        return video_names, labels_list, src_batch, tgt_batch, mask

class SimpleT5Model(nn.Module):

    def __init__(self, embed_size=256):
        super(SimpleT5Model, self).__init__()
        config = AutoConfig.from_pretrained('t5-base')
        self.t5 = T5ForConditionalGeneration.from_pretrained('t5-base', config=config)

        self.embedding = nn.Sequential(nn.Linear(66,embed_size), nn.ReLU(),nn.Linear(embed_size,768))
        
    def forward(self, input_ids, attention_mask, decoder_input_ids=None, labels=None, use_embeds=True):
        if use_embeds:
            batch_size, seq_length, feature_dim = input_ids.shape
            input_embeds = self.embedding(input_ids.view(-1,feature_dim)).view(batch_size, seq_length, -1)
            output = self.t5(inputs_embeds=input_embeds, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids, labels=labels)
        else:
            output = self.t5(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids, labels=labels)

        return output


def evaluate(dataset, model, device, beam_size=2):
    model.eval()
    data_loader = DataLoader(dataset, batch_size=8, collate_fn=dataset.collate_fn, shuffle=False)
    results = {}

    tokenizer = AutoTokenizer.from_pretrained('t5-base', use_fast=True)
    start_token_id = 3

    with torch.no_grad():
        for video_names, _, src_batch,_,_ in tqdm(data_loader): 
            src_batch = src_batch.to(device)

            input_embeds = model.embedding(src_batch)

            decoder_input_ids = torch.tensor([[start_token_id]] * src_batch.shape[0]).to(device)
            
            generated_ids = model.t5.generate(
                inputs_embeds=input_embeds,
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

    with open('./generated_labels.json', 'w') as f:
        json.dump(results, f, indent=4)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    parser = argparse.ArgumentParser()
    tokenizer = AutoTokenizer.from_pretrained('t5-base', use_fast=True)
    model = SimpleT5Model().to(DEVICE)
    # Load the trained model
    model_path = '/home/peihsin/projects/MotionExpert/models/axel_com_10.pt'
    model.load_state_dict(torch.load(model_path))
    parser.add_argument('--test_data', default='/home/peihsin/projects/humanML/dataset/test.pkl') 
    args = parser.parse_args()

    # Test the model
    device = torch.device('cuda')
    test_dataset = HumanMLDataset(args.test_data, tokenizer)    
    evaluate(test_dataset, model, device, beam_size=2)
