"""
Intent: Format the pickle file as required in training and testing phases.
Author: Tom
Last update: 2023/09/26
"""
import pickle
import numpy as np
import json
import os
import cv2
from transformers import BertTokenizer, AutoTokenizer
from argparse import ArgumentParser, Namespace


tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--pickle_dir", type=str, default="/home/weihsin/datasets/FigureSkate/total", help="Path to the pickle folder"
    )
    parser.add_argument(
        "--input_file", type=str, default="train.pkl", help="Filename of the input pickle file"
    )
    parser.add_argument(
        "--output_file", type=str, default="train_out.pkl", help="Filename of the output pickle file"
    )
    args = parser.parse_args()
    return args


def label_fn_package(context_list): 
    encoding = tokenizer( context_list, padding=True)
    output_final = encoding.input_ids
    token_type_ids = encoding.token_type_ids 
    attention_mask = encoding.attention_mask
    return output_final, token_type_ids, attention_mask


def build_file2label(label_dir):
    """
    Input: a list of label filenames (.json)
    Output: a dictionary<filename, label>   # filename only contains the base filename without extension
    """
    # Fetch all the label files
    label_files = [f for f in os.listdir(label_dir) if f.endswith('.json')]
    file2label = dict()
    for file in label_files:
        filename, extension = os.path.splitext(file)
        with open(os.path.join(label_dir, file), 'r') as json_file:
            data = json.load(json_file)
            for datum in data:
                file2label[filename + "_" + str(datum['id'])] = datum['context']
    return file2label


def main(args):
    """
    Input format: pickle file ('video_name', 'features', 'labels')
    Output format: pickle file ('video_name', 'features', 'labels', 'output_sentence', 'output', 'token_type_ids', 'attention_mask')
    """

    # Load in directory paths
    pickle_dir = args.pickle_dir
    input_file = args.input_file
    output_file = args.output_file

    input_filepath = os.path.join(pickle_dir, input_file)
    with open(input_filepath, 'rb') as pk_file:
      output_data = []
      data = pickle.load(pk_file)
      for sample in data:
        for label in sample['labels']:
          output, token_type_ids, attention_mask = label_fn_package(label)
          output_data.append({
            'video_name': sample['video_name'],
            'features': sample['features'],
            'labels': sample['labels'],
            'output_sentence': label,
            'output': output,
            'token_type_ids': token_type_ids,
            'attention_mask': attention_mask
          })
      # Write out the file
      output_filepath = os.path.join(pickle_dir, output_file)
      with open(output_filepath, 'wb') as file:
          pickle.dump(output_data, file)

if __name__ == "__main__":
    args = parse_args()
    main(args)
    