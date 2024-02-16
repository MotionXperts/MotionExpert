
import pickle
import numpy as np
import json
import os
import cv2
from transformers import BertTokenizer, AutoTokenizer
from argparse import ArgumentParser, Namespace



def main():
    pkl_file = "/home/weihsin/datasets/VQA/test_local.pkl"
    # read pkl file
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)
    results = {}
    for itme in data :
        
        results[itme['video_name']] = itme['label']

    json_file_path = '/home/weihsin/projects/MotionExpert/STAGCN_output_finetune/ground_truth.json'
    with open(json_file_path, 'w') as f:
        json.dump(results, f,indent = 1)
if __name__ == "__main__":
    main()
    