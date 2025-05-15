import pickle
import json
import torch
import argparse
import os
import numpy as np


def convert_to_serializable(obj):
    if isinstance(obj, torch.Tensor):
        if obj.dtype in (torch.int32, torch.int64):
            return int(obj)
        return f"Tensor(shape={list(obj.shape)}, dtype={obj.dtype})"
    elif isinstance(obj, np.ndarray):
        return f"ndarray(shape={list(obj.shape)}, dtype={obj.dtype})"
    elif isinstance(obj, (np.integer, np.floating, )):
        return int(obj)
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    elif obj.dtype in (np.int32, np.int64, np.float32, np.float64,torch.int32, torch.int64, torch.float32, torch.float64):
        return int(obj)
    else:
        return str(obj)

def pickle_to_json(pickle_file, json_file):
    with open(pickle_file, 'rb') as f:
        data = pickle.load(f)
    
    # Sort the data by 'video_name' if it's a list of dictionaries
    if isinstance(data, list) and all(isinstance(item, dict) and 'video_name' in item for item in data):
        data.sort(key=lambda x: x['video_name'])
    
    converted_data = convert_to_serializable(data)
    
    with open(json_file, 'w') as f:
        json.dump(converted_data, f, indent=2)

def convert(pickle_path,json_path):

    if not os.path.exists(pickle_path):
        print(f"Error: Pickle file '{pickle_path}' does not exist.")
        return

    print(f"Converting '{pickle_path}' to '{json_path}'...")
    pickle_to_json(pickle_path, json_path)
    print("Conversion complete.")

def load_video_name(pickle_file) :
    with open(pickle_file, 'rb') as f :
        dataset = pickle.load(f)

    video_name_list = []
    for data in dataset :
        video_name_list.append(data['video_name'])
    
    return video_name_list