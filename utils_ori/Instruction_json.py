import pickle
import json

def load_pklle(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)   

    data_write = {}
    for item in data:
        data_write[item['video_name']] = item['labels']
    
    with open('instruction.json', 'w') as f:
        json.dump(data_write, f, indent=4)

if __name__ == '__main__':
    file_path = '/home/weihsin/datasets/VQA/train_local.pkl'
    load_pklle(file_path)