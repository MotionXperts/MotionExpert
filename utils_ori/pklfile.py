import os
import json
import pickle
import numpy as np
import random
import random
import shutil

def find_every_action(folder_path):
    data = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith('.json'):

                video_path = os.path.join(root, file)
                video_name = os.path.basename(os.path.dirname(video_path))
                file_path = os.path.join(folder_path, video_name,file)
                with open(file_path, "r") as json_file:
                    json_data = json.load(json_file)
                    print(file_path)
                    print(len(json_data)) 
                    if(len(json_data) != 6) : continue
                    features_item = np.array(json_data["features"])
                    json_data["features"] = features_item
                    output_item = np.array(json_data["output"])
                    json_data["output"] = output_item
                    attention_mask_item = np.array(json_data["attention_mask"])
                    json_data["attention_mask"] = attention_mask_item
                    token_type_ids_item = np.array(json_data["token_type_ids"])
                    json_data["token_type_ids"] = token_type_ids_item
                    data.append(json_data)
    return data
                
def pack_json_files_to_pkl(folder_path,name,data):
    test_pkl = os.path.join(folder_path,name)

    with open(test_pkl, "wb") as pkl_file:
        pickle.dump(data, pkl_file)

def copy_files(folders, source_folder, destination_folder):
    for folder in folders:
        source_path = os.path.join(source_folder, folder)
        destination_path = os.path.join(destination_folder, folder)
        shutil.copytree(source_path, destination_path)

def dataset_split(data_folder,train_folder,test_folder,validation_folder):

    random.seed(42)

    # 获取文件夹中的所有文件列表
    file_list = os.listdir(data_folder)

    # 随机打乱文件列表
    random.shuffle(file_list)

    # 计算划分的索引
    total_files = len(file_list)
    train_split = int(0.7 * total_files)
    test_split = int(0.2 * total_files)

    # 将文件分配到不同的集合中
    train_files = file_list[:train_split]
    test_files = file_list[train_split:train_split + test_split]
    validation_files = file_list[train_split + test_split:]
    print(train_files)
    copy_files(train_files, data_folder, train_folder)
    copy_files(test_files, data_folder, test_folder)
    copy_files(validation_files, data_folder, validation_folder)

def merge_pkl_files(file1_path, file2_path, merged_file_path):
    # Read the first .pkl file
    with open(file1_path, 'rb') as file1:
        data1 = pickle.load(file1)

    # Read the second .pkl file
    with open(file2_path, 'rb') as file2:
        data2 = pickle.load(file2)

    # Merge the data
    if isinstance(data1, list) and isinstance(data2, list):
        merged_data = data1 + data2
    elif isinstance(data1, dict) and isinstance(data2, dict):
        merged_data = {**data1, **data2}
    else:
        raise ValueError("Unsupported data types: data1={}, data2={}".format(type(data1), type(data2)))

    # Save the merged data to a new .pkl file
    with open(merged_file_path, 'wb') as merged_file:
        pickle.dump(merged_data, merged_file)

    print("Merge completed")


if __name__ == "__main__":
    #data_folder = '/home/weihsin/datasets/Axel_clip_3Dskeleton/'
    #train_folder = '/home/weihsin/datasets/Axel_clip_3Dskeleton/train_folder'
    #test_folder = '/home/weihsin/datasets/Axel_clip_3Dskeleton/test_folder'
    #validation_folder = '/home/weihsin/datasets/Axel_clip_3Dskeleton/validation_folder'

    #train_name = "train.pkl"
    ##train_data = find_every_action(train_folder)
    #pack_json_files_to_pkl(train_folder,train_name,train_data)
    #print("train.pkl finish")

    #validation_name = "validation.pkl"
    #validation_data = find_every_action(validation_folder)
    #pack_json_files_to_pkl(validation_folder,validation_name,validation_data)
    #print("validation.pkl finish")

    #all_name = "all.pkl"
    #all_data = find_every_action(data_folder)
    #pack_json_files_to_pkl(data_folder,all_name,all_data)
    #print("all.pkl finish")
    file1_path = '/home/weihsin/datasets/FigureSkate/Axel_clip_3Dskeleton/train_folder/train.pkl'
    file2_path = '/home/weihsin/datasets/FigureSkate/Axel_clip_3Dskeleton/validation_folder/validation.pkl'
    merged_file_path = 'merged_file.pkl'

    merge_pkl_files(file1_path, file2_path, merged_file_path)