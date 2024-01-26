import pickle
import numpy as np
import json
import os
from transformers import BertTokenizer
from transformers import AutoTokenizer
folder3Dskeleton = '/home/weihsin/datasets/Axel_clip_3Dskeleton' 
folderlabel = '/home/weihsin/datasets/Axel_clip_video'
# tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
output_final = []
token_type_ids = []
attention_mask = []
def label_fn_package(context_list) : 
    # output_final = tokenizer.encode_plus(context_list)
    encoding = tokenizer( context_list,padding=True)
    output_final = encoding.input_ids
    token_type_ids = encoding.token_type_ids 
    attention_mask = encoding.attention_mask
    return output_final,token_type_ids,attention_mask

def label_fn(label_content):
    output = tokenizer.encode(label_content)
    encoding = tokenizer( [label_content],padding=True)
    return output, encoding.token_type_ids ,encoding.attention_mask

def deal (video_path,video_name,root,label,output,token_type_ids,attention_mask):
    with open(video_path, "rb") as rd:
        data = pickle.load(rd)
    data = {key: data[key].tolist() if isinstance(data[key], np.ndarray) else data[key] for key in data}
    index = 0
    for frame in data['pred_xyz_24_struct'] :
 
        for i in range(0,len(frame)) :
            x = frame[i][0]
            y = frame[i][1]
            z = frame[i][2]

        data['pred_xyz_24_struct'][index] = [item for sublist in frame[0:22] for item in sublist]
        index+=1

    ##############################################################
    
    result = {
        "video_name": video_name,
        "features": data['pred_xyz_24_struct'],
        "label":label,
        "output":output,
        "token_type_ids":token_type_ids ,
        "attention_mask":attention_mask
    }
    video_name += ".json"
    video_path = os.path.join(root, video_name)
    print(video_path)
    with open(video_path, "w") as file:
        json.dump(result, file)

context_list = []
video_name_list = []
video2index_dict = dict()
index = 0
for root, dirs, files in os.walk(folder3Dskeleton):
    for file in files:
        if file.lower().endswith('.pk'):
            #os.path.splitext(file)[0]
            video_path = os.path.join(root, file)
            video_name = os.path.basename(os.path.dirname(video_path))
            OK =False
            for root_name, dirs_name, file_name in os.walk(folderlabel):
                for file in file_name:
                    if file == video_name + ".json":
                        file_path = os.path.join(folderlabel, file)
                        with open(file_path,"rb") as f:
                            labeldata = json.load(f)
                        OK = True
            if OK == True : 
                label_context = ""
                for item in labeldata:
                    label_context += item['context'] 
                video_name_list.append(video_name)
                context_list.append(label_context)
                video2index_dict[video_name] = index
                index += 1
output_final,token_type_ids,attention_mask = label_fn_package(context_list)

for root, dirs, files in os.walk(folder3Dskeleton):
    for file in files:
        if file.lower().endswith('.pk'):
            #os.path.splitext(file)[0]
            video_path = os.path.join(root, file)
            video_name = os.path.basename(os.path.dirname(video_path))           
            deal(video_path,
                video_name,
                root,
                context_list[video2index_dict[video_name]],
                output_final[video2index_dict[video_name]],
                token_type_ids[video2index_dict[video_name]],
                attention_mask[video2index_dict[video_name]]
                )
