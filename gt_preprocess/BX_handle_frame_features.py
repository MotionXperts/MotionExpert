import pickle, os, random, json, numpy as np, torch

train = True
alignment = True

if(train == True) :
    new_boxing_dataset = "../dataset/BX_train_clean.json"

with open(new_boxing_dataset, "r") as f:
    new_boxing_dataset = json.load(f)

pkldata = []

def readpkl(path) :
    with open(path, "rb") as rd:
        dataset = pickle.load(rd)
    return dataset

# new_bx_path = "/home/weihsin/datasets/BoxingDatasetPkl/new_boxing_dataset_20250318_norm.pkl"
Old_BX_train_path = "../dataset/BX_aligned_train.pkl"
Old_BX_test_path = "../dataset/BX_aligned_test.pkl"

Old_BX_train_path = "/home/weihsin/datasets/BoxingDatasetPkl/boxing_train.pkl"
Old_BX_test_path = "/home/weihsin/datasets/BoxingDatasetPkl/boxing_test.pkl"

# new_bx_feature = readpkl(new_bx_path)
Old_BX_train = readpkl(Old_BX_train_path)
Old_BX_test = readpkl(Old_BX_test_path)
Old_BX = Old_BX_train + Old_BX_test

def deal(video_path) :
    data = readpkl(video_path)

    processed = []
    for frame in data['pred_xyz_24_struct'] :
        # 22 joints × 3 coords = 66
        flattened = [coord for joint in frame[0:22] for coord in joint]
        processed.append(flattened)

    coordinates = torch.tensor(processed, dtype=torch.float32)
    return coordinates
'''
# New BX dataset
def find_coordinates(video_name) :
    for data in new_bx_feature :
        if data["video_name"] == video_name :
            return torch.tensor(data["features"], dtype=torch.float32)
'''
def find_frame(video_name) :
    for idx, data in enumerate(Old_BX) :
        if data["video_name"] == video_name :
            return data

new_dataset = []
if alignment == True: 
    # Handle Features
    for data in new_boxing_dataset :
        if '-' not in data["video_name"] :
            new_data= {}
            # do not consider camera 4:
            camera_view = str(random.choice([1, 2, 3]))
            path = "/home/weihsin/projects/HybrIK/Boxing/cam" + camera_view + "/" + data["video_name"] + "/res.pk"
            new_data["video_name"] = data["video_name"]
            if ("front" in data["video_name"]):
                new_data["motion_type"] = "Jab"
            if ("back" in data["video_name"]):
                new_data["motion_type"] =  "Cross"
            new_data["coordinates"] = deal(path)
            new_data["camera_view"] = camera_view
            new_data["labels"] = data["labels"]
            new_data["augmented_labels"] = data["augmented_labels"]
            # Handle Frames
            oldbx_data = find_frame(data["video_name"]+"_cam"+camera_view)
            print(oldbx_data["original_seq_len"])
            if len(new_data["coordinates"]) != oldbx_data["original_seq_len"]:
                print("Original length not the same :",data["video_name"])

            new_data["original_seq_len"] = oldbx_data["original_seq_len"]
            new_data["aligned_start_frame"] = oldbx_data["start_frame"]
            new_data["aligned_end_frame"] = oldbx_data["end_frame"]
            new_data["aligned_std_start_frame"] = oldbx_data["std_start_frame"]
            new_data["aligned_std_end_frame"] = oldbx_data["std_end_frame"]
            new_data["aligned_seq_len"] = oldbx_data["aligned_seq_len"]
            new_dataset.append(new_data)

# split list new_dataset to 4 : 1
random.seed(42)
random.shuffle(new_dataset)
split_index = int(len(new_dataset) * 0.8)
train_data = new_dataset[:split_index]
test_data = new_dataset[split_index:]

with open('../dataset/BX_train.pkl', 'wb') as f:
    pickle.dump(train_data, f)
with open('../dataset/BX_test.pkl', 'wb') as f:
    pickle.dump(test_data, f)

json_train = []
for idx, data in enumerate(train_data) :
    json_data = {}
    json_data["video_name"] = data["video_name"]
    json_data["motion_type"] = data["motion_type"]
    json_data["camera_view"] = data["camera_view"]
    json_data["labels"] = data["labels"]
    json_data["augmented_labels"] = data["augmented_labels"]
    json_data["original_seq_len"] = data["original_seq_len"]
    json_data["aligned_start_frame"] = data["aligned_start_frame"]
    json_data["aligned_end_frame"] = data["aligned_end_frame"]
    json_data["aligned_std_start_frame"] = data["aligned_std_start_frame"]
    json_data["aligned_std_end_frame"] = data["aligned_std_end_frame"]
    json_data["aligned_seq_len"] = data["aligned_seq_len"]
    if (idx == 0):
        print("video_name : ", type(data["video_name"]))
        print("motion type : ",type(data["motion_type"]))
        print("coordinates : ",type(data["coordinates"]))
        print("camera_view : ",type(data["camera_view"]))
        print("labels : ", type(data["labels"]))
        print("augmented_labels : ", type(data["augmented_labels"]))
        print("original_seq_len : ", type(data["original_seq_len"]))
        print("aligned_start_frame : ", type(data["aligned_start_frame"]))
        print("aligned_end_frame : ", type(data["aligned_end_frame"]))
        print("aligned_std_start_frame : ", type(data["aligned_std_start_frame"]))
        print("aligned_std_end_frame : ", type(data["aligned_std_end_frame"]))
        print("aligned_seq_len : ", type(data["aligned_seq_len"]))
    json_train.append(json_data)

json_test = []
for idx, data in enumerate(test_data) :
    json_data = {}
    json_data["video_name"] = data["video_name"]
    json_data["motion_type"] = data["motion_type"]
    json_data["camera_view"] = data["camera_view"]
    json_data["labels"] = data["labels"]
    json_data["augmented_labels"] = data["augmented_labels"]
    json_data["original_seq_len"] = data["original_seq_len"]
    json_data["aligned_start_frame"] = data["aligned_start_frame"]
    json_data["aligned_end_frame"] = data["aligned_end_frame"]
    json_data["aligned_std_start_frame"] = data["aligned_std_start_frame"]
    json_data["aligned_std_end_frame"] = data["aligned_std_end_frame"]
    json_data["aligned_seq_len"] = data["aligned_seq_len"]
    if (idx == 0):
        print("video_name : ", type(data["video_name"]))
        print("motion type : ",type(data["motion_type"]))
        print("coordinates : ",type(data["coordinates"]))
        print("camera_view : ",type(data["camera_view"]))
        print("labels : ", type(data["labels"]))
        print("augmented_labels : ", type(data["augmented_labels"]))
        print("original_seq_len : ", type(data["original_seq_len"]))
        print("aligned_start_frame : ", type(data["aligned_start_frame"]))
        print("aligned_end_frame : ", type(data["aligned_end_frame"]))
        print("aligned_std_start_frame : ", type(data["aligned_std_start_frame"]))
        print("aligned_std_end_frame : ", type(data["aligned_std_end_frame"]))
        print("aligned_seq_len : ", type(data["aligned_seq_len"]))
    json_test.append(json_data)

with open('../dataset/BX_train.json', 'w', encoding='utf-8') as f:
    json.dump(json_train, f, ensure_ascii=False, indent=4)

with open('../dataset/BX_test.json', 'w', encoding='utf-8') as f:
    json.dump(json_test, f, ensure_ascii=False, indent=4)
'''
# New BX dataset
new_boxing_alignment = []
# Handle Features
idx = 0
for data in new_boxing_dataset:
    if '-' not in data["video_name"]:
        # do not consider camera 4:
        view = str(random.choice([1, 2, 3]))
        path = "/home/weihsin/projects/HybrIK/Boxing/cam" + view + "/" + data["video_name"] + "/res.pk"
        data["coordinates"] = deal(path)
        data["view"] = view
    else:
        video_name = data["video_name"]
        features = find_coordinates(data["video_name"])
            
        for i in range(1,4):
            new_video_name = video_name + '_cam'+str(i)
            new_video_nameMP4 = new_video_name + ".mp4"
            path_video = os.path.join("/home/weihsin/datasets/newBX/processed_videos", new_video_nameMP4)
            item = {"video_file" : path_video,
                    "frame_label" : torch.zeros(len(features)),
                    "seq_len" : len(features),
                    "name": new_video_name,
                    "key" : idx
                    "labels" : data["labels"]
                    "features" : data["coordinates"]}
            idx = idx + 1
            new_boxing_alignment.append(item)
'''