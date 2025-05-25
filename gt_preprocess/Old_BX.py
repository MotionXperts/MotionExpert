import pickle, os, random, json, numpy as np, torch, pickle

def readpkl(path) :
    with open(path, "rb") as rd:
        dataset = pickle.load(rd)
    return dataset

aggregate_BX_train_path = "../dataset/boxing_aggregate_train.pkl"
aggregate_BX_test_path = "../dataset/boxing_aggregate_test.pkl"

BX_front_path = ""
BX_back_path = ""
aggregate_BX_train = readpkl(aggregate_BX_train_path)
aggregate_BX_test = readpkl(aggregate_BX_test_path)

agg_BX_train = []
agg_BX_test = []

standard_path = "/home/andrewchen/Error_Localize/standard_features_boxing.pkl"
standard = readpkl(standard_path)
front_feature = standard[1]["features"]
back_feature = standard[0]["features"]

standard_pkl = []
Jab_item = {
    "video_name": "Jab",
    "coordinates": torch.from_numpy(front_feature).float()
}
Cross_item = {
    "video_name": "Cross",
    "coordinates": torch.from_numpy(back_feature).float()
}
standard_pkl.append(Jab_item)
standard_pkl.append(Cross_item)

print("Jab : 109", len(Jab_item["coordinates"]))
print("Cross : 76 ", len(Cross_item["coordinates"]))
jab_feature = Jab_item["coordinates"]
cross_feature = Cross_item["coordinates"]
output_path = "../dataset/BX_standard.pkl"
with open(output_path, 'wb') as f:
    pickle.dump(standard_pkl, f)


datasets = [aggregate_BX_train, aggregate_BX_test]
for idx, dataset in enumerate(datasets) :
    for data in dataset :
        item = {}
        for key, value in data.items(): 
            if key == "features" or key == "frame_label" or key == "subtraction":
                continue
            
            if("front" in data["video_name"]) :
                std_feature = jab_feature
            else :
                std_feature = cross_feature

            item["video_name"] = data["video_name"]
            item["coordinates"] = torch.from_numpy(data["features"]).float()
            # alignmend
            start_frame, end_frame = int(data['start_frame']), int(data['end_frame'])
            trimmed_start = int(data['trimmed_start'])
            length = int(data['end_frame']) - int(data['start_frame'])
            if data['standard_longer'] :
                std_start = start_frame
                usr_start = trimmed_start
            else :
                usr_start = start_frame + trimmed_start
                std_start = 0
            item["original_seq_len"] = len(data['features'])
            item["aligned_start_frame"] = usr_start
            item["aligned_end_frame"] = usr_start + length
            if (item["aligned_end_frame"] > len(data['features'])) :
                video_name = data["video_name"]
                usr_end = usr_start + length
                len_of_feature = len(data['features'])
                print(f"Aligned : {video_name} invalid usr, start frame : {usr_start}, end frame : {usr_end}, usr featrue len : {len_of_feature}")
            item["aligned_std_start_frame"] = std_start
            item["aligned_std_end_frame"] = std_start + length
            if (item["aligned_std_end_frame"] > len(std_feature)) :
                video_name = data["video_name"]
                std_end = std_start + length
                len_of_feature = len(std_feature)
                print(f"Aligned : {video_name} invalid usr, start frame : {std_start}, end frame : {std_end}, std featrue len : {len_of_feature}")
            item["aligned_seq_len"] = length
            '''
            # Error
            error_start, error_end = int(data['error_start_frame']), int(data['error_end_frame'])
            length = error_end - error_start
            if data['standard_longer'] :
                std_start = int(data['start_frame']) + error_start
                usr_start = trimmed_start + error_start
            else :
                std_start = error_start
                usr_start = int(data['start_frame']) + error_start
            item["error_start_frame"] = usr_start
            item["error_end_frame"] = usr_start + length
            if (item["error_end_frame"] > len(data['features'])) :
                video_name = data["video_name"]
                usr_end = usr_start + length
                len_of_feature = len(data['features'])
                print(f"Error : {video_name} invalid usr, start frame : {usr_start}, end frame : {usr_end}, usr featrue len : {len_of_feature}")
            item["error_std_start_frame"] = std_start
            item["error_std_end_frame"] = std_start + length
            if (item["error_std_end_frame"] > len(std_feature)) :
                video_name = data["video_name"]
                std_end = std_start + length
                len_of_feature = len(std_feature)
                print(f"Error : {video_name} invalid usr, start frame : {std_start}, end frame : {std_end}, std featrue len : {len_of_feature}")
            item["error_seq_len"] = length
            '''
            # print(f"{key} : ", type(value))

        if (idx == 0) :
            agg_BX_train.append(item)
        else :
            agg_BX_test.append(item)
        
'''
with open('../dataset/BX_aligned_train.json', 'w', encoding='utf-8') as f:
    json.dump(agg_BX_train, f, ensure_ascii=False, indent=4)

with open('../dataset/BX_aligned_test.json', 'w', encoding='utf-8') as f:
    json.dump(agg_BX_test, f, ensure_ascii=False, indent=4)

'''
with open('../dataset/BX_aligned_train.pkl', 'wb') as f:
    pickle.dump(agg_BX_train, f)

with open('../dataset/BX_aligned_test.pkl', 'wb') as f:
    pickle.dump(agg_BX_test, f)


'''
video_file :  <class 'str'>
frame_label :  <class 'torch.Tensor'>
seq_len :  <class 'int'>
name :  <class 'str'>
id :  <class 'int'>
labels :  <class 'list'>
start_frame :  <class 'torch.Tensor'>
end_frame :  <class 'torch.Tensor'>
standard_longer :  <class 'bool'>
subtraction :  <class 'torch.Tensor'>
video_name :  <class 'str'>
trimmed_start :  <class 'int'>
trimmed_end :  <class 'int'>
'''


