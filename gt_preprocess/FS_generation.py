import pickle, os, glob, json, re, torch
from FS_augmentation import FS_augmentation
from clean_instruction import clean_instruction

folders = {
    "Axel": "/home/weihsin/datasets/Axel",
    "Axel_com": "/home/weihsin/datasets/Axel_com",
    "Loop": "/home/weihsin/datasets/Loop",
    "Lutz": "/home/weihsin/datasets/Lutz"
}
train = False
if (train == True):
    pkl = "/home/weihsin/datasets/SkatingDatasetPkl/skating_gt_train.pkl"
else :
    pkl = "/home/weihsin/datasets/SkatingDatasetPkl/skating_gt_test.pkl"

with open(pkl, 'rb') as f :
    dataset = pickle.load(f)

data_dict = []
not_found = []

video_name_exist = []

for data in dataset :
    print("video_name: ",data["video_name"])

    video_name = data["video_name"]
    ori_video_name = video_name.split('_')[0]
    if ori_video_name in video_name_exist:
        continue
    video_name_exist.append(ori_video_name)

    motion_type = ""
    for name, path in folders.items():
        target_path = os.path.join(path, ori_video_name)
        if os.path.isdir(target_path):
            motion_type = name
            break

    print("target_path: ",target_path)
    if motion_type != "":
        json_files = glob.glob(os.path.join(target_path, "*.json"))
        with open(json_files[0], 'r') as f:
            clips = json.load(f)
        labels = []

        new_clips=[]
        time_clips = []
        for clip in clips:
            timestamp = clip["timestamp"]
            matches = re.findall(r"[\d.]+", timestamp)
            instruction = clean_instruction(clip["context"])
            if len(matches) == 2:
                start_time = float(matches[0])
                end_time = float(matches[1])

                find = False
                for idx, time_clip in enumerate(time_clips):
                    if(time_clip[0] == start_time and time_clip[1] == end_time):
                        new_clips[idx]["context"] = new_clips[idx]["context"] + " " +instruction 
                        print("new_concatenate :",new_clips[idx]["context"])
                        find = True
                if find == False:
                    time_clips.append([start_time,end_time])
                    item = {"id": len(new_clips),
                            "timestamp": clip["timestamp"],
                            "context": instruction
                    }
                    new_clips.append(item)

        for clip in new_clips:
            clip_id = clip["id"]
            timestamp = clip["timestamp"]
            instruction = clean_instruction(clip["context"])

            matches = re.findall(r"[\d.]+", timestamp)
            if len(matches) == 2:
                start_time = float(matches[0])
                end_time = float(matches[1])
                print("Start:", start_time, "End:", end_time)
            label = []
            label.append(instruction)
            labels.append(instruction)
            aug_labels = []

            for i in range(0, 5) :
                aug_label = ""
                try_times = 0
                while aug_label.strip() == "" and try_times < 5:
                    aug_label = FS_augmentation(instruction, motion_type, i)
                    aug_label = clean_instruction(aug_label)
                    if aug_label.strip() == "" :
                        print("Invalid rephrased instruction")
                        try_times += 1
                aug_labels.append(aug_label)
                
            item = {
                "video_name": ori_video_name + "_" + str(clip_id),
                "motion_type": motion_type,
                "coordinates": -1,
                "labels": label,
                "augmented_labels": aug_labels,
                "original_seq_len" : -1,
                "gt_start_frame" : round(start_time * 30),
                "gt_end_frame" : round(end_time * 30),
                "gt_std_start_frame" : -1,
                "gt_std_end_frame" : -1,
                "gt_seq_len" : -1
            }
            data_dict.append(item)

        item = {
            "video_name": ori_video_name,
            "motion_type": motion_type,
            "coordinates": -1,
            "labels": labels,
            "original_seq_len" : -1,
            "aligned_start_frame" : -1,
            "aligned_end_frame" : -1,
            "aligned_std_start_frame" : -1,
            "aligned_std_end_frame" : -1,
            "aligned_seq_len" : -1,
            "error_start_frame" : -1,
            "error_end_frame" : -1,
            "error_std_start_frame" : -1,
            "error_std_end_frame" : -1,
            "error_seq_len" : -1
        }
        data_dict.append(item)
    else:
        not_found.append(video_name)
        print("Not found:", video_name)

new_skating_dataset = ""
if (train == True):
    new_skating_dataset = "../dataset/FS_train.json"
else:
    new_skating_dataset = "../dataset/FS_test.json"
# save new_skating_dataset to json
with open(new_skating_dataset, 'w', encoding='utf-8') as f:
    json.dump(data_dict, f, indent=4, ensure_ascii=False)

if (train == True):
    skating_pkl = "../dataset/FS_train.pkl"
else:
    skating_pkl = "../dataset/FS_test.pkl"

with open(skating_pkl, 'wb') as f:
    pickle.dump(data_dict, f)

with open(new_skating_dataset, "r") as file:
    skating_dataset = json.load(file)
video_number = 0
number = 0
error_number = 0
for data in skating_dataset:
    if "_" in data["video_name"]:
        if len(data["augmented_labels"]) == 5:
            number += 1
        else :
            error_number += 1
    else:
        video_number += 1

print("video_number :",video_number)
print("error_number :", error_number)
print("number :",number)
