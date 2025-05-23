import pickle, os, glob, json, re, torch
from BX_augmentation import BX_augmentation
from BX_translation import BX_translate_to_english
from clean_instruction import clean_instruction

def find_instruction(dataset_list, target_video_name):
    for video_name, comment in dataset_list.items() :
        if (target_video_name == video_name) :
            return comment

folders = {
    "Coach1" : "/home/weihsin/datasets/BoxingDatasetPkl/coach1.json",
    "Coach2" : "/home/weihsin/datasets/BoxingDatasetPkl/coach2.json",
    "Coach3" : "/home/weihsin/datasets/BoxingDatasetPkl/coach3.json",
    "Coach_rm" : "/home/weihsin/datasets/BoxingDatasetPkl/coach_rm.json"
}

train = True
if (train == True) :
    pkl = "/home/weihsin/datasets/BoxingDatasetPkl/boxing_train.pkl"
else :
    pkl = "/home/weihsin/datasets/BoxingDatasetPkl/boxing_test.pkl"

# open json
data_dict = []

with open(folders["Coach1"], "r") as f:
    dataset_Coach1 = json.load(f)

with open(folders["Coach2"], "r") as f:
    dataset_Coach2 = json.load(f)

with open(folders["Coach3"], "r") as f:
    dataset_Coach3 = json.load(f)

for video_name, comment in dataset_Coach1.items() :
    ori_video_name = video_name.split('.mp4')[0]

    coach1_instruction = comment
    coach2_instruction = find_instruction(dataset_Coach2, video_name)
    coach3_instruction = find_instruction(dataset_Coach3, video_name)

    coach1_eng_instruction = BX_translate_to_english(coach1_instruction)
    coach1_eng_instruction = clean_instruction(coach1_eng_instruction)
    print("translate eng: ",coach1_eng_instruction)
    coach2_eng_instruction = BX_translate_to_english(coach2_instruction)
    coach2_eng_instruction = clean_instruction(coach2_eng_instruction)
    print("translate eng: ",coach2_eng_instruction)
    coach3_eng_instruction = BX_translate_to_english(coach3_instruction)
    coach3_eng_instruction = clean_instruction(coach3_eng_instruction)
    print("translate eng: ",coach3_eng_instruction)
    labels = [coach1_eng_instruction,
              coach2_eng_instruction,
              coach3_eng_instruction]

    if "front" in ori_video_name:
        motion_type = "Jab"
    elif "back" in ori_video_name:
        motion_type = "Cross"

    aug_labels = []
    for idx in range(0,3) :
        aug_label = ""
        try_times = 0
        while aug_label.strip() == "" and try_times < 5:
            aug_label = BX_augmentation(labels[idx], motion_type, 0)
            aug_label = clean_instruction(aug_label)
            print("aug_label: ",aug_label)
            if aug_label.strip() == "" :
                print("Invalid rephrased instruction")
                try_times += 1
        aug_labels.append(aug_label)

    item = {
            "video_name": ori_video_name,
            "motion_type": motion_type,
            "coordinates": -1,
            "labels": labels,
            "augmented_labels": aug_labels,
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

######################################################################

new_boxing_path= "/home/weihsin/datasets/BoxingDatasetPkl/newBoxingDataset/text"
txt_filenames = [f for f in os.listdir(new_boxing_path) if f.endswith(".txt")]

# alignment model only learn jab & cross
motion_type_list = {
    "1" : "Jab",
    "2" : "Cross",
    "3" : "Lead Hook",
    "4" : "Rear Hook",
    "5" : "Lead Uppercut",
    "6" : "Rear Uppercut"
}

for name in txt_filenames:
    ori_video_name = name.split('.txt')[0]
    motion_type = ""
    if "1-" in ori_video_name:
        motion_type = motion_type_list["1"]
    elif "2-" in ori_video_name:
        motion_type = motion_type_list["2"]
    elif "3-" in ori_video_name:
        motion_type = motion_type_list["3"]
    elif "4-" in ori_video_name:
        motion_type = motion_type_list["4"]
    elif "5-" in ori_video_name:
        motion_type = motion_type_list["5"]
    elif "6-" in ori_video_name:
        motion_type = motion_type_list["6"]

    instruction_path = os.path.join(new_boxing_path,name)
    instruction= open(instruction_path).read()
    eng_instruction = BX_translate_to_english(instruction)
    print("translate eng: ",eng_instruction)
    label = []
    label.append(clean_instruction(eng_instruction))
    aug_labels = []

    for i in range(1, 6):
        aug_label = ""
        try_times = 0
        while aug_label.strip() == "" and try_times < 5:
            aug_label = BX_augmentation(eng_instruction, motion_type, i)
            print("aug_label: ",aug_label)
            aug_label = clean_instruction(aug_label)
            if aug_label.strip() == "" :
                print("Invalid rephrased instruction")
                try_times += 1
        aug_labels.append(clean_instruction(aug_label))

    item = {
            "video_name": ori_video_name,
            "motion_type": motion_type,
            "coordinates": -1,
            "labels": label,
            "augmented_labels": aug_labels,
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

new_boxing_dataset = ""
new_boxing_pkl = ""
if(train == True) :
    new_boxing_dataset = "../dataset/BX_train.json"
    new_boxing_pkl = "../dataset/BX_train.pkl"
else:
    new_boxing_dataset = "../dataset/BX_test.json"
    new_boxing_pkl = "../dataset/BX_test.pkl"

# save new_skating_dataset to json
with open(new_boxing_dataset, 'w', encoding='utf-8') as f:
    json.dump(data_dict, f, indent=4, ensure_ascii=False)

with open(new_boxing_pkl, 'wb') as f:
    pickle.dump(data_dict, f)

with open(new_boxing_dataset, "r") as file:
    boxing_dataset = json.load(file)
video_number = 0
new_dataset_number = 0
old_dataset_number = 0
error_number = 0
for data in boxing_dataset:
    if (len(data["augmented_labels"]) == 5 and len(data["labels"]) == 1):
        new_dataset_number += 1
    elif (len(data["augmented_labels"]) == 3 and len(data["labels"]) == 3):
        old_dataset_number += 1
    else :
        error_number+=1

print("error_number :",error_number)
print("new_dataset_number :", new_dataset_number)
print("old_dataset_number :",old_dataset_number)