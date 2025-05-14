import pickle, os, json, torch, numpy as np
from torch.utils.data import Dataset

# The direction of the bone.
bonelink = [(0, 1), (0, 2), (0, 3), (1, 4), (2, 5), (3, 6), (4, 7), (5, 8), (6, 9), (7, 10), (8, 11),
            (9, 12), (9, 13), (9, 14), (12, 15), (13, 16), (14, 17), (16, 18), (17, 19), (18, 20), (19, 21)]

def get_coords(joint_coords) :
    # Initialize the coordinates (x, y, z) for 22 joints and 22 bones.
    joint = np.zeros((3, len(joint_coords), 22))
    bone = np.zeros((3, len(joint_coords), 22))
    for i in range(len(joint_coords)) :
        for j in range(0, len(joint_coords[i]), 3) :
            joint[:, i, j // 3] = joint_coords[i, j : j + 3]
    for v1, v2 in bonelink :
        bone[:, :, v2] = joint[:, :, v1] - joint[:, :, v2]
    skeleton_coords = np.concatenate((joint, bone), axis=0)
    return skeleton_coords

def get_std_coords(sport, motion_type, std_coords_list) :
    if sport == 'Skating' :
        if 'Axel' in motion_type :
            return std_coords_list[0]
        elif 'Axel_com' in motion_type :
            return std_coords_list[1]
        elif 'Loop' in motion_type :
            return std_coords_list[2]
        elif 'Lutz' in motion_type :
            return std_coords_list[3]
    elif sport == 'Boxing' :
        if 'back' in motion_type :
            return std_coords_list[0]
        elif 'front' in motion_type :
            return std_coords_list[1]

def get_label(pretrain, labels, aug_labels) :
    if not pretrain :
        labels = labels + aug_labels

    newlabels = ["Motion Description : " + label if pretrain else "Motion Instruction : " + label for label in labels]
    return newlabels

def get_segment(setting, item) :
    # Deal with GT segment selection.
    if setting == 'GT' :
        std_start = item["gt_std_start_frame"]
        usr_start = item["gt_start_frame"]
        length = item["gt_seq_len"]
    # Deal with error segment selection.
    elif setting == 'ERROR' :
        std_start = item["error_std_start_frame"]
        usr_start = item["error_start_frame"]
        length = item["error_seq_len"]
    # Deal with aligned segment selection.
    elif setting == 'ALIGNED' :
        std_start = item["std_start_frame"]
        usr_start = item["start_frame"]
        length = item["aligned_seq_len"]

    return std_start, usr_start, length

class DatasetLoader(Dataset) :
    def __init__(self, cfg, pretrain, pkl_file, train=True) :
        self.cfg = cfg

        with open(pkl_file, 'rb') as f :
            self.data_list = pickle.load(f)
            print("pkl_file", pkl_file)

        self.samples = []
        index_dict = {}

        if not pretrain :
            standard_path = cfg.STANDARD_PATH
            with open(standard_path, 'rb') as f :
                standard_features_file = pickle.load(f)
            std_coords_list = [get_coords(item['features']) for item in standard_features_file]

        for item in self.data_list :
            video_name = item['video_name']
            # Setting is 'ERROR' or 'ALIGNED' or 'NO_REF'
            if self.cfg.SETTING != 'GT' :
                if "_" in video_name :
                    continue
            else :
                if "_" not in video_name :
                    continue

            skeleton_coords = get_coords(item['features'])

            if pretrain == True or self.cfg.SETTING == "NO_SEGMENT":
                std_coords = skeleton_coords
                labels = get_label(pretrain, item['labels'], None)
                # Use the whole sequence without segmentation.
                usr_start, length = 0, skeleton_coords.shape[1]
                subtraction = torch.empty(0)
            else :
                motion_type = item['motion_type']
                std_coords = get_std_coords(cfg.TASK.SPORT, motion_type, std_coords_list)
                labels = get_label(pretrain, item['labels'], item['aug_labels'])

                # Specify the segment.
                start_frame, end_frame, length = int(item['start_frame']), int(item['end_frame']), int(item['end_frame']) - int(item['start_frame'])
                if self.cfg.TASK.DIFF_TYPE == 'RGB' :
                    subtraction = item['subtraction']
                    trimmed_start = item['trimmed_start']
                    if item['standard_longer'] :
                        std_start = start_frame
                        usr_start = trimmed_start
                    else :
                        usr_start = start_frame + trimmed_start
                        std_start = 0
                    # RGB setting will use subtraction instead of std_coords.
                    std_coords = torch.empty(0)
                    seq_len = skeleton_coords.shape[1]
                    subtraction = subtraction[: seq_len]

                elif self.cfg.TASK.DIFF_TYPE == 'Skeleton' :
                    std_start, usr_start, length = get_segment(self.cfg.SETTING, item)
                    # Skeleton setting will use std_coords instead of subtraction.
                    std_coords = std_coords[:, std_start : std_start + length]
                    subtraction = torch.empty(0)

            # Get the feature with the number of frames matching the number of frames in the standard feature.
            skeleton_coords = skeleton_coords[:, usr_start : usr_start + length]
            seq_len = skeleton_coords.shape[1]
            frame_mask = torch.ones(22)
            # Every ground truth should be used to learn.
            if train == True :
                for label in labels :
                    if seq_len == 0 or seq_len == 1 :
                        print(f"Skipping {video_name} as no frames found")
                        continue
                    self.samples.append((video_name,
                                        torch.FloatTensor(skeleton_coords),
                                        seq_len,
                                        torch.FloatTensor(frame_mask),
                                        label,
                                        labels,
                                        torch.FloatTensor(std_coords),
                                        subtraction))
            # Every sample only need to feed into the model once when testing.
            else :
                self.samples.append((video_name,
                                    torch.FloatTensor(skeleton_coords),
                                    seq_len,
                                    torch.FloatTensor(frame_mask),
                                    labels[0],
                                    labels,
                                    torch.FloatTensor(std_coords),
                                    subtraction))
            if pretrain == False :
                # Save data for visulization attention graph on 2D skeleton.
                index_dict[video_name] = {"seq_len" : seq_len,
                                          "feature_start_frame" : usr_start,
                                          "std_start_frame" : std_start}

        print('Number of sample : ', len(self.samples))

        if (pretrain == False) :
            with open(cfg.LOGDIR + '/index_dict_results.json', 'w') as f :
                json.dump(index_dict, f, indent = 4)

    def __len__(self) :
        return len(self.samples)

    def __getitem__(self, idx) :
        if torch.is_tensor(idx) :
            idx = idx.tolist()
        return self.samples[idx]
