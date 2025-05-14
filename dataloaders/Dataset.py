import pickle, os, json, torch, numpy as np
from torch.utils.data import Dataset

# The direction of the bone.
bonelink = [(0, 1), (0, 2), (0, 3), (1, 4), (2, 5), (3, 6), (4, 7), (5, 8), (6, 9), (7, 10), (8, 11),
            (9, 12), (9, 13), (9, 14), (12, 15), (13, 16), (14, 17), (16, 18), (17, 19), (18, 20), (19, 21)]

def generate_data(current_keypoints) :
    # Initialize the coordinates (x, y, z) for 22 joints and 22 bones.
    joint = np.zeros((3, len(current_keypoints), 22))
    bone = np.zeros((3, len(current_keypoints), 22))
    for i in range(len(current_keypoints)) :
        for j in range(0, len(current_keypoints[i]), 3) :
            joint[:, i, j // 3] = current_keypoints[i, j : j + 3]
    for v1, v2 in bonelink :
        bone[:, :, v2] = joint[:, :, v1] - joint[:, :, v2]
    coordinates = np.concatenate((joint, bone), axis=0)
    return coordinates

class DatasetLoader(Dataset) :
    def __init__(self, cfg, pretrain, pkl_file) :
        self.cfg = cfg

        with open(pkl_file, 'rb') as f :
            self.data_list = pickle.load(f)
            print("pkl_file", pkl_file)

        if pretrain :
            self.standard_features_list = [generate_data(self.data_list[0]['features'])]
        else :
            standard_path = cfg.STANDARD_PATH
            with open(standard_path, 'rb') as f :
                standard_features_file = pickle.load(f)
                self.standard_features_list = [generate_data(item['features']) for item in standard_features_file]

        self.samples = []
        max_len = 0  
        index_dict = {}
        print('Number of data in dataset : ', len(self.data_list))

        for item in self.data_list :
            video_name = item['video_name']
            features = generate_data(item['features'])

            # Specify the corresponding standard features for the video.
            # Motion Description.
            if len(self.standard_features_list) == 1 :
                std_features = self.standard_features_list[0]
            # Motion Instruction.
            elif cfg.TASK.SPORT == 'Skating' :
                if 'Axel' in item['original_video_file'] :
                    std_features = self.standard_features_list[0]
                elif 'Axel_com' in item['original_video_file'] :
                    std_features = self.standard_features_list[1]
                elif 'Loop' in item['original_video_file'] :
                    std_features = self.standard_features_list[2]
                else :
                    std_features = self.standard_features_list[3]
            elif cfg.TASK.SPORT == 'Boxing' :
                if 'back' in item['video_name'] :
                    std_features = self.standard_features_list[0]
                elif 'front' in item['video_name'] :
                    std_features = self.standard_features_list[1]

            # Specify the label.
            if cfg.TASK.SPORT == 'Skating' :
                if 'train' in pkl_file :
                    labels = item['augmented_labels']
                    labels.append(item['revised_label'])
                elif cfg.args.eval_name == 'segment' or cfg.args.eval_name == 'untrimmed' :
                    labels = ['']
                else:
                    labels = [item['revised_label']]
            elif cfg.TASK.SPORT == 'Boxing' or cfg.TASK.SPORT == 'Manipulate' :
                if cfg.args.eval_name == 'segment' or cfg.args.eval_name == 'untrimmed' :
                    labels = ['']
                else :
                    labels = item['labels']

            # Specify the task of training.
            newlabels = ["Motion Description : " + label if pretrain else "Motion Instruction : " + label for label in labels]

            # Specify the trimmed_start.
            if 'trimmed_start' in item :
                trimmed_start = item['trimmed_start']
            else :
                trimmed_start = 0

            # Specify the number of frame.
            start_frame, end_frame = int(item['start_frame']), int(item['end_frame'])
            length = end_frame - start_frame - 1

            if self.cfg.TASK.DIFFERENCE_TYPE == 'RGB' :
                subtraction = item['subtraction']
                if item['standard_longer'] :
                    std_start = start_frame
                    usr_start = trimmed_start
                else :
                    usr_start = start_frame + trimmed_start
                    std_start = 0
                # RGB setting will use subtraction instead of std_features.
                std_features = torch.empty(0)
                subtraction = subtraction[: features.shape[1]]

            elif self.cfg.TASK.DIFFERENCE_TYPE == 'Skeleton' :
                # Deal with error segment selection.
                if self.cfg.args.eval_name == 'segment' :
                    error_start, error_end = int(item['error_start_frame']), int(item['error_end_frame']) - 1
                    length = error_end - error_start
                    if item['error_end_frame'] == 0 :
                        usr_start, std_start, length = 0, 0, 1
                    else :
                        if item['standard_longer'] :
                            std_start = start_frame + error_start
                            usr_start = trimmed_start + error_start
                        else :
                            std_start = error_start
                            usr_start = start_frame + error_start
                else :
                    std_start = item["std_start_frame"]
                    usr_start = item["start_frame"]
                    length = item["aligned_seq_len"]

                # Skeleton setting will use std_features instead of subtraction.
                std_features = std_features[:, std_start : std_start + length]
                subtraction = torch.empty(0)
            # Get the feature with the number of frames matching the number of frames in the standard feature.
            features = features[:, usr_start : usr_start + length]

            # Every ground truth should be used to learn.
            for label in newlabels :
                if features.shape[1] == 0 or features.shape[1] == 1 :
                    print(f"Skipping {video_name} as no frames found")
                    continue
                self.samples.append((features, label, video_name, subtraction, std_features, newlabels))

            max_len = max(max_len, len(features[0]))

            # Save data for visulization attention graph on 2D skeleton.
            index_dict[video_name] = {"seq_len" : length,
                                      "feature_start_frame" : usr_start,
                                      "std_start_frame" : std_start}

        self.max_len = max_len
        print('Number of sample : ', len(self.samples))

        if (cfg.args.eval_name != 'untrimmed' and cfg.TASK.SPORT != 'Boxing' and cfg.TASK.DIFFERENCE_SETTING != 'No') :
            with open(cfg.LOGDIR + '/index_dict_results.json', 'w') as f :
                json.dump(index_dict, f, indent = 4)

    def __len__(self) :
        return len(self.samples)

    def __getitem__(self, idx) :
        if torch.is_tensor(idx) :
            idx = idx.tolist()

        # Features : 6 x frame x 22.
        features, label, video_name, subtraction, std_features, labels = self.samples[idx]
        keypoints_mask = torch.ones(22)
        current_len = torch.tensor(len(features[0]))

        # Change self.standard to std_features.
        return  video_name, torch.FloatTensor(features), torch.FloatTensor(keypoints_mask), torch.FloatTensor(std_features), current_len, label, subtraction, labels
