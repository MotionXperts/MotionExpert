import pickle,os,sys, json
import torch
import numpy as np
from torch.utils.data import Dataset
# from VideoAlignment.dataset.data_augment import create_data_augment
USER = os.environ['USER']

bonelink = [(0, 1), (0, 2), (0, 3), (1, 4), (2,5), (3,6), (4, 7), (5, 8), (6, 9), (7, 10), 
            (8, 11), (9, 12), (9, 13), (9, 14), (12, 15), (13, 16), (14, 17), (16, 18), (17, 19), (18,  20), (19, 21)]

def generate_data(current_keypoints ):
    joint_coordinate    = np.zeros((3,len(current_keypoints),22))
    bone_coordinate     = np.zeros((3,len(current_keypoints),22))
    for i in range(len(current_keypoints)):
        for j in range(0,len(current_keypoints[i]),3):
            joint_coordinate[:, i, j//3] = current_keypoints[i, j: j+3]

    for v1, v2 in bonelink:
        bone_coordinate[:, :, v2] = joint_coordinate[:, :, v1] - joint_coordinate[:, :, v2]

    coordinates = np.concatenate((joint_coordinate, bone_coordinate), axis=0)
    return coordinates

class DatasetLoader(Dataset):
    def __init__(self,cfg, pretrain, pkl_file):
        self.cfg = cfg
        with open(pkl_file, 'rb') as f:
            self.data_list = pickle.load(f)
            print("pkl_file",pkl_file)
        if not pretrain:
            standard_path = cfg.STANDARD_PATH

            with open(standard_path, 'rb') as f:
                standard_features_file = pickle.load(f)
                self.standard_features_list = [generate_data(item['features']) for item in standard_features_file]

        self.samples = []
        max_len = 0  
        
        if pretrain:
            self.standard_features_list = [generate_data(self.data_list[0]['features'])]

        print('Data List Length:', len(self.data_list))
        index_dict = {}
        for item in self.data_list:
            features =  generate_data(item['features'])
            ## Figure Skating
            if len(self.standard_features_list) == 1:
                    std_features = self.standard_features_list[0]
            else:
                if cfg.TASK.SPORT == 'Skating' :
                    if   'Axel'     in item['original_video_file']:
                        std_features = self.standard_features_list[0]
                    elif 'Axel_com' in item['original_video_file']:
                        std_features = self.standard_features_list[1]
                    elif 'Loop'     in item['original_video_file']:
                        std_features = self.standard_features_list[2]
                    else:
                        std_features = self.standard_features_list[3]

                if cfg.TASK.SPORT == 'Boxing' :
                    if 'back'         in item['video_name']:
                        std_features = self.standard_features_list[0]
                    elif 'front'    in item['video_name']:
                        std_features = self.standard_features_list[1]

            video_name = item['video_name']
            # check_video_name = item['video_name'].rsplit('_', 1)[0]
            # if check_video_name not in video_list:
            #     video_list.append(check_video_name)
            # else:
            #     continue
            trimmed_start = item['trimmed_start'] if 'trimmed_start' in item else 0
            if not item['standard_longer']:
                start_frame = item['start_frame']   + trimmed_start
                end_frame   = item['end_frame']     + trimmed_start
                item['std_start_frame'] = 0
                item['std_end_frame']   = end_frame - start_frame
            else:
                item['std_start_frame'] = item['start_frame']
                item['std_end_frame']   = item['end_frame'] - 1
                start_frame = trimmed_start
                end_frame   = trimmed_start + (item['std_end_frame'] - item['std_start_frame'])
            max_len = max(max_len, len(features[0]))
            
            if 'train' in pkl_file:
                if cfg.TASK.SPORT == 'Skating' :
                    labels = item['augmented_labels']
                    labels.append(item['revised_label'])
                if cfg.TASK.SPORT == 'Boxing' :
                    labels = item['labels']
            ## Dummy label to make sure there is a sample for the video,but the label should be calculated externally. Used for untrimmed evaluation
            ## elif self.cfg.args.no_calc_score:
            ##    labels = ['']
            else:
                # Skating Setting
                if cfg.TASK.SPORT == 'Skating' :
                    if cfg.args.eval_name == 'segment' or cfg.args.eval_name == 'untrimmed':
                        labels = ['']
                    else:
                        labels = [item['revised_label']]

                elif cfg.TASK.SPORT == 'Boxing' or cfg.TASK.SPORT == 'Manipulate':
                    labels = item['labels']
                else:
                    raise Exception(f"Sport {cfg.TASK.SPORT} not supported or should be specified.")
            if hasattr(self.cfg.TASK,'DIFFERENCE_TYPE') and self.cfg.TASK.DIFFERENCE_TYPE== 'RGB':
                subtraction = item['subtraction']
                features = features[:,start_frame:end_frame] 
                subtraction = subtraction[:features.shape[1]]
                assert features.shape[1] == subtraction.shape[0],f"""
                {generate_data(item['features']).shape}
                features.shape[2] = {features.shape[1]} in \n 
                {item['video_file']},\n
                original features.shape[2] = {item['features'].shape[1]}, \b
                subtraction.shape[0] = {subtraction.shape[0]}, \n 
                original_start_frame = {item['start_frame']}, original_end_frame = {item['end_frame']},\n
                trimmed_start = {item['trimmed_start']}, trimmed_end = {item['trimmed_end']}, \n
                standard_longer? {item['standard_longer']},
                """
            elif hasattr(self.cfg.TASK,'DIFFERENCE_TYPE') and self.cfg.TASK.DIFFERENCE_TYPE== 'Skeleton':
                # deal with error segment selection
                if self.cfg.args.eval_name == 'segment': ## if segmenting, need to further adjust the video and the standard
                    print(f"Error segmenting {item['video_name']}")
                    if item['error_end_frame'] == 0:
                        feature_start_frame, feature_end_frame  = 0, 1
                        std_start_frame, std_error_end_frame    = 0, 1
                    else:
                        if not item['standard_longer']:
                            feature_start_frame = start_frame + int(item['error_start_frame'])
                            feature_end_frame   = start_frame + int(item['error_end_frame'])-1
                            std_start_frame     = int(item['error_start_frame'])
                            std_error_end_frame = int(item['error_end_frame']) -1
                        else:
                            feature_start_frame = int(item['trimmed_start'])    +   int(item['error_start_frame'])
                            feature_end_frame   = int(item['trimmed_start'])    + ( int(item['error_end_frame']) - 1)
                            std_start_frame     = int(item['start_frame'])      +   int(item['error_start_frame'])
                            std_error_end_frame = int(item['start_frame'])      + ( int(item['error_end_frame']) - 1)
                    features        = features[:,int(feature_start_frame):int(feature_end_frame)]
                    std_features    = std_features[:,int(std_start_frame):int(std_error_end_frame)]
                else:
                    features        = features[:,int(start_frame):int(end_frame)] 
                    std_features    = std_features[:,int(item['std_start_frame']):int(item['std_end_frame'])]
                    index_dict[item['video_name']] = {
                        # "original_seq_len"  : int(item['original_seq_len']),
                        "start_frame" : start_frame,
                        "end_frame" : end_frame,
                        "seq_len" : int(end_frame-start_frame),
                        "feature_start_frame" : int(start_frame),
                        "feature_end_frame" : int(end_frame),
                        "std_start_frame" : int(item['std_start_frame']),
                        "std_error_end_frame" : int(item['std_end_frame']),
                    }
                # subtraction = item['subtraction']
                subtraction = torch.empty(0)
                if features.shape[1] !=  std_features.shape[1]:
                    shorter         = min(features.shape[1],std_features.shape[1])
                    features        = features[:,:shorter]
                    std_features    = std_features[:,:shorter]
                assert features.shape[1] ==  std_features.shape[1],f"""
                features.shape: {features.shape}
                std_features.shape: {std_features.shape}
                item["features"].shape: {item["features"].shape}
                features.shape[2] = {features.shape[1]} in \n 
                {item['video_file']},\n
                original features.shape[2] = {item['features'].shape[1]}, \b
                std_features.shape[1] = {std_features.shape[1]}, \n 
                original_start_frame = {item['start_frame']}, original_end_frame = {item['end_frame']},\n
                trimmed_start = {item['trimmed_start']}, trimmed_end = {item['trimmed_end']}, \n
                standard_longer? {item['standard_longer']},
                """
            else:
                subtraction = torch.empty(0)
            newlabels = ["Motion Description : " + label if pretrain else "Motion Instruction : " + label for label in labels]
            for label in newlabels:
                if features.shape[1] == 0 or features.shape[1] == 1:
                    print(f"Skipping {video_name} as no frames found")
                    continue
                self.samples.append((features, label, video_name,subtraction, std_features, newlabels))

        # generate a tensor that is zero, shape is (64,128)
        # self.samples.append((self.standard_features_list[0], '', 'back',      torch.zeros(self.standard_features_list[0].shape[1],128), self.standard_features_list[0])) 
        # self.samples.append((self.standard_features_list[1], '', 'front',     torch.zeros(self.standard_features_list[1].shape[1],128), self.standard_features_list[1])) 
        # self.samples.append((self.standard_features_list[0], '', 'Axel',      torch.zeros(self.standard_features_list[0].shape[1],128), self.standard_features_list[0])) 
        # self.samples.append((self.standard_features_list[1], '', 'Axel_com',  torch.zeros(self.standard_features_list[1].shape[1],128), self.standard_features_list[1])) 
        # self.samples.append((self.standard_features_list[2], '', 'Loop',      torch.zeros(self.standard_features_list[2].shape[1],128), self.standard_features_list[2])) 
        # self.samples.append((self.standard_features_list[3], '', 'Lutz',      torch.zeros(self.standard_features_list[3].shape[1],128), self.standard_features_list[3])) 
        print('Sample length:', len(self.samples))
        self.max_len = max_len  
        if (cfg.args.eval_name != 'untrimmed' and cfg.TASK.SPORT != 'Boxing' and cfg.TASK.DIFFERENCE_SETTING != 'No'):
            with open(cfg.JSONDIR+'/index_dict_results.json', 'w') as f:
                json.dump(index_dict, f, indent=4)
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # features: 6 x frame x 22
        features, label, video_name, subtraction, std_features, labels = self.samples[idx]
        keypoints_mask  = torch.ones(22)       
        current_len     = torch.tensor(len(features[0]))

        # change self.standard to std_features
        return  video_name, torch.FloatTensor(features), torch.FloatTensor(keypoints_mask),  torch.FloatTensor(std_features), current_len, label, subtraction, labels
