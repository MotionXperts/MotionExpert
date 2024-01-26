"""
Intent: Create the pickle file from the skeleton folder and the label folder (Needed to be flatten before use)
Author: Tom
Last update: 2023/09/20
"""
import pickle
import numpy as np
import json
import os
import cv2
from transformers import BertTokenizer, AutoTokenizer
from argparse import ArgumentParser, Namespace


tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--output_dir", type=str, default="/home/weihsin/datasets/FigureSkate/total", help="Path to the output folder"
    )
    parser.add_argument(
        "--skeleton_dir", type=str, default="/home/weihsin/datasets/FigureSkate/total", help="Path to the skeleton folder"
    )
    parser.add_argument(
        "--label_dir", type=str, default="/home/weihsin/datasets/FigureSkateOri/total", help="Path to the label folder"
    )
    parser.add_argument(
        "--complete", "-c", action="store_true", help="Whether to form complete version of pickle file"
    )
    args = parser.parse_args()
    return args


def xyxy2xywh(bbox):
    x1, y1, x2, y2 = bbox

    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    return [cx, cy, w, h]


def label_fn_package(context_list): 
    encoding = tokenizer( context_list, padding=True)
    output_final = encoding.input_ids
    token_type_ids = encoding.token_type_ids 
    attention_mask = encoding.attention_mask
    return output_final, token_type_ids, attention_mask


def build_file2label(label_dir):
    """
    Input: a list of label filenames (.json)
    Output: a dictionary<filename, label>   # filename only contains the base filename without extension
    """
    # Fetch all the label files
    label_files = [f for f in os.listdir(label_dir) if f.endswith('.json')]
    file2label = dict()
    for file in label_files:
        filename, extension = os.path.splitext(file)
        with open(os.path.join(label_dir, file), 'r') as json_file:
            data = json.load(json_file)
            for datum in data:
                file2label[filename + "_" + str(datum['id'])] = datum['context']
    return file2label


def main(args):
    # Load in directory paths
    output_dir = args.output_dir
    skeleton_dir = args.skeleton_dir
    label_dir = args.label_dir
    complete = args.complete
    
    # Fetch all the skeleton files
    skeleton_files = [ os.path.join(root, file) for root, _, files in os.walk(skeleton_dir) for file in files if file.endswith('.pk')]
    # Build the dictionary mapping from filename to label
    file2label = build_file2label(label_dir)
    
    output_data = []
    for skeleton_file in skeleton_files:
        base_dir = os.path.basename(os.path.dirname(skeleton_file))     # video_name
        label = file2label[base_dir]
        with open(skeleton_file, "rb") as sk_file:
            skeleton = pickle.load(sk_file)
            features = skeleton['pred_xyz_24_struct']
            bboxes = skeleton['bbox']
            num_frame = len(features)

            # Transform coordinate system
            # for i in range(num_frame):
                # bbox_xywh = xyxy2xywh(bboxes[i])
                # features[i] = features[i] * bbox_xywh[2]
                # features[i][:, 0] = features[i][:, 0] + bbox_xywh[0]
                # features[i][:, 1] = features[i][:, 1] + bbox_xywh[1]
                
                # features[i][:, 0] -= skeleton['transl_camsys'][i][0] 
                # features[i][:, 1] -= skeleton['transl_camsys'][i][1]
                # features[i][:, 2] -= skeleton['transl_camsys'][i][2]
                # features[i][:, 0] += skeleton['pred_cam_root'][i][0]
                # features[i][:, 1] += skeleton['pred_cam_root'][i][1]
                # features[i][:, 2] += skeleton['pred_cam_root'][i][2]

                # Transform: y = z; z = y
                # features[i][:, 1], features[i][:, 2] = features[i][:, 2], features[i][:, 1]
                # for j in range(len(features[i])):
                #     features[i][j][1], features[i][j][2] = features[i][j][2], features[i][j][1]

            # Extract 22 key points & Flatten out the skeleton key points
            features = np.array([ feature[0:22].flatten() for feature in features ])
            output, token_type_ids, attention_mask = label_fn_package(label[0])
            output_data.append({
                'video_name': base_dir,
                'features': features,
                'labels': label,
                'output_sentence': label[0],
                'output': output,
                'token_type_ids': token_type_ids,
                'attention_mask': attention_mask
            })
            # Choose what to output
            # if complete:
            #     output_data.append({
            #         'video_name': base_dir,
            #         'features': features,
            #         'labels': label,
            #         'output_sentence': label[0],
            #         'output': output,
            #         'token_type_ids': token_type_ids,
            #         'attention_mask': attention_mask
            #     })
            # else:
            #     output_data.append({
            #         'video_name': base_dir,
            #         'features': features,
            #         'labels': label,
            #     })
            
    # Write out the file
    output_file = os.path.join(output_dir, 'data.pkl')
    with open(output_file, 'wb') as file:
        pickle.dump(output_data, file)

if __name__ == "__main__":
    args = parse_args()
    main(args)
    