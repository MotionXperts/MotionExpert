import os
import json
from collections import defaultdict
from argparse import ArgumentParser
from PIL import Image
import torch
import language_evaluation
import pickle

def readJSON(file_path):
    try:
        with open(file_path) as f:
            data = json.load(f)
        return data
    except Exception as e:
        raise Exception(f"Error reading json file: {e}")

def readPickle(file_path):
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        return data
    except:
        return None

def getGTCaptions(cfg, annotations):
    video_name_to_gts = defaultdict(list)
    for item in annotations:
        video_name = item['video_name']
        output_sentence = item['labels']
        video_name_to_gts[video_name] = output_sentence
    return video_name_to_gts

class BLEUScore:
    def __init__(self):
        self.evaluator = language_evaluation.CocoEvaluator(coco_types=["BLEU"])

    def __call__(self, predictions, gts):
        predicts = []
        answers = []
        for img_name in predictions.keys():
            predicts.append(predictions[img_name])
            answers.append(gts[img_name][0] if isinstance(gts[img_name], list) else gts[img_name])

        results = self.evaluator.run_evaluation(predicts, answers)
        return results

class CIDERScore:
    def __init__(self):
        self.evaluator = language_evaluation.CocoEvaluator(coco_types=["CIDEr"])

    def __call__(self, predictions, gts):
        predicts = []
        answers = []
        for img_name in predictions.keys():
            predicts.append(predictions[img_name])
            #answers.append(gts[img_name])
            answers.append(gts[img_name][0] if isinstance(gts[img_name], list) else gts[img_name])
        results = self.evaluator.run_evaluation(predicts, answers)
        return results['CIDEr']
