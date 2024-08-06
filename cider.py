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
    except:
        return None

def readPickle(file_path):
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        return data
    except:
        return None

def getGTCaptions(annotations):
    video_name_to_gts = defaultdict(list)
    for item in annotations:
        video_name = item['video_name']
        output_sentence = item['revised_label']
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

def main(args):
    # Read data
    predictions = readJSON(args.pred_file)
    annotations = readPickle(args.annotation_file)

    # Preprocess annotation file
    gts = getGTCaptions(annotations)

    # Check predictions content is correct
    assert type(predictions) is dict
    assert set(predictions.keys()) == set(gts.keys())
    assert all([type(pred) is str for pred in predictions.values()])

    # CIDErScore
    cider_score = CIDERScore()(predictions, gts)
    bleu_score = BLEUScore()(predictions, gts)

    print(f"CIDEr: {cider_score}")
    print(f"BLEU: {bleu_score}")    


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--pred_file",default='/home/peihsin/projects/MotionExpert/results_epoch15.json', help="Prediction json file")
    parser.add_argument("--annotation_file", default="/home/peihsin/projects/humanML/dataset/rm_test.pkl", help="Annotation json file")

    args = parser.parse_args()

    main(args)