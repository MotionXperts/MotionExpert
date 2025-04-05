from bert_score import score
from nlgmetricverse import NLGMetricverse,load_metric
from cider import readJSON, readPickle, getGTCaptions, BLEUScore, CIDERScore
from collections import Counter
import json, argparse, warnings, openai, os
from tqdm import tqdm
from utils.retrieve_most_similar_label import compute_similar_score

def  calculate(label, labels):
        print("label",label)
        print("labels",labels)
        labels = [labels] if isinstance(labels, str) else labels
        key = os.getenv("OPENAI_KEY")
        prompt = f"""
            Given the following choices:
            {labels},
            which one is the most similar to the label {label}?

            Reply with the following template:
            "The most similar label is: <index>"

            Only put number in the <index> field and do not include any other information except for the template.
            """
        openai.api_key = key
        response = openai.chat.completions.create(model="gpt-4o-2024-08-06",
                                                  messages=[{"role": "user",
                                                             "content": f'Original instruction: "{prompt}"'}])
        string = response.choices[0].message.content
        index = string.split('The most similar label is: ')[-1].replace("\"","").replace(".","")
        gt = ""
        try:
            gt = labels[int(index)]
        except:
            gt = labels[0]
        return gt

def gts(cfg):
    # pkl_file = "/home/c1l1mo/datasets/scripts/skating_pipeline/Skating_GT_test/aggregate.pkl"
    # pkl_file = "/home/weihsin/datasets/FigureSkate/HumanML3D_l/local_human_test.pkl"
    # pkl_file = "/home/c1l1mo/datasets/boxing_safetrim/boxing_GT_test/aggregate.pkl"

    if cfg.TASK.SPORT == "Skating" :
        if cfg.TASK.Setting == "GT" :
            pkl_file = "/home/weihsin/datasets/SkatingDatasetPkl/skating_gt_test.pkl"
        elif cfg.TASK.Setting == "UNTRIMMED" :
            pkl_file = "/home/weihsin/datasets/SkatingDatasetPkl/skating_untrimmed.pkl"
        elif cfg.TASK.Setting == "SEGMENT" :
            pkl_file = "/home/weihsin/datasets/SkatingDatasetPkl/skating_segment.pkl"
    elif cfg.TASK.SPORT == "Boxing" :
        if cfg.TASK.Setting == "GT" :
            pkl_file = "/home/weihsin/datasets/BoxingDatasetPkl/boxing_GT_test.pkl"
        elif cfg.TASK.Setting == "SEGMENT" :
            pkl_file = "/home/andrewchen/Error_Localize/boxing_aggregate_error_segment.pkl"

    annotations = readPickle(pkl_file)
    gts = getGTCaptions(cfg,annotations)

    return gts

# Usage : First select the target epoch need to generate ground truth.
def main():
    warnings.filterwarnings("ignore")
    cfg = argparse.Namespace()
    cfg.TASK = argparse.Namespace()
    # cfg.TASK.SPORT = "Skating"
    cfg.TASK.SPORT = "Boxing"
    cfg.TASK.Setting = "GT"
    ground_truth = gts(cfg)

    All_file = {}

    target_dir = "./results/finetune_boxing_no_ref"
    epoch_number = "75"
    target_dir = "./results/boxing_0304"
    epoch_number = "70"
    target_dir = "./results/boxing_evaluation"
    epoch_number = "175"
    file_name = os.path.join(target_dir, f"jsons/results_epoch{epoch_number}.json")
    predictions = {}
    with open(file_name, 'r') as f:
                json_data = json.load(f)
                for(k, v) in json_data.items():
                    if k == 'standard' :
                        print("standard")
                        continue
                    if 'Motion Instruction : ' in v:
                       v = v.replace('Motion Instruction : ', '')
                    if v == "" :
                        continue

                    predictions[k] = v

                for k in predictions :
                    ground_truth[k] = calculate(predictions[k], ground_truth[k])

    # All_file calculate bertscore sort and then calculate bleu1, bleu4, rouge, cider
    All_file = dict(sorted(All_file.items(), key = lambda item: item[1]['scores']['bertscore'], reverse = True))

    output_gt_path = os.path.join(target_dir, f"jsons/gt.json")
    with open(output_gt_path, 'w') as f :
        json.dump(ground_truth, f, indent = 4)

if __name__ == "__main__" :
    main()