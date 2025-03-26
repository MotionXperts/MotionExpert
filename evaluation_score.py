import pickle , os, json, re, argparse, warnings
from bert_score import score
from nlgmetricverse import NLGMetricverse, load_metric
from cider import readJSON, readPickle, getGTCaptions, BLEUScore, CIDERScore
from collections import Counter
from tqdm import tqdm

def read_data(path) :
    with open(path, 'r', encoding = 'utf-8') as f :
        ground_truth = json.load(f)
    return ground_truth

def calculate_scores(cfg, predictions,gts) :
        metrics = [load_metric("bleu", resulting_name = "bleu_1", compute_kwargs = {"max_order" : 1}),
                   load_metric("bleu", resulting_name = "bleu_4", compute_kwargs = {"max_order" : 4}),
                   load_metric("rouge"),
                   load_metric("cider")]

        Evaluator = NLGMetricverse(metrics)
        # Convert predictions and gts to list to fit with bert_score.
        # Make sure predictions and gts are in the same order.
        ori_predictions = dict(sorted(predictions.items()))
        ori_gts = dict(sorted(gts.items()))

        predictions = list(ori_predictions.values())
        gts = list(ori_gts.values())
        if cfg.TASK.SPORT == "Skating" :
            scores = Evaluator(predictions = predictions, references = gts, reduce_fn = "mean")

        elif cfg.TASK.SPORT == "Boxing" :
            scores = Evaluator(predictions = predictions,references = gts, reduce_fn = "max")
        score_results = {}
        score_results["bleu_1"] = scores["bleu_1"]['score']
        score_results["bleu_4"] = scores["bleu_4"]['score']
        score_results["rouge"] = scores["rouge"]['rougeL']
        score_results["cider"] = scores["cider"]['score']
        P, R, F1 = score(predictions, gts, lang = "en", verbose = False, idf = True, rescale_with_baseline = True)
        score_results["bertscore"] = F1.mean().item()

        if (cfg.TASK.SPORT == "Boxing") :
            F1_max_all = []
            # Enumerate over two dictionaries simultaneously.
            for item in tqdm(ori_predictions) :
                F1_all = []
                each_gts = ori_gts[item]
                each_predictions = ori_predictions[item]
                for one_gts in each_gts :
                    try :
                        P_each, R_each, F1_each = score([each_predictions], [one_gts], lang = "en",
                                                        verbose=False, idf = False,
                                                        rescale_with_baseline = True)
                    except :
                        F1_each = 0
                    F1_all.append(F1_each)
                F1_max = max(F1_all)
                F1_max_all.append(F1_max)
            F1_max_average = sum(F1_max_all) / len(F1_max_all)
            score_results["bertscore"] = float(F1_max_average[0])
            print("F1_max_average", F1_max_average)
        return score_results

def gts(cfg) :
    # pkl_file = "../datasets/scripts/skating_pipeline/Skating_GT_test/aggregate.pkl"
    # pkl_file = "../datasets/FigureSkate/HumanML3D_l/local_human_test.pkl"
    if cfg.TASK.SPORT == "Skating" :
        pkl_file = "../../../datasets/scripts/skating_pipeline/Skating_GT_test/aggregate.pkl"
    elif cfg.TASK.SPORT == "Boxing" :
        pkl_file = "../../../datasets/BoxingDatasetPkl/boxing_GT_test_aggregate.pkl"
        pkl_file = "../../../datasets/BoxingDatasetPkl/boxing_GT_test.pkl"
    annotations = readPickle(pkl_file)
    gts = getGTCaptions(cfg,annotations)
    # segment_gt_path = "./results/finetune_error_seg/jsons/segment_gt.json"
    # gts = read_data(segment_gt_path)
    return gts

def main() :
    warnings.filterwarnings("ignore")
    cfg = argparse.Namespace()
    cfg.TASK = argparse.Namespace()
    cfg.TASK.SPORT = "Skating"
    cfg.TASK.SPORT = "Boxing"
    ground_truth = gts(cfg)

    All_file = {}
    # Motion Description
    # folder_path = "./STAGCN_output_local_new"
    # Motion Instruction
    # folder_path = "./STAGCN_output_finetune_new2"
    # folder_path = "./results/finetune_error_seg/jsons"
    # folder_path = "./results/finetune_skeleton_t5_test/jsons"
    # folder_path = "./results/finetune_boxing_error/jsons"
    # folder_path = "./results/finetune_skating_no_ref/jsons"
    # folder_path = "./results/finetune_boxing_no_ref/jsons"
    # folder_path = "./results/finetune_boxing_new/jsons"
    # folder_path = "./results/finetune_boxing/jsons"
    # folder_path = "./results/finetune_boxing_0303/jsons"
    # folder_path = "./results/boxing_0304/jsons"
    epoch_pattern = re.compile(r"^results_epoch(\d+)\.json$")
    for file_name in os.listdir(folder_path) :
        match = epoch_pattern.match(file_name)
        if match :
            epoch_num = int(match.group(1))
        if file_name.endswith('.json') and file_name.startswith('results_epoch'):
            file_path = os.path.join(folder_path, file_name)
            predictions = {}
            with open(file_path, 'r') as f :
                json_data = json.load(f)
                for(k, v) in json_data.items() :
                    if 'Motion Instruction : ' in v :
                       v = v.replace('Motion Instruction : ', '')
                    predictions[k] = v

                value_counts = Counter(predictions.values())
                most_common_value, most_common_count = max(value_counts.items(), key = lambda x: x[1])
                All_file[file_name] = {"scores": calculate_scores(cfg, predictions, ground_truth),
                                       "most_common_value" : most_common_value,
                                       "most_common_count" : most_common_count}

    # All_file calculate bertscore sort and then calculate bleu1, bleu4, rouge, cider
    # All_file = dict(sorted(All_file.items(), key=lambda item: item[1]['scores']['bertscore'], reverse=True))
    All_file = dict(sorted(All_file.items(), key = lambda item: item[1]['most_common_count'], reverse = True))

    # path_name = 'lora_skating_t5_6_bertscore.json'
    # path_name = 'error_segment_bertscore.json'
    # path_name = 'finetune_skating_test_bertscore.json'
    # path_name = 'boxing_error_test_bertscore.json'
    # path_name = './RGB_boxing_bertscore.json'
    # path_name = './boxing_no_ref_test_bertscore100_200.json'
    # path_name = './results/finetune_boxing_new/max_value.json'
    # path_name = './results/finetune_boxing/max_value.json'
    # path_name = './results/finetune_boxing_0303/max_value.json'
    # path_name = './results/boxing_0304/max_value.json'
    path_name = './results/boxing_0304/boxing_no_ref_test_bertscore.json'
    with open(path_name, 'w') as f:
        json.dump(All_file, f, indent=4)

if __name__ == "__main__" :
    main()