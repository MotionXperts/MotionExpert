from bert_score import score
from nlgmetricverse import NLGMetricverse,load_metric
import pickle , os, json
import torch
## calculate scores
def calculate_scores(predictions,gts):
        metrics = [
            load_metric("bleu",resulting_name="bleu_1",compute_kwargs={"max_order":1}),
            load_metric("bleu",resulting_name="bleu_4",compute_kwargs={"max_order":4}),
            load_metric("rouge"),
            load_metric("cider"),
            load_metric("bertscore", resulting_name="bertscore_1"),
        ]
        Evaluator = NLGMetricverse(metrics)

        ## need to convert predictions and gts to list to fit with bert_score
        ### make sure predictions and gts are in the same order
        predictions = dict(sorted(predictions.items()))
        gts = dict(sorted(gts.items()))

        predictions = list(predictions.values())
        gts = list(gts.values())
        
        scores = Evaluator(predictions=predictions,references=gts, reduce_fn="max")
                
        score_results = {}
        score_results["bleu_1"] = scores["bleu_1"]['score']
        score_results["bleu_4"] = scores["bleu_4"]['score']
        score_results["rouge"] = scores["rouge"]['rougeL']
        score_results["cider"] = scores["cider"]['score']
        # score_results["bertscore"] = scores["bertscore_1"]['score']
        P,R,F1 = score(predictions,gts,lang="en",verbose=False,idf=True,rescale_with_baseline=True)
        score_results["bertscore"] = F1.mean().item()
        # max_F1 = torch.max(F1, dim=1)[0]
        # print("F1:", F1)
        # score_results["bertscore"] = F1.mean().item()
        return score_results
def gts():
    # pkl_file = "./datasets/boxing_safetrim/boxing_GT_test/aggregate.pkl"
    pkl_file = './Error_Localize/aggregate_vibe.pkl'
    groud_truth = {}
    with open(pkl_file, 'rb') as f:
        data_list = pickle.load(f)
        for item in data_list:
            # item.video_name and item.labels
            # if item['video_name'] == 'standard':
            #     continue
            groud_truth[item['video_name']] = item['revised_label']
            # groud_truth[item['video_name']] = item['labels']
        return groud_truth

def main():
    groud_truth = gts()
    path_name = "./MotionExpert_v2/MotionExpert/results/finetune_skeleton/Metrics_ground_truth_mean.json"
    # path_name = './MotionExpert_v2/boxing_eval/Metrics_ground_truth_bertMean_otherMax.json'
    All_file = {}
    folder_path = "./MotionExpert_v2/MotionExpert/vibe_eval"
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.json') and file_name.startswith('results_epoch'):
            file_path = os.path.join(folder_path, file_name)
            predictions = {}
            with open(file_path, 'r') as f:
                json_data = json.load(f)
                for(k, v) in json_data.items():
                   if k == 'standard' :
                        print("standard")
                        continue
                   predictions[k] = v
                All_file[file_name] = calculate_scores(predictions,groud_truth)
    '''
    All_file = {}
    TM2T_path_name = "./projects/MotionExpert/_TM2T.json"
    predictions = {}
    with open(TM2T_path_name, 'r') as f:
        json_data = json.load(f)
        for(k, v) in json_data.items():
                   if k == 'standard' :
                        print("standard")
                        continue
                   if 'Motion Instruction : ' in v:
                       v = v.replace('Motion Instruction : ', '')
                   predictions[k] = v
        All_file['TM2T'] = calculate_scores(predictions,groud_truth)
    '''
    #All_file 先用 bertscore sort 再用其他的 bleu1, bleu4, rouge, cider
    All_file = dict(sorted(All_file.items(), key=lambda item: item[1]['bertscore'], reverse=True))
    with open(path_name, 'w') as f:
        json.dump(All_file, f, indent=4)
    
if __name__ == "__main__" :
    main()