from bert_score import score
from nlgmetricverse import NLGMetricverse,load_metric
import pickle , os, json
## calculate scores
def calculate_scores(predictions,gts):
        metrics = [
            load_metric("bleu",resulting_name="bleu_1",compute_kwargs={"max_order":1}),
            load_metric("bleu",resulting_name="bleu_4",compute_kwargs={"max_order":4}),
            load_metric("rouge"),
            load_metric("cider"),
        ]
        Evaluator = NLGMetricverse(metrics)

        ## need to convert predictions and gts to list to fit with bert_score
        ### make sure predictions and gts are in the same order
        predictions = dict(sorted(predictions.items()))
        gts = dict(sorted(gts.items()))

        predictions = list(predictions.values())
        gts = list(gts.values())

        scores = Evaluator(predictions=predictions,references=gts)
        score_results = {}
        score_results["bleu_1"] = scores["bleu_1"]['score']
        score_results["bleu_4"] = scores["bleu_4"]['score']
        score_results["rouge"] = scores["rouge"]['rougeL']
        score_results["cider"] = scores["cider"]['score']

        P,R,F1 = score(predictions,gts,lang="en",verbose=False,idf=True,rescale_with_baseline=True)
        score_results["bertscore"] = F1.mean().item()
        return score_results
def gts():
    pkl_file = "/home/c1l1mo/projects/VideoAlignment/result/scl_skating_long_50/output_test_label_para6.pkl"
    groud_truth = {}
    with open(pkl_file, 'rb') as f:
        data_list = pickle.load(f)
        for item in data_list:
            # item.video_name and item.labels
            if item['video_name'] == 'standard':
                continue
            groud_truth[item['video_name']] = item['labels']
        return groud_truth

def main():
    groud_truth = gts()
    path_name = "/home/weihsin/projects/MotionExpert_prtrain_gt.json"
    '''
    All_file = {}
    folder_path = "/home/weihsin/projects/MotionExpert/STAGCN_output_local_new"
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
                   if 'Motion Instruction : ' in v:
                       v = v.replace('Motion Instruction : ', '')
                   predictions[k] = v
                All_file[file_name] = calculate_scores(predictions,groud_truth)
    '''
    All_file = {}
    TM2T_path_name = "/home/weihsin/projects/MotionExpert/_TM2T.json"
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
    #All_file 先用 bertscore sort 再用其他的 bleu1, bleu4, rouge, cider
    All_file = dict(sorted(All_file.items(), key=lambda item: item[1]['bertscore'], reverse=True))
    path_name = "/home/weihsin/projects/TM2T_score.json"
    with open(path_name, 'w') as f:
        json.dump(All_file, f, indent=4)
    
if __name__ == "__main__" :
    main()