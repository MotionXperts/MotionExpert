from bert_score import score
from nlgmetricverse import NLGMetricverse,load_metric
import pickle , os, json
import torch

def find_ground_truth(filename, ground_truths):
    for data in ground_truths:
        if (data["video_name"] == filename) :
            return data["labels"]

def read_data(ground_truth_path, predict_path, ground_truth_train_path = None) :
    with open(ground_truth_path, 'r', encoding = 'utf-8') as f :
        ground_truth = json.load(f)
    if ground_truth_train_path != None:
        with open(ground_truth_train_path, 'r', encoding = 'utf-8') as f :
            ground_truth_train = json.load(f)
        ground_truth = ground_truth_train + ground_truth
    with open(predict_path, 'r', encoding = 'utf-8') as f :
        predictions = json.load(f)
    results = {}
    for key in predictions.keys() :
        print(key)
        results[key] = find_ground_truth(key, ground_truth)
    return results, predictions

# calculate scores
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

        scores = Evaluator(predictions = predictions, references = gts, reduce_fn = "max")

        score_results = {}
        score_results["bleu_1"] = scores["bleu_1"]['score']
        score_results["bleu_4"] = scores["bleu_4"]['score']
        score_results["rouge"] = scores["rouge"]['rougeL']
        score_results["cider"] = scores["cider"]['score']

        is_multi_candidate = isinstance(predictions[0], list)

        if not is_multi_candidate :
            P,R,F1 = score(predictions, gts, lang = "en", verbose = False, idf = True, rescale_with_baseline = True)
            score_results["bertscore"] = F1.mean().item()
        else :
            best_predictions = []
            for prediction, gt in zip(predictions, gts):
                references = [gt] * len(prediction)

                P, R, F1 = score(prediction, references, lang = "en", verbose = False, idf = True, rescale_with_baseline = True)

                best_idx = F1.argmax().item()
                best_predictions.append(prediction[best_idx])

                print(f"Group predictions: {prediction}")
                print(f"BERTScores (F1): {[round(f.item(), 4) for f in F1]}")
                print(f"Best: {prediction[best_idx]}\n")
            P, R, F1 = score(best_predictions, gts, lang = "en", verbose = False, idf = True, rescale_with_baseline = True)
            score_results["bertscore"] = F1.mean().item()
        return score_results

def main():
    ground_truth_path = "./dataset/BX_test.json"
    ground_truth_train_path = "./dataset/BX_train.json"
    path_name = "./results/boxing_aligned/metrics.json"
    All_file = {}
    folder_path = "./results/boxing_aligned/best_json/"
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.json') and file_name.startswith('results_epoch'):
            file_path = os.path.join(folder_path, file_name)
            groud_truth, predictions = read_data(ground_truth_path, file_path, ground_truth_train_path)
            All_file[file_name] = calculate_scores(predictions, groud_truth)

    All_file = dict(sorted(All_file.items(), key=lambda item: item[1]['bertscore'], reverse=True))
    with open(path_name, 'w') as f:
        json.dump(All_file, f, indent=4)
    
if __name__ == "__main__" :
    main()