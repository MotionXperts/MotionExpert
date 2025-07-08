import json, argparse, tqdm, time, os, re
import anthropic
from dotenv import load_dotenv
from detection import read_template, read_data, acc, g_eval
# BodyPart
indicator = 'BodyPart'

# Usage :
# $ python SportIndicator/BodyPart_evaluation.py

if __name__ == '__main__':
    load_dotenv()
    argparser = argparse.ArgumentParser()

    # BodyPart
    argparser.add_argument('--prompt_fp', type=str, default='./SportIndicator/GEval_template/GEval_Bodypart_template.txt')
    # Skating
    # argparser.add_argument('--predict', type=str, default='./SportIndicator/geval_epoch_135.json')
    # argparser.add_argument('--predict', type=str, default='./SportIndicator/geval_instruction_by_GPT4-o.json')
    # argparser.add_argument('--predict', type=str, default='./SportIndicator/geval_llama32_skating.json')      
    # argparser.add_argument('--output', type=str, default="./results/boxing_aligned/.")
    # argparser.add_argument('--output', type=str, default="./results/boxing_llama/.")
    # argparser.add_argument('--output', type=str, default="./results/boxing_gpt/.")
    # argparser.add_argument('--output', type=str, default="./results/GT_BX/.")
    argparser.add_argument('--output', type=str, default="./results/GT_FS/.")
    # Boxing
    # argparser.add_argument('--predict', type=str, default='./SportIndicator/geval_boxing_epoch70.json')
    # argparser.add_argument('--predict', type=str, default="./results/boxing_aligned/geval/geval_epoch_115.json")
    # argparser.add_argument('--predict', type=str, default="./results/boxing_llama/geval/geval_epoch_31.json")
    # argparser.add_argument('--predict', type=str, default="./results/boxing_gpt/geval/geval_epoch_31.json")
    # argparser.add_argument('--predict', type=str, default="./results/GT_BX/geval/results.json")
    argparser.add_argument('--predict', type=str, default="./results/GT_FS/geval/results.json")
    args = argparser.parse_args()

    api_key     = os.getenv("ANTHROPIC_KEY")
    prompt      = read_template(args.prompt_fp)

    Scores      = {}
    filename    = os.path.basename(args.predict)
    filename    = os.path.splitext(filename)[0]

    all_filename = indicator + f'_Detection_avg{filename}.json'
    all_filepath = os.path.join(args.output, all_filename)
        
    file_path   = os.path.join(args.predict, filename)
            
    # G-eval
    results             = read_data(args.predict)
    avg_score, score    = g_eval(args, results, prompt, api_key, filename, indicator)
    avg_score_name      = indicator + '_Detection_avg_score'
    score_name          = indicator + '_Detection_score'

    Scores[filename] = {avg_score_name : avg_score,
                        score_name : score}

    # Accuracy
    acc_score_name      = indicator + '_acc_score'
    acc_shot_count_name = indicator + '_acc_shot_count'
    acc_name             = indicator + '_acc'
    acc_score, acc_shot_count = acc(args, filename, indicator)
    
    Scores[acc_score_name] = acc_score
    Scores[acc_shot_count_name] = acc_shot_count
    Scores[acc_name] = acc_score/acc_shot_count

    with open(all_filepath, 'w') as f:
        json.dump(Scores, f, indent=4)