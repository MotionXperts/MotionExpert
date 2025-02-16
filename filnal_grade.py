import openai
import json, argparse, tqdm, time, os, re
import anthropic
from dotenv import load_dotenv

def read_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        datas = json.load(f)
    total_score = 0
    count = 0
    for data in datas:
        if data['score'] :
            print(data)
            total_score += data['score']
            count += 1
    
    return total_score / count


if __name__ == '__main__':
    load_dotenv()
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--prompt_fp', type=str, default='./utils/claude.txt')
    argparser.add_argument('--save_fp', type=str, default='./gpt4_con_detailed_openai.json')
    argparser.add_argument('--ground_truth', type=str, default='./ground_truth_test.json')
    argparser.add_argument('--predict', type=str, default='./results/finetune_skeleton_t5_6/jsons/.')
    argparser.add_argument('--output', type=str, default='./results/finetune_skeleton_t5_6/geval')
    args = argparser.parse_args()
    api_key = os.getenv("ANTHROPIC_KEY")

    all_files = os.listdir(args.output)

    filtered_files = [
        f for f in all_files 
        if f.endswith('.json') and f.startswith('geval_epoch')
    ]

    

    all_epoch = {}
    all_filename = 'allgeval.json'
    all_filepath = os.path.join(args.output,all_filename)
    for file_name in filtered_files :
        file_path = os.path.join(args.output, file_name)
        try:
            epoch = int(file_name.split('epoch_')[1].split('.')[0])
        except:
            epoch = '125_o'
        print(file_name)
        score = read_data(file_path)
        all_epoch[file_name] = {  'score' : score,
                                  'epoch' : epoch}

    sorted_all_epoch = dict(sorted(all_epoch.items(), key=lambda item: item[1]['score'], reverse=True))

    with open(all_filepath, 'w') as f:
        json.dump(sorted_all_epoch, f, indent=4)