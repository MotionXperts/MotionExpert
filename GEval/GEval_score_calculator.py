import openai, json, argparse, tqdm, time, os, re, anthropic
from dotenv import load_dotenv

def read_template(prompt_fp):
    text_template = open(args.prompt_fp).read()
    return text_template

def read_data(ground_truth_path, predict_path) :
    with open(ground_truth_path, 'r', encoding = 'utf-8') as f :
        ground_truth = json.load(f)
    
    with open(predict_path, 'r', encoding = 'utf-8') as f :
        predictions = json.load(f)
    results = []
    for key in ground_truth.keys() :
        results.append({"file_name" : key,
                        "source" : ground_truth[key],
                        "system_output" : predictions.get(key, "")})
    return results

def g_eval(args,summeval, prompt, api_key, epoch) :
    new_json = []
    count, ignore, all_score = 0, 0, 0
    for instance in tqdm.tqdm(summeval) :
        source = instance['source']
        system_output = instance['system_output']
        # Handle output from GPT4_o.
        if type(system_output) == list :
            system_output = system_output[0]
        cur_prompt = prompt.replace('{{Document}}', source).replace('{{Summary}}', system_output)
        instance['prompt'] = cur_prompt
        
        client = anthropic.Anthropic(api_key = api_key)
        try:
            _response = client.messages.create(model = "claude-3-5-sonnet-20240620",
                                               messages = [{"role" : "user", "content" : cur_prompt}],
                                               max_tokens = 5)
            content = _response.content
            text_value = content[0].text
            match = re.search(r'\d+', text_value)
            instance['all_responses'] = text_value
            instance['score'] = int(match.group())
            
            new_json.append(instance)

            count += 1
            all_score += instance['score']

        except Exception as e:
            print('Exception:', e)
            ignore += 1
            
    output_filename = f'geval_epoch_{epoch}.json'
    output_filepath = os.path.join(args.output, output_filename)
    with open(output_filepath, 'w') as f :
        json.dump(new_json, f, indent = 4)
    return all_score / count

if __name__ == '__main__' :
    load_dotenv()
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--prompt_fp', type = str, default = './GEval_Consistency_Template.txt')

    # argparser.add_argument('--ground_truth', type=str, default='./ground_truth_test.json')
    
    # argparser.add_argument('--predict', type=str, default='./results/finetune_skeleton_t5_6/jsons/.')
    # argparser.add_argument('--output', type=str, default='./results/finetune_skeleton_t5_6/geval/.')

    # argparser.add_argument('--predict', type=str, default='./BaseLine/llama32_skating.json')
    # argparser.add_argument('--output', type=str, default='./BaseLine/')

    # argparser.add_argument('--predict', type=str, default='./results/finetune_skeleton_t5_test/jsons/.')
    # argparser.add_argument('--output', type=str, default='./results/finetune_skeleton_t5_test/geval/.')

    # Segment
    # argparser.add_argument('--ground_truth', type=str, default='./results/finetune_error_seg/jsons/segment_gt.json')
    # argparser.add_argument('--predict', type=str, default='./results/finetune_error_seg/jsons/results_epoch_90_segment.json')
    # argparser.add_argument('--output', type=str, default='./results/finetune_error_seg/geval/.')

    # Untrimmed
    # argparser.add_argument('--ground_truth', type=str, default='./results/finetune_untrimmed/jsons/untrimmed_gt.json')
    # argparser.add_argument('--predict', type=str, default='./results/finetune_untrimmed/jsons/results_epoch_90_untrimmed.json')
    # argparser.add_argument('--output', type=str, default='./results/finetune_untrimmed/geval/.')

    # Error Segment of Boxing
    # argparser.add_argument('--ground_truth', type=str, default='./boxing_error_gt.json')
    # argparser.add_argument('--predict', type=str, default='./results/finetune_boxing_error/jsons/results_epochsegment0.json')
    # argparser.add_argument('--output', type=str, default='./results/finetune_boxing_error/geval/.')

    # Boxing
    # argparser.add_argument('--ground_truth', type=str, default='./results/finetune_boxing/boxing_gt.json')
    # argparser.add_argument('--predict', type=str, default='./results/finetune_boxing/jsons/results_epoch175.json')
    # argparser.add_argument('--output', type=str, default='./results/finetune_boxing/geval/.')

    # BoxingRGB
    # argparser.add_argument('--ground_truth', type=str, default='./RGB_boxing_gt.json')
    # argparser.add_argument('--predict', type=str, default='./results/finetune_boxing/jsons/results_epoch14.json')
    # argparser.add_argument('--output', type=str, default='./results/finetune_boxing/geval/.')

    # argparser.add_argument('--ground_truth', type=str, default='./ground_truth_test.json')
    # argparser.add_argument('--predict', type=str, default='./results/finetune_skating_no_ref/jsons/results_epoch80.json')
    # argparser.add_argument('--output', type=str, default='./results/finetune_skating_no_ref/geval/.')

    # argparser.add_argument('--ground_truth', type=str, default='./No_ref_boxing_gt.json')
    # argparser.add_argument('--predict', type=str, default='./results/finetune_boxing_no_ref/jsons/results_epoch75.json')
    # argparser.add_argument('--output', type=str, default='./results/finetune_boxing_no_ref/geval/.')

    # argparser.add_argument('--ground_truth', type=str, default='./results/boxing_0304/No_ref_boxing_gt.json')
    # argparser.add_argument('--predict', type=str, default='./results/boxing_0304/jsons/results_epoch70.json')
    # argparser.add_argument('--output', type=str, default='./results/boxing_0304/geval/.')

    # Skating Ground Truth
    # argparser.add_argument('--ground_truth', type = str, default = './results/skating_gt/skating_gt.json')
    # argparser.add_argument('--predict', type = str, default = './results/skating_gt/jsons')
    # argparser.add_argument('--output', type = str, default = './results/skating_gt/geval/.')

    # Skating Evaluation
    # argparser.add_argument('--ground_truth', type = str, default = './ground_truth_test.json')
    # argparser.add_argument('--predict', type = str, default = './results/skating_evaluation/jsons')
    # argparser.add_argument('--output', type = str, default = './results/skating_evaluation/geval/.')

    # Boxing Evaluation
    # argparser.add_argument('--ground_truth', type = str, default = '../results/boxing_evaluation/boxing_gt.json')
    # argparser.add_argument('--predict', type = str, default = '../results/boxing_evaluation/jsons')
    # argparser.add_argument('--output', type = str, default = '../results/boxing_evaluation/geval/.')

    argparser.add_argument('--ground_truth', type = str, default = '../results/skating_gt/skating_gt.json')
    argparser.add_argument('--predict', type = str, default = '../results/skating_gt/jsons')
    argparser.add_argument('--output', type = str, default = '../results/skating_gt/geval/.')

    args = argparser.parse_args()
    api_key = os.getenv("ANTHROPIC_KEY")
    prompt = read_template(args.prompt_fp)

    if os.path.isdir(args.predict) :
        all_files = os.listdir(args.predict)

        filtered_files = [f for f in all_files 
                          if f.endswith('.json') and f.startswith('results_epoch')
                          and int(f.split('epoch')[1].split('.')[0]) > 30]

        if not os.path.exists(args.output) :
            os.makedirs(args.output)

        all_epoch = {}
        all_filename = 'allgeval.json'
        all_filepath = os.path.join(args.output, all_filename)
        for file_name in filtered_files :
            file_path = os.path.join(args.predict, file_name)
            epoch = int(file_name.split('epoch')[1].split('.')[0])
            filename_without_extension = file_name.split('.json')[0]
            print(filename_without_extension)
            results = read_data(args.ground_truth, file_path)
            score = g_eval(args, results, prompt, api_key, epoch)
            all_epoch[filename_without_extension] = {'score' : score, 'epoch' : epoch}

        sorted_all_epoch = dict(sorted(all_epoch.items(), key = lambda item : item[1]['score'], reverse = True))

        with open(all_filepath, 'w') as f :
            json.dump(sorted_all_epoch, f, indent = 4)

    else :
        Scores = {}
        if not os.path.exists(args.output) :
            os.makedirs(args.output)

        filename = os.path.basename(args.predict)
        filename_without_extension = os.path.splitext(filename)[0]

        all_filename = f'geval_all_{filename}.json'
        # all_filepath = f'RGB_geval_all_{filename}.json'
        all_filepath = os.path.join(args.output,all_filename)
        file_path = os.path.join(args.predict, filename)

        results = read_data(args.ground_truth, args.predict)
        score = g_eval(args, results, prompt, api_key, filename)
        Scores[filename] = {'score' : score}

        with open(all_filepath, 'w') as f :
            json.dump(Scores, f, indent = 4)