import json, argparse, tqdm, time, os, re
import anthropic
def read_template(prompt_fp):
    text_template = open(prompt_fp).read()
    return text_template

def read_data(predict_path):   
    with open(predict_path, 'r', encoding='utf-8') as f:
        predictions = json.load(f)

    return predictions

def acc(args, filename, metric):
    output_filename = metric + '_Detection_{{filename}}.json'

    output_filename = output_filename.replace('{{filename}}',filename)
    output_filepath = os.path.join(args.output,output_filename)
    with open(output_filepath, 'r') as f:
        datas = json.load(f)

    acc_shot_count = 0
    acc_score = 0

    score_name = metric + '_Detection_score'
    for data in datas:
        if isinstance(data["score"], list):
            bestscore = max(data["score"])
        else :
             bestscore = data["score"]

        if data[score_name] != 0 and bestscore > 1:
            acc_shot_count += 1
            acc_score += bestscore
    return acc_score, acc_shot_count

def g_eval(args,summeval, prompt, api_key, filename, metric):
    new_json = []
    count, ignore, all_score = 0, 0, 0

    score_name = metric + '_Detection_score'
    prompt_name = metric + '_Detection_prompt'

    for instance in tqdm.tqdm(summeval):
        system_output   = instance['system_output']
        cur_prompt      = prompt.replace('{{Instruction}}', system_output)
        instance[prompt_name] = cur_prompt
        
        client = anthropic.Anthropic(api_key=api_key)
        try:
            _response   = client.messages.create(model="claude-3-5-sonnet-20240620",
                                                 messages=[{"role": "user", "content": cur_prompt}],
                                                 max_tokens=5,)
            content     = _response.content
            print(f"Claude response: {content}")  # 檢查 Claude 回應的完整內容

            text_value  = content[0].text
            match       = re.search(r'\d+', text_value)
            instance[score_name] = int(match.group())
            
            new_json.append(instance)

            count += 1
            all_score += instance[score_name]

        except Exception as e:
            print(f"Error occurred: {e}")
            ignore += 1
    
    output_filename = metric + '_Detection_{{filename}}.json'
    output_filename = output_filename.replace('{{filename}}',filename)

    output_filepath = os.path.join(args.output,output_filename)
    with open(output_filepath, 'w') as f:
        json.dump(new_json, f, indent=4)
    return all_score/count, all_score