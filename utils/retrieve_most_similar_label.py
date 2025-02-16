import json
import pickle
import openai
import os

def find_all_choices(root_name,jsn_path):
    with open(jsn_path) as f:
        data = json.load(f)
    choices = []
    for item in data:
        if item['video_name'] == root_name: ## video_name should be the correct one, when trimming, we trim videos into jumps.
            choices.append(item)
    assert len(choices) > 0, f"No labels found for {root_name}"
    return choices

def call_gpt(prompt, api_key):
    openai.api_key = api_key
    response = openai.chat.completions.create(
        model="gpt-4o-2024-08-06",
        messages=[
            {"role": "user", "content": f'Original instruction: "{prompt}"'}
        ]
    )
    return response.choices[0].message.content

def compute_similar_score(cfg,predictions,key,eval_name,epoch):
    """
    Will only work on untrimmed eval_name.
    predictions: Dict<root_name>: prediction
    """
    if eval_name != 'untrimmed':
        print("\033[91m {} \033[00m".format("This function only works on untrimmed eval_name. The following is EXPECTED to fail if using other eval_name."))

    all_choices = []
    print("len(predictions): ",len(predictions))
    # print("predictions: ",predictions)
    for vid_name in predictions:
        choices = find_all_choices(vid_name,os.path.join(cfg.LOGDIR,f'{eval_name}.json')) ### All_choices: list of list of dictionaries
        all_choices.append(choices) ## choices should always be one. The true label candidates come from index_2_commnent field.

    max_choices = []
    runs = {}
    # if os.path.exists(os.path.join(cfg.LOGDIR,f'run_{eval_name}_{epoch}.json')):
    #     with open(os.path.join(cfg.LOGDIR,f'run_{eval_name}_{epoch}.json')) as f:
    #         runs = json.load(f)
    #     print(f"Loaded previous runs from {os.path.join(cfg.LOGDIR,f'run_{eval_name}_{epoch}.json')}")
    #     print(f"Previous runs: {runs}")
    #     for pred in runs:
    #         max_choices.append(runs[pred]['max_choice'])
# else:
    for prediction,choices in zip(predictions,all_choices):
        label = predictions[prediction]
        assert len(choices) ==1, f"Choices should be one, but got {len(choices)} for {prediction}"

        labels = [choices[0]['index_2_comment'][x]['comment'] for x in choices[0]['index_2_comment']]
        if len(labels) == 0:
            print(f"Skipping {prediction} as no labels found")
            max_choice=-1
        elif len(labels) == 1:
            max_choice = 0
        else:
            prompt = f"""
            Given the following choices:
            {labels},
            which one is the most similar to the label {label}?

            Reply with the following template:
            "The most similar label is: <index>"

            Only put number in the <index> field and do not include any other information except for the template.
            """

            max_choice = 0
            max_choice = call_gpt(prompt,key)

        max_choices.append(max_choice)


            ## save the log to result, just in case
        run = {
            "label": label,
            "choices": choices,
            "max_choice": max_choice
        }
        runs[prediction] = run
        with open(os.path.join(cfg.LOGDIR,f'run_{eval_name}_{epoch}.json'),'w') as f:
            json.dump(runs,f,indent=2)
    print(f"Max choices: {max_choices}")
    # print(f"All Choices: {all_choices}")
    assert len(max_choices) == len(all_choices), f"Expected {len(all_choices)} max choices, but got {len(max_choices)}"
    print("Successfully computed max choices")
    annotation = []
    abandoned = []
    for choice,(index,i) in zip(all_choices,enumerate(max_choices)):
        try:
            if(int(i)==-1):
                print(f"Skipping {choice[0]['video_name']} (index:{index}) as no labels found")
                abandoned.append(choice[0]['video_name'])
                continue
            annotation.append({
                'video_name': choice[0]['video_name'],
                'revised_label': choice[0]['index_2_comment'][str(int(i)+1)]['comment']
            })
        except:
            split = i.split('The most similar label is: ')[-1].replace("\"","").replace(".","")
            print(choice[0]['video_name'],split)
            annotation.append({
                'video_name': choice[0]['video_name'],
                'revised_label': choice[0]['index_2_comment'][str(int(split)+1)]['comment']
            })
        result_json = cfg.JSONDIR+'/GPT_score.json'
        with open(result_json, 'w') as f:
            json.dump(annotation, f,indent = 4)
            print(f"GPT_score saved in {result_json}")

        abandon_json = cfg.JSONDIR+'/abandon_GPT.json'
        with open(abandon_json, 'w') as f:
            json.dump(annotation, f,indent = 4)
            print(f"GPT_score saved in {abandon_json}")

    return annotation , abandoned