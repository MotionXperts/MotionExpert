import json
import copy
from tqdm import tqdm
import os
from translate import Translator

filename = '/home/weihsin/datasets/BoxingClip/completedTasks.json'

def reformat(sentence):
  strs = [ s.strip() for s in sentence.split('\n') ]
  return '. '.join(strs)

def translate(sentence):
  try:
    translator = Translator(from_lang="zh", to_lang="en")
    translated_text = translator.translate(sentence)
    return translated_text
  except Exception as e:
    print(f"{e} Input=({sentence})")
    return ""

sentence_set = set()
with open(filename, 'r') as file:
  data = json.load(file)
  obj_list = []
  task_list = data['completed_tasks']
  for task in tqdm(task_list, desc="Processing"):
    task_copy = copy.deepcopy(task)
    translated_type = translate(reformat(task['task']['type']))
    translated_desc = translate(reformat(task['description']))
    task_copy['task']['type'] = translated_type
    task_copy['description'] = translated_desc
    obj_list.append(task_copy)

json_string = json.dumps(obj_list, indent=4)
base, ext = os.path.splitext(filename)
output_filename = base + '_en' + ext
with open(output_filename, 'w') as file:
  file.write(json_string)
