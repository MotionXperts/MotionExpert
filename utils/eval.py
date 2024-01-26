
import pickle
import evals.nlgeval
from argparse import ArgumentParser, Namespace
import matplotlib.pyplot as plt
import os
from evals.nlgeval import compute_individual_metrics

class visualizer():
  def __init__(self):
    pass

  def plot_line_chart(self, data, plot_folder):
    if not os.path.exists(plot_folder):
      os.makedirs(plot_folder)
    ids = [ datum['id'] for datum in data ]
    metrics = [ 'Bleu_1', 'Bleu_2', 'Bleu_3', 'Bleu_4', 'METEOR', 'ROUGE_L', 'CIDEr' ]
    for metric in metrics:
      vals = [ datum[metric] for datum in data ]
      average = sum(vals) / len(vals) if len(vals) > 0 else 0
      print(f'Average {metric} score = {average}')
      plt.clf()
      plt.plot(ids, vals)
      plt.title(metric)
      plt.ylabel(f'{metric} score')
      plt.xlabel("id")
      plt.savefig(os.path.join(plot_folder, f'{metric}.png'))

def parse_args() -> Namespace:
  parser = ArgumentParser()
  parser.add_argument(
      "--filepath", type=str, default="/home/tom/sporttech/skating_coach/experimental/motion2text/result.pkl", help="Path to the filename"
  )
  parser.add_argument(
      "--plots_folder", type=str, default="/home/peihsin/projects/MotionExpert/utils/plots", help="Path to the folder for the plots outptu"
  )
  parser.add_argument(
      "--txt_file", "-t", action="store_true", help="Whether the input file is a text file"
  )
  args = parser.parse_args()
  return args

def main(args):
  # Extracting the arguments
  filename = args.filepath
  plot_folder = args.plots_folder

  assert os.path.exists(filename), f"The file '{filename}' does not exists."

  if args.txt_file:
    special_tokens = [ '[CLS]', '[SEP]', '[PAD]','<extra_id_0>' ]
    loaded_data = []
    with open(filename, 'r') as file:
      content = file.read()
      delimiter = "----------"
      blocks = content.split(delimiter)
      for block in blocks:
        for token in special_tokens:
          block = block.replace(token, "")
        lines = [ line for line in block.split('\n') if line.strip() != "" ]
        if len(lines) > 2:
          reference = lines[1].replace('Ground Truth:', '').strip()
          hypothesis = lines[2].replace('Prediction:', '').strip()
          loaded_data.append({"Ground Truth": reference, "Prediction": hypothesis})
  else:
    with open(filename, 'rb') as file:
      loaded_data = pickle.load(file)
  
  """
  loaded_data = { references: [], hypothesis: "" }
  """

  evals = []
  idx = 0
  for datum in loaded_data:
    references = datum["Ground Truth"]
    hypothesis = datum["Prediction"]
    metrics_dict = compute_individual_metrics(references, hypothesis, no_overlap=False, no_skipthoughts=True, no_glove=True)
    metrics_dict["id"] = idx
    evals.append(metrics_dict)
    
    print(f"[Test case {idx}] Ground Truth: {references}, Prediction: {hypothesis}")
    print(metrics_dict)
    print("-"*100)
    idx += 1

  vis = visualizer()
  vis.plot_line_chart(evals, plot_folder)
        

if __name__ == "__main__":
    args = parse_args()
    main(args)
    

