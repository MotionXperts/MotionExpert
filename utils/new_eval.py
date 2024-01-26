
import pickle
import evals.nlgeval
from argparse import ArgumentParser, Namespace
import matplotlib.pyplot as plt
import os
from evals.nlgeval import compute_metrics_batch, compute_individual_metrics

class visualizer():
  def __init__(self):
    pass

  def plot_line_chart(self, data, plot_folder):
    if not os.path.exists(plot_folder):
      os.makedirs(plot_folder)
    ids = [ datum['id'] for datum in data ]
    metrics = [ 'Bleu_1', 'Bleu_2', 'Bleu_3', 'Bleu_4', 'METEOR', 'ROUGE_L', 'CIDEr' ]
    with open(os.path.join(plot_folder, 'summary.txt'), 'w') as file:
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
        # Write the file
        file.write(f'Average {metric} score = {average}\n')

def parse_args() -> Namespace:
  parser = ArgumentParser()
  parser.add_argument(
      "--filepath", type=str, default="/home/tom/sporttech/skating_coach/experimental/motion2text/result.pkl", help="Path to the pickle filename"
  )
  parser.add_argument(
      "--plots_folder", type=str, default="/home/tom/sporttech/skating_coach/experimental/motion2text/plots", help="Path to the folder for the plots outptu"
  )
  parser.add_argument(
      "--result_filename", "-f", type=str, help="Path to the filename of output summary (.txt)"
  )
  parser.add_argument(
      "--txt_file", "-t", action="store_true", help="Whether the input file is a text file"
  )
  parser.add_argument(
      "--verbose", "-v", action="store_true", help="Whether to evaluate the score one record at a time"
  )
  parser.add_argument(
      "--skipthoughts", action="store_true", help="Whether to activate skipThoughts"
  )
  parser.add_argument(
      "--glove", action="store_true", help="Whether to activate glove"
  )
  args = parser.parse_args()
  return args

def main(args):
  # Extracting the arguments
  filename = args.filepath
  plot_folder = args.plots_folder

  assert os.path.exists(filename), f"The file '{filename}' does not exists."

  refs = {}
  hyps = {}
  idx = 0
  if args.txt_file:
    special_tokens = [ '[CLS]', '[SEP]', '[PAD]', '<extra_id_0>' ]
    with open(filename, 'r') as file:
      content = file.read()
      delimiter = "----------"
      blocks = content.split(delimiter)
      for block in blocks:
        for token in special_tokens:
          block = block.replace(token, "")
        lines = [ line for line in block.split('\n') if line.strip() != "" ]
        if len(lines) > 2:
          reference = lines[1].strip()
          hypothesis = lines[2].strip()
          refs[idx] = [reference]
          hyps[idx] = [hypothesis]
          idx += 1
  else:
    with open(filename, 'rb') as file:
      loaded_data = pickle.load(file)
      for datum in loaded_data:
        refs[idx] = datum["Ground Truth:"]
        hyps[idx] = datum["Prediction:"]
        idx += 1
  
  assert len(refs) == len(hyps)

  # Evaluation 
  if not args.verbose:    # All at once
    res = compute_metrics_batch(refs, hyps, no_overlap=False, no_skipthoughts=(not args.skipthoughts), no_glove=(not args.glove))
    if args.result_filename:
      res_filename = args.result_filename
    else:
      res_filename = os.path.join(plot_folder, 'summary.txt')
    with open(res_filename, 'w') as file:
      for metric in res:
        file.write(f'{metric} = {round(res[metric], 6)}\n')
  else:  # One at a time
    evals = []
    for idx in range(0, len(refs)):
      references = refs[idx][0]
      hypothesis = hyps[idx][0]
      metrics_dict = compute_individual_metrics(references, hypothesis, no_overlap=False, no_skipthoughts=(not args.skipthoughts), no_glove=(not args.glove))
      metrics_dict["id"] = idx
      evals.append(metrics_dict)
      
      print(f"[Test case {idx}]")
      print(f" {references}")
      print(f" {hypothesis}")
      print(metrics_dict)
      print("-"*100)
      idx += 1

    vis = visualizer()
    vis.plot_line_chart(evals, plot_folder)
        

if __name__ == "__main__":
    args = parse_args()
    main(args)
    

