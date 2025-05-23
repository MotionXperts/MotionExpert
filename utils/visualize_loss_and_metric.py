import re
import matplotlib.pyplot as plt

target_path = "skating_gt_ClosestSimGT"
log_path = f"/home/weihsin/projects/MotionExpert_tmp/{target_path}/stdout.log"

target_path = "boxing_gt_ClosestSimGT"
target_path = "skating_gt_PerGT"
target_path = "boxing_gt_PerGT"
log_path = f"/home/weihsin/projects/MotionExpert_tmp/MotionExpert/results/{target_path}/stdout.log"

with open(log_path, "r", encoding = "utf-8") as f:
    lines = f.readlines()

train_loss, eval_loss = {}, {}
bleu_1, bleu_4, rouge, cider, bertscore = {}, {}, {}, {}, {}

train_loss_re = re.compile(r"Epoch (\d+) : Loss ([0-9.]+)")
eval_loss_re = re.compile(r"Epoch (\d+) : Loss ([0-9.]+)")
metric_re = re.compile(r"Epoch (\d+) : (bleu_1|bleu_4|rouge|cider|bertscore) : ([0-9.]+)")

for line in lines :
    if "main.py" in line and "Loss" in line :
        match = train_loss_re.search(line)
        if match :
            epoch, loss = int(match.group(1)), float(match.group(2))
            train_loss[epoch] = loss
    elif "evaluation.py" in line and "Loss" in line :
        match = eval_loss_re.search(line)
        if match :
            epoch, loss = int(match.group(1)), float(match.group(2))
            eval_loss[epoch] = loss
    elif "evaluation.py" in line and any(m in line for m in ["bleu_1", "bleu_4", "rouge", "cider", "bertscore"]) :
        match = metric_re.search(line)
        if match :
            epoch, metric_name, value = int(match.group(1)), match.group(2), float(match.group(3))
            locals()[metric_name][epoch] = value * 100

sorted_train_epochs = sorted(train_loss.keys())
sorted_eval_epochs = sorted(eval_loss.keys())

xtick_positions = [e for e in sorted_train_epochs if e % 5 == 0]

if sorted_train_epochs[-1] % 5 != 0 and sorted_train_epochs[-1] not in xtick_positions :
    xtick_positions.append(sorted_train_epochs[-1])
plt.xticks(xtick_positions)

# Loss
plt.figure(figsize = (10, 6))
plt.plot(sorted_train_epochs, [train_loss[e] for e in sorted_train_epochs], label = "Training Loss", color = 'blue')
if eval_loss :
    plt.plot(sorted_eval_epochs, [eval_loss[e] for e in sorted_eval_epochs], label = "Evaluation Loss", color = 'orange')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title(f"{target_path} : Training vs Evaluation Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{target_path}_loss.png")
plt.close()

# Metrics
plt.figure(figsize = (10, 6))

def plot_metric(metric_dict, label, color) :
    if metric_dict :
        x = sorted(metric_dict.keys())
        y = [metric_dict[e] for e in x]
        plt.plot(x, y, label = label, marker = 'o', color = color)

plot_metric(bleu_1, "BLEU-1", "lime")
plot_metric(bleu_4, "BLEU-4", "magenta")
plot_metric(rouge, "ROUGE", "orange")
plot_metric(cider, "CIDEr", "cyan")
plot_metric(bertscore, "BERTScore", "red")


all_epochs = sorted(set().union(bleu_1, bleu_4, rouge, cider, bertscore))
xtick_positions = [e for e in all_epochs if e % 5 == 0]

if all_epochs[-1] % 5 != 0 and all_epochs[-1] not in xtick_positions:
    xtick_positions.append(all_epochs[-1])
plt.xticks(xtick_positions)

plt.xlabel("Epoch")
plt.ylabel("Score")
plt.title(f"{target_path} : Evaluation Metrics")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{target_path}_metrics.png")
plt.close()