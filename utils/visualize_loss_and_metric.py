import re, os, json, matplotlib.pyplot as plt

task = "skating_gt_ClosestSimGT"
task_path = f"/home/weihsin/projects/MotionExpert_tmp/{task}"

task = "boxing_gt_ClosestSimGT"
task = "skating_gt_PerGT"
task = "boxing_gt_PerGT"
task = "pretrain_ref"
task = "pretrain"
task = "boxing_aligned_best"
task_path = f"./results/{task}"

log_path = os.path.join(task_path, "stdout.log")

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
plt.title(f"{task} : Training vs Evaluation Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()

image_path = os.path.join(task_path, f"loss.png")

plt.savefig(image_path)
plt.close()

# Metrics
def plot_metric_with_top3(metric_data, label, color):
    metric_data_sorted = sorted(metric_data, key=lambda x: x[0])
    epochs = [e for e, _ in metric_data_sorted]
    scores = [s for _, s in metric_data_sorted]

    top3 = sorted(metric_data_sorted, key=lambda x: x[1], reverse=True)[:3]

    for epoch, score in top3:
        plt.annotate(f"{score:.4f}",
                     xy = (epoch, score),
                     xytext = (epoch + 2, score + 6),
                     arrowprops = dict(facecolor = color, arrowstyle="->"),
                     fontsize = 8,
                     color = "black")

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

plot_metric_with_top3(list(bleu_1.items()), "BLEU-1", "lime")
plot_metric_with_top3(list(bleu_4.items()), "BLEU-4", "magenta")
plot_metric_with_top3(list(rouge.items()), "ROUGE", "orange")
plot_metric_with_top3(list(cider.items()), "CIDEr", "cyan")
plot_metric_with_top3(list(bertscore.items()), "BERTScore", "red")

all_epochs = sorted(set().union(bleu_1, bleu_4, rouge, cider, bertscore))
xtick_positions = [e for e in all_epochs if e % 5 == 0]

if all_epochs[-1] % 5 != 0 and all_epochs[-1] not in xtick_positions:
    xtick_positions.append(all_epochs[-1])
plt.xticks(xtick_positions)

plt.xlabel("Epoch")
plt.ylabel("Score")
plt.title(f"{task} : Evaluation Metrics")
plt.legend()
plt.grid(True)
plt.tight_layout()

image_path = os.path.join(task_path, f"metrics.png")

plt.savefig(image_path)

plt.close()

Geval_path = os.path.join(task_path, "geval/allgeval.json")

with open(Geval_path, "r", encoding="utf-8") as f:
    Geval = json.load(f)

# Geval has epoch (from most high to low)
geval_epochs = [v["epoch"] for k, v in Geval.items()]

# According to the order of geval to construct metrics JSON
metrics_ordered = {}

for epoch in geval_epochs:
    epoch_key = f"epoch{epoch}"
    metrics_ordered[epoch_key] = {
        "BLEU-1": bleu_1.get(epoch),
        "BLEU-4": bleu_4.get(epoch),
        "ROUGE": rouge.get(epoch),
        "CIDEr": cider.get(epoch),
        "BERTScore": bertscore.get(epoch),
    }

metrics_path = os.path.join(task_path, "metrics.json")
with open(metrics_path, "w", encoding="utf-8") as f:
    json.dump(metrics_ordered, f, indent=2, ensure_ascii=False)