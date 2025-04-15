import os, json, re, random
import numpy as np
import matplotlib.pyplot as plt

folder_path = "../results/boxing_gt_ClosestSimGT/jsonsbest_gt_indices"

files = [f for f in os.listdir(folder_path) if f.startswith("best_gt_indices_")]
files.sort(key=lambda x: int(re.search(r'_(\d+)\.json$', x).group(1)))

all_samples, all_epochs = [], []

for file in files:
    with open(os.path.join(folder_path, file)) as f:
        data = json.load(f)
        epoch_values = list(data.values())
        if not all_samples:
            all_samples = list(data.keys())
        all_epochs.append(epoch_values)

# During training, only present the best ground truth indices for every 5 epochs.
selected_epoch_indices = [i for i in range(len(files)) if i % 5 == 0]
selected_epochs = [all_epochs[i] for i in selected_epoch_indices]
selected_files = [files[i] for i in selected_epoch_indices]

data_array = np.array(selected_epochs).T

# Only select 10 samples
selected_sample_indices = random.sample(range(len(all_samples)), 10)
selected_sample_names = [all_samples[i] for i in selected_sample_indices]

plt.figure(figsize = (12, 8))

for i in selected_sample_indices :
    plt.plot(selected_epoch_indices, data_array[i], label = f"Sample {all_samples[i]}", marker = 'o', alpha = 0.6)

plt.xlabel("Epoch")
plt.ylabel("Selected Ground Truth Index (0, 1, 2, 3)")
plt.title("Sample GT Selection Over Epochs (Every 5 Epochs, 20 Samples)")
plt.xticks(selected_epoch_indices)
plt.yticks([0, 1, 2, 3])
plt.grid(True)
plt.legend(loc = 'center left', bbox_to_anchor = (1, 0.5))
plt.tight_layout()
output_path = "./sample_gt_selection.png"
plt.savefig(output_path)