# MotionXpert
```                            
    __ ( }       __  __       _   _            __   __                _   
  '---. _`---,  |  \/  |     | | (_)           \ \ / /               | |  
  ___/ /        | \  / | ___ | |_ _  ___  _ __  \ V / _ __   ___ _ __| |_  
/,---'\\        | |\/| |/ _ \| __| |/ _ \| '_ \  > < | '_ \ / _ \ '__| __|
      //        | |  | | (_) | |_| | (_) | | | |/ . \| |_) |  __/ |  | |_ 
     '==        |_|  |_|\___/ \__|_|\___/|_| |_/_/ \_\ .__/ \___|_|   \__|
                                                      | |                  
                                                      |_|                   
```
---

## Install 
create and activate a virtual env
```shell
$ conda create -n motion2text python=3.7
$ conda activate motion2text
$ pip install -r requirements.txt
```
In case of installation of language_evaluation, you need to install from github source code

## Prepare

### Dataset
The dataset is saved as a pickle file and is of type `<class 'list'>`.
Each entry in the dataset contains the following information :
- `features` : `<class 'torch.Tensor'>` shape : `torch.Size([number of frames, 66])`
```
Each 66-dimensional feature vector in the dataset represents the 3D coordinates of 22 skeletal joints captured within a single frame. The vector is constructed by sequentially concatenating the x, y, and z coordinates of each joint. Specifically, the first three elements correspond to the x, y, and z coordinates of joint 0, the next three elements to joint 1, and this pattern continues up to joint 21.
```
- `labels` : `<class 'list'>`
e.g.
```
['The back foot is not lifted. The rear hand is not protecting the chin. Not using the strength of the body, only the hands. Not clenching the fist when punching.', 'Feet should be shoulder-width apart.', "The lower body is not participating at all in the punching, but the puncher's power generation is pretty good. More body rotation should be added for power generation. The head should not move forward with it.", 'The back hand has no defense. The punches are not solid.']
```
- `video_name` : `<class 'str'>` e.g. `7_front_4_cam1`
- `start_frame` : `<class 'int'>` e.g. `0`
- `end_frame` `<class 'int'>` e.g. `51`
- `std_start_frame` `<class 'int'>` e.g. `39`
- `std_end_frame` `<class 'int'>` e.g. `90`
- `original_seq_len` `<class 'int'>` e.g. `51`
- `aligned_seq_len` `<class 'int'>` e.g. `51`

### Config File
The template config file for pretrain :

`TASK.PRETRAIN_SETTING` can choose to use `STAGCN` or `Attention`.
The first one is the implementation of [Spatial Temporal Attention Graph Convolutional Networks with Mechanics-Stream for Skeleton-based](https://openaccess.thecvf.com/content/ACCV2020/papers/Shiraki_Spatial_Temporal_Attention_Graph_Convolutional_Networks_with_Mechanics-Stream_for_Skeleton-based_ACCV_2020_paper.pdf) and the second one is Ours implementation.

`TASK.PRETRAIN_DIFFERENCE` can choose to be `true` or `false`. 
If the `TASK.PRETRAIN_DIFFERENCE` is `true`, the model will use the difference information.

```shell
HIDDEN_CHANNEL: 32
OUT_CHANNEL: 128
TRANSFORMATION:
  REDUCTION_POLICY: 'TIME_POOL'
TASK:
  PRETRAIN: true
  PRETRAIN_SETTING: 'Attention'
  PRETRAIN_DIFFERENCE : true
DATA: 
  TRAIN: '{The PATH of the pretrain training dataset}'
  TEST: '{The PATH of the pretrain testing dataset}'
  BATCH_SIZE: 16
OPTIMIZER:
  LR: 1e-4
  MAX_EPOCH: 50
  WARMUP_STEPS: 5000
BRANCH: 1
LOGDIR: ./results/pretrain
args:
  eval_multi: false
```
The template config file for finetune :
```shell
HIDDEN_CHANNEL: 32
OUT_CHANNEL: 128
TRANSFORMATION:
  REDUCTION_POLICY: 'TIME_POOL'
TASK:
  PRETRAIN: false
  PRETRAIN_SETTING: 'Attention'
  PRETRAIN_DIFFERENCE : true
WEIGHT_PATH: '{The PATH of MotionExpert}/MotionExpert/results/pretrain/pretrain_checkpoints/checkpoint_epoch_00008.pth'
DATA: 
  TRAIN: '{The PATH of the finetune training dataset}'
  TEST: '{The PATH of the finetune testing dataset}'
  BATCH_SIZE: 16
OPTIMIZER:
  LR: 1e-4
  MAX_EPOCH: 50
  WARMUP_STEPS: 5000
BRANCH: 1
LOGDIR: ./results/finetune
args:
  eval_multi: false
```
Create the directory `results` in the directory `{The PATH of MotionExpert}/MotionExpert`.

### Pretrain
Step 1 : create the `pretrain` directory.

Step 2 : Put the `config.yaml` (for example : The template config file for pretrain) `pretrain` directory.

Step 3 : After pretrain, the `pretrain_checkpoints` directory will be created automatically like the following :

```
Motion Expert
    | - results
        | - pretrain
            | -  pretrain_checkpoints
                | - ...
            | -  config.yaml 
```
For the users : 

After suspending the training, it will continue training from the last epoch next time.

For the developers : 

If you want to **restart** the whole training process, you need to delete whole `pretrain_checkpoints` directory, otherwise it training from the last epoch next time.

### Finetuning
Step 1 : create the `finetune` directory.

Step 2 : create the `pretrain_checkpoints` directory.

Step 3 : Put the pretrained checkpoint file (for example : checkpoint_epoch_00008.pth) in `pretrain_checkpoints` directory.

Run the following command from the MotionExpert directory.
```bash
wget -O ./results/skating_gt/pretrain_checkpoints/best_pretrained_weight.pth 'https://www.dropbox.com/scl/fi/gnkhtz0h6mnhftxpo4cgb/checkpoint_epoch_00009.pth?rlkey=jheub4udl83ppobv53ibufnkq&st=n3amwn1r&dl=1'
```

Step 4 : Place the `config.yaml` file in the `finetune` directory. For example, the config file `./results/skating_gt/skating_gt.yaml` should be placed in the `./results/skating_gt/` directory.

Step 5 : Run the following command from the MotionExpert directory.
```bash
torchrun --nproc_per_node=1 --master_port=29051 main.py --cfg_file ./results/skating_gt/skating_gt.yaml > output/skating_gt
```

Step 6 : After finetuning, the `checkpoints` directory will be created automatically like the following :

```
Motion Expert
    | - results
        | - finetune
            | - checkpoints
                | ...
            | - pretrain_checkpoints
                | - best_pretrained_weight.pth
            | - config.yaml 
```

Additionally, if you are finetuning from an existing checkpoint, you will have to further create a folder called pretrain_checkpoints, and put the desired checkpoint into that folder.

For the developers: 

If you want to **restart** the whole training process, you need to delete whole `checkpoints` directory, otherwise it training from the last epoch next time.


## Build
#### template command
```shell
$ torchrun --nproc_per_node <specify_how_many_gpus_to_run> main.py --cfg_file <path_to_cfg_file>
```
or, if the above yield Error ```detected multiple processes in same device```

```shell
$ python -m torch.distributed.launch --nproc_per_node <specify_how_many_gpus_to_run> main.py --cfg_file <path_to_cfg_file>
```
#### Run pretrain setting
```shell
$ python -m torch.distributed.launch --nproc_per_node 1 main.py --cfg_file {The PATH of MotionExpert}/MotionExpert/results/pretrain/config.yaml
```
#### Run finetune setting
```shell
$ python -m torch.distributed.launch --nproc_per_node 1 main.py --cfg_file {The PATH of MotionExpert}/MotionExpert/results/finetune/config.yaml 
```

### Submodule - VideoAlignment

We use [VideoAlignment](https://github.com/MotionXperts/VideoAlignment) as our submodule to handle branch2 alignment code.

To clone the submodule, after you ```git clone``` this repo, run the followings:

```shell
$ cd VideoAlignment
$ git submodule init
$ git submodule update
```

If you need to update the VideoAlignment Submodule branch wheh command `git pull`
```shell
$ git submodule update
$ git pull
```

## Development Rcecord
CoachMe Development Rcecord : https://hackmd.io/@weihsinyeh/MotionXperts

Concept Difference : https://hackmd.io/@uan/HJvPoxNw6

## Reference
* [HybrIK: Hybrid Analytical-Neural Inverse Kinematics for Body Mesh Recovery](https://github.com/Jeff-sjtu/HybrIK)
* [Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition (STGCN)](https://github.com/yysijie/st-gcn)
* [Spatial Temporal Attention Graph Convolutional Networks with Mechanics-Stream for Skeleton-based Action Recognition (STA-GCN)](https://github.com/machine-perception-robotics-group/SpatialTemporalAttentionGCN)
* [T5 model document](https://huggingface.co/docs/transformers/model_doc/t5)
* [BertViz - Visualize Attention in NLP Models](https://github.com/jessevig/bertviz)
* [Dynamic time warping](https://github.com/minghchen/CARL_code/blob/master/utils/dtw.py)
* [CARL](https://arxiv.org/abs/2203.14957)
* [Temporal Cycle-Consistency Learning](https://arxiv.org/abs/1904.07846)
