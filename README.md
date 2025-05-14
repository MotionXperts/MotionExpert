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
$ conda create -n coachme python=3.10
$ conda activate coachme
$ pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121
# transformers
$ pip install transformers
# easydict
$ pip install easydict
# dotenv
$ pip install python-dotenv
# language_evaluation
$ pip install git+https://github.com/bckim92/language-evaluation.git
# loralib
$ pip install loralib
# bert_score
$ pip install bert-score
# openai
$ pip install openai
# nlgmetricverse
$ pip install nlg-metricverse
$ pip install timm==0.9.10
```
In case of installation of language_evaluation, you need to install from github source code

## Prepare
### Dataset
The dataset is saved as a pickle file and is of type `<class 'list'>`.
Each entry in the dataset contains the following information :
- `video_name` : `<class 'str'>` e.g. `7_front_4_cam1`
- `motion_type` : `<class 'str'>` e.g. `front`
- `features` : `<class 'torch.Tensor'>` shape : `torch.Size([number of frames, 66])`
```
Each 66-dimensional feature vector in the dataset represents the 3D coordinates of 22 skeletal
joints captured within a single frame. The vector is constructed by sequentially concatenating
the x, y, and z coordinates of each joint. Specifically, the first three elements correspond
to the x, y, and z coordinates of joint 0, the next three elements to joint 1, and this
pattern continues up to joint 21.
```
- `labels` : `<class 'list'>`
e.g.
```
['The back foot is not lifted. The rear hand is not protecting the chin. Not using the strength of the body, only the hands. Not clenching the fist when punching.',
 'Feet should be shoulder-width apart.',
 "The lower body is not participating at all in the punching, but the puncher's power generation is pretty good. More body rotation should be added for power generation. The head should not move forward with it.",
 'The back hand has no defense. The punches are not solid.']
```
**Setting : aligned**
- `start_frame` : `<class 'int'>` e.g. `0`
- `end_frame` `<class 'int'>` e.g. `51`
- `std_start_frame` `<class 'int'>` e.g. `39`
- `std_end_frame` `<class 'int'>` e.g. `90`
- `original_seq_len` `<class 'int'>` e.g. `51`
- `aligned_seq_len` `<class 'int'>` e.g. `51`

**Setting : error**
- `error_start_frame` : `<class 'int'>`
- `error_end_frame` `<class 'int'>`
- `error_std_start_frame` `<class 'int'>`
- `error_std_end_frame` `<class 'int'>`
- `error_seq_len` `<class 'int'>`

**Setting : GT**
- `gt_start_frame` : `<class 'int'>`
- `gt_end_frame` `<class 'int'>`
- `gt_std_start_frame` `<class 'int'>`
- `gt_std_end_frame` `<class 'int'>`
- `gt_seq_len` `<class 'int'>`
### Config File

|Task | Ref | Segment | config file |
| - | - | - | - |
Pretrain | X | NO_SEGMENT | `./results/pretrain/pretrain.yaml` |
Pretrain | V | NO_SEGMENT | `./results/pretrain_ref/pretrain_ref.yaml` |
Pretrain | Pad0 | NO_SEGMENT | `./results/pretrain_pad/pretrain_pad.yaml` |

|Task | Ref | Segment | config file |
| - | - | - | - |
Skating | X | NO_SEGMENT | `./results/skating/skating.yaml` |
Skating | V | GT | `./results/skating_gt/skating_gt.yaml` |
Skating | V | error | `./results/skating_error/skating_error.yaml` |
Skating | V | aligned | `./results/skating_aligned/skating_aligned.yaml` |
Boxing | X | NO_SEGMENT | `./results/boxing/boxing.yaml` |
Boxing | V | error | `./results/boxing_error/boxing_error.yaml` |
Boxing | V | aligned | `./results/boxing_aligned/boxing_aligned.yaml` |


### Pretrain
Take `pretrain_ref` setting as an example:
- Step 1 : Create the `pretrain_ref` directory under the `./results` directory.
- Step 2 : Place the `pretrain_ref.yaml`  file inside the `./results/pretrain_ref` directory.
- Step 3 : Run the following command:
```bash
$ python -m torch.distributed.run --nproc_per_node=1 --master_port=29050 main.py --cfg_file ./results/pretrain_ref/pretrain_ref.yaml > output/pretrain_ref
```
- Step 4 : After pretraining, the `./results/pretrain_ref` directory will be created automatically  as follows:
```bash
Motion Expert
    | - results
        | - pretrain_ref
            | -  pretrain_checkpoints
                | - ...
            | -  pretrain_ref.yaml
```
- For users: If training is interrupted, it will resume from the last saved epoch the next time you run the command.
- For developers: If you want to restart the entire training process from scratch, you must delete the entire `pretrain_checkpoints` directory. Otherwise, training will resume from the last saved epoch.

### Finetuning
Step 1 : create the `finetune` directory.

Step 2 : create the `pretrain_checkpoints` directory.

Step 3 : Put the pretrained checkpoint file (for example : checkpoint_epoch_00008.pth) in `pretrain_checkpoints` directory.

Run the following command from the MotionExpert directory.
```bash
$ wget -O ./results/skating_gt/pretrain_checkpoints/best_pretrained_weight.pth 'https://www.dropbox.com/scl/fi/gnkhtz0h6mnhftxpo4cgb/checkpoint_epoch_00009.pth?rlkey=jheub4udl83ppobv53ibufnkq&st=n3amwn1r&dl=1'
```

Step 4 : Place the `config.yaml` file in the `finetune` directory. For example, the config file `./results/skating_gt/skating_gt.yaml` should be placed in the `./results/skating_gt/` directory.

Step 5 : Run the following command from the MotionExpert directory.
```bash
$ python -m torch.distributed.run --nproc_per_node=1 --master_port=29051 main.py --cfg_file ./results/skating_gt/skating_gt.yaml > output/skating_gt
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
$ python -m torch.distributed.run --nproc_per_node <specify_how_many_gpus_to_run> main.py --cfg_file <path_to_cfg_file>
```
or, if the above yield Error ```detected multiple processes in same device```

```shell
$ python -m torch.distributed.run --nproc_per_node <specify_how_many_gpus_to_run> main.py --cfg_file <path_to_cfg_file>
```
#### Run pretrain setting
```shell
$ python -m torch.distributed.run --nproc_per_node 1 main.py --cfg_file {The PATH of MotionExpert}/MotionExpert/results/pretrain/config.yaml
```
#### Run finetune setting
```shell
$ python -m torch.distributed.run --nproc_per_node 1 main.py --cfg_file {The PATH of MotionExpert}/MotionExpert/results/finetune/config.yaml 
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
