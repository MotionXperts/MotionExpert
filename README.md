# CoachMe: Decoding Sport Elements with a Reference-Based Coaching Instruction Generation Model (ACL 2025)

<a href='https://aclanthology.org/2025.acl-long.1413/'><img src='https://img.shields.io/badge/ACL_2025_paper-CoachMe-red'></a> &nbsp;
<a href='https://motionxperts.github.io/'><img src='https://img.shields.io/badge/Project_Page-CoachMe-green'></a> &nbsp;
<a href='https://youtu.be/m7LDiiOyHjQ?list=PL_9a5ic6GUikB6J5lTi7Dg7LgRpXS9E0H'><img src='https://img.shields.io/badge/Video-CoachMe-yellow'></a> &nbsp;
<a href='https://arxiv.org/abs/2509.11698'><img src='https://img.shields.io/badge/2509.11698-arXiv-red'></a> &nbsp;

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
# anthropic for Claude
$ pip install anthropic
```
In case of installation of language_evaluation, you need to install from github source code

## Prepare
### Data Augmentation
The implementation of data augmentation can be found in the `./gt_preprocess` directory.

### Dataset
**The released dataset is located in the `./dataset` directory.**

The dataset is saved as a pickle file and is of type `<class 'list'>`.
The type of each entry is `<class 'dict'>`.
Each entry in the dataset contains the following information :
- `video_name` : `<class 'str'>` e.g. `test_0`
- `motion_type` : `<class 'str'>` e.g. `Jab`
```
In the FS dataset, there are four figure skating jumps: "Single_Axel", "Double_Axel", "Loop", and "Lutz".
In the BX dataset, there are two boxing techniques: "Jab" and "Cross".
```
- `coordinates` : `<class 'torch.Tensor'>` shape : `torch.Size([number of frames, 66])`
```
Each 66-dimensional feature vector in the dataset represents the 3D coordinates of 22 skeletal
joints captured within a single frame. The vector is constructed by sequentially concatenating
the x, y, and z coordinates of each joint. Specifically, the first three elements correspond
to the x, y, and z coordinates of joint 0, the next three elements to joint 1, and this
pattern continues up to joint 21.
```
- `camera_view` : `<class 'str'>`
```
 The 'camera_view' field is present only in the BX dataset.
```
- `labels` : `<class 'list'>`
e.g.
```
['The back foot is not lifted. The rear hand is not protecting the chin. Not using the strength of the body, only the hands. Not clenching the fist when punching.',
 'Feet should be shoulder-width apart.',
 "The lower body is not participating at all in the punching, but the puncher's power generation is pretty good. More body rotation should be added for power generation. The head should not move forward with it.",
 'The back hand has no defense. The punches are not solid.']
```
- `augmented_labels` : `<class 'list'>`
- `original_seq_len` `<class 'int'>` e.g. `51`

**Setting : `GT`**
- `gt_start_frame` : `<class 'int'>`
- `gt_end_frame` `<class 'int'>`
- `gt_std_start_frame` `<class 'int'>`
- `gt_std_end_frame` `<class 'int'>`
- `gt_seq_len` `<class 'int'>`

**Setting : `ALIGNED`**
- `aligned_start_frame` : `<class 'int'>` e.g. `0`
- `aligned_end_frame` `<class 'int'>` e.g. `51`
- `aligned_std_start_frame` `<class 'int'>` e.g. `39`
- `aligned_std_end_frame` `<class 'int'>` e.g. `90`
- `aligned_seq_len` `<class 'int'>` e.g. `51`

**Setting : `ERROR`**
- `error_start_frame` : `<class 'int'>`
- `error_end_frame` `<class 'int'>`
- `error_std_start_frame` `<class 'int'>`
- `error_std_end_frame` `<class 'int'>`
- `error_seq_len` `<class 'int'>`

### Standard Dataset
The dataset is saved as a pickle file and is of type `<class 'list'>`.
The type of each entry is `<class 'dict'>`.
Each entry in the dataset contains the following information :
- `video_name` : `<class 'str'>` e.g. `Jab`
```
The naming of `video_name` is currently the same as `motion_type`.
In the FS dataset, there are four figure skating jumps: "Single_Axel", "Double_Axel", "Loop", and "Lutz".
In the BX dataset, there are two boxing techniques: "Jab" and "Cross".
However, in the future, users will be able to choose their preferred standard video, in which case `video_name` may differ from `motion_type`.
```
- `motion_type` : `<class 'str'>` e.g. `Jab`
```
In the FS dataset, there are four figure skating jumps: "Single_Axel", "Double_Axel", "Loop", and "Lutz".
In the BX dataset, there are two boxing techniques: "Jab" and "Cross".
```
- `coordinates` : `<class 'torch.Tensor'>` shape : `torch.Size([number of frames, 66])`
```
Each 66-dimensional feature vector in the dataset represents the 3D coordinates of 22 skeletal
joints captured within a single frame. The vector is constructed by sequentially concatenating
the x, y, and z coordinates of each joint. Specifically, the first three elements correspond
to the x, y, and z coordinates of joint 0, the next three elements to joint 1, and this
pattern continues up to joint 21.
```

### Pretrained Dataset
Both `humanml3D_train.pkl` and `humanml3D_test.pkl` correspond to the train and test datasets in HumanML3D, respectively.

The dataset is saved as a pickle file and is of type `<class 'list'>`.
The type of each entry is `<class 'dict'>`.
Each entry in the dataset contains the following information :
- `video_name` : `<class 'str'>` e.g. `000001`
- `coordinates` : `<class 'torch.Tensor'>` shape : `torch.Size([number of frames, 66])`
- `labels` : `<class 'list'>` e.g.
```
['a man squats extraordinarily low then bolts up in an unsatisfactory jump.',
 'a person falls to the ground in a sitting motion and then pops back up in a standing position.',
 'a person squats down then jumps',
 'a descends into a falling motion and thens bounces back up.']
```

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

## Best Epoch

|Task | Ref | Segment | best epoch |
| - | - | - | - |
Pretrain | X | NO_SEGMENT | `checkpoint_epoch_00009.pth` |
Pretrain | V | NO_SEGMENT | `checkpoint_epoch_00012.pth` |
Pretrain | Pad0 | NO_SEGMENT | |

|Task | Ref | Segment | best epoch |
| - | - | - | - |
Skating | X | NO_SEGMENT | |
Skating | V | GT | |
Skating | V | error | |
Skating | V | aligned | |
Boxing | X | NO_SEGMENT | |
Boxing | V | error | |
Boxing | V | aligned | |

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

## Evaluation
The implementation for visualizing the attention mechanism in Human Pose Perception can be found in the repository: https://github.com/MotionXperts/Evaluation.

The G-eval template for consistency score and sport indicators can be found in the `./GEval` directory.

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

## Citation
If you find our work useful, please consider cite this work as
```
@inproceedings{yeh-etal-2025-coachme,
    title = "{C}oach{M}e: Decoding Sport Elements with a Reference-Based Coaching Instruction Generation Model",
    author = "Yeh, Wei-Hsin  and
      Su, Yu-An  and
      Chen, Chih-Ning  and
      Lin, Yi-Hsueh  and
      Ku, Calvin  and
      Chiu, Wenhsin  and
      Hu, Min-Chun  and
      Ku, Lun-Wei",
    editor = "Che, Wanxiang  and
      Nabende, Joyce  and
      Shutova, Ekaterina  and
      Pilehvar, Mohammad Taher",
    booktitle = "Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2025",
    address = "Vienna, Austria",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.acl-long.1413/",
    pages = "29126--29151",
    ISBN = "979-8-89176-251-0",
    abstract = "Motion instruction is a crucial task that helps athletes refine their technique by analyzing movements and providing corrective guidance. Although recent advances in multimodal models have improved motion understanding,generating precise and sport-specific instruction remains challenging due to the highly domain-specific nature of sports and the need for informative guidance. We propose CoachMe, a reference-based model that analyzes the differences between a learner{'}s motion and a reference under temporal and physical aspects. This approach enables both domain-knowledge learning and the acquisition of a coach-like thinking process that identifies movement errors effectively and provides feedback to explain how to improve. In this paper, weillustrate how CoachMe adapts well to specific sports such as skating and boxing by learning from general movements and then leveraging limited data. Experiments show that CoachMe provides high-quality instructions instead of directions merely in the tone of a coach but without critical information. CoachMe outperforms GPT-4o by 31.6{\%} in G-Eval on figure skating and by 58.3{\%} on boxing. Analysisfurther confirms that it elaborates on errors and their corresponding improvement methods in the generated instructions. You can find CoachMe here: \url{https://motionxperts.github.io/}"
}
```