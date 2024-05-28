# Motion2text

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
#### create and activate a virtual env
```
$ conda create -n motion2text python=3.7
$ conda activate motion2text
$ pip install -r requirements.txt
```
In case of installation of language_evaluation, you need to install from github source code

## Prepare
You will have to first define a config file to run. Examples can be found under configs/


Create a folder called ```results``` and an name of the experiment (in the following example, the experiment is named "pretrain") under ```results```, then put the defined config file under it. 

The repo architecture may look like this.
```
    <root>
    |    -  alignment
    |    -  configs
    |    -  dataloaders
    |    -  results
               |    -     pretrain
                              |      -  config.yaml
    ...
    |    -  train_t5.py  
```
Make sure to change the ```LOGDIR``` path in config.yaml to the related path from repo root.

### Finetuning

Additionally, if you are finetuning from an existing checkpoint, you will have to further create a folder called pretrain_checkpoints, and put the desired checkpoint into that folder.

Example: 
```
    <root>
    |    -  alignment
    |    -  configs
    |    -  dataloaders
    |    -  results
               |    -     finetune-branch2
                              |      -  pretrain_checkpoints
                                                |        -     (put your pretrain checkpoint here) 
                              |      -  config.yaml
    ...
    |    -  train_t5.py  
```

## Build

```
$ torchrun --nproc_per_node <specify_how_many_gpus_to_run> train_t5_stagcn.py --cfg_file <path_to_cfg_file>
```

or, if the above yield Error ```detected multiple processes in same device```

run

```
$ python -m torch.distributed.launch --nproc_per_node <specify_how_many_gpus_to_run> train_t5_stagcn.py --cfg_file <path_to_cfg_file>
```

## All you need to know in SportTech
[開發紀錄](https://hackmd.io/@weihsinyeh/MotionXperts)

## Reference
* [HybrIK: Hybrid Analytical-Neural Inverse Kinematics for Body Mesh Recovery](https://github.com/Jeff-sjtu/HybrIK)
* [Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition (STGCN)](https://github.com/yysijie/st-gcn)
* [Spatio-Temporal adaptive graph convolutional network model (STAGCN)](https://github.com/QiweiMa-LL/STAGCN)
* [T5 model document](https://huggingface.co/docs/transformers/model_doc/t5)
* [BertViz - Visualize Attention in NLP Models](https://github.com/jessevig/bertviz)
* [Dynamic time warping](https://github.com/minghchen/CARL_code/blob/master/utils/dtw.py)
* [CARL](https://arxiv.org/abs/2203.14957)
* [Temporal Cycle-Consistency Learning](https://arxiv.org/abs/1904.07846)
