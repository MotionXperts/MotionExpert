import argparse
import os,sys
from easydict import EasyDict
import yaml
import torch
def to_dict(config):
    if isinstance(config, list):
        return [to_dict(c) for c in config]
    elif isinstance(config, EasyDict):
        return dict([(k, to_dict(v)) for k, v in config.items()])
    else:
        return config

def parse_args():
    parser = argparse.ArgumentParser()

    localrank = '--local_rank'
    if torch.__version__ == '2.2.2':
        localrank = '--local-rank'

    parser.add_argument('--data', default='ntu', help='dataset')
    parser.add_argument(localrank, default=0, type=int, help='rank in local processes')
    parser.add_argument('--local',type=bool,default = True)
    parser.add_argument('--prefix', default='HumanML', help='prefix for saved filenames')
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--ckpt', default=None, help='absolute path to the checkpoint')
    parser.add_argument('--cfg_file' , type=str, help='absolute path to the config.yaml')
    parser.add_argument('--pretrained_ckpt',type=str,help='absolute path to the pretrained checkpoint, \
                        specify if you want to load a pretrained model which is not in the same directory as the experiment')
    parser.add_argument('--eval_multi',action='store_true',help='evaluate multiple checkpoints')
    parser.add_argument('--no_calc_score',default=False,action='store_true',help='do not calculate score')
    parser.add_argument('--eval_name', default='test', help='name of the evaluation')
    parser.add_argument('--multi_label_evals',action='store_true',help='evaluate multiple labels')
    parser.add_argument('--gpt_sim',default=False,action='store_true',help='use gpt to find similarity between predictions and ground truth')
    parser.add_argument('--Finetune', type=bool, default=True)
    args = parser.parse_args()

    config = load_config(args)
    config.args = args

    return args

def load_config(args):
    cfg=EasyDict()
    if args.cfg_file is not None and os.path.exists(args.cfg_file):
        print(f'Using config from {(args.cfg_file)}.')
        with open(args.cfg_file, 'r') as config_file:
            config_dict = yaml.safe_load(config_file)
        cfg.update(config_dict)
    elif args.cfg_file is not None:
        print(f"{args.cfg_file} not found")
        sys.exit(1)
    
    cfg.PRETRAINED_FOLDER = os.path.join(cfg.LOGDIR,'pretrain_models')
    cfg.CKPTDIR = os.path.join(cfg.LOGDIR,'checkpoints')
    cfg.JSONDIR = os.path.join(cfg.LOGDIR,'jsons')

    os.makedirs(cfg.LOGDIR,exist_ok=True)
    os.makedirs(cfg.JSONDIR,exist_ok=True)
    return cfg

if __name__ == "__main__":
    print("testing parser functionallity")
    args = parse_args()
    print(f'args: ' , args)