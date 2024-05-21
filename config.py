from easydict import EasyDict as edict
import os

# Get the configuration dictionary whose keys can be accessed with dot.
CONFIG = edict()

# ========TRANSFORMER PARAMETERs=========
CONFIG.USE_EMBS = False

if CONFIG.USE_EMBS:
    #### USE_EMBS:
    CONFIG.D_MODEL = 16
    CONFIG.NUM_HEADS = 2
    CONFIG.DIM_FEEDFORWARD = 128
else:
    #### USE_KEYPOINTS:
    CONFIG.D_MODEL = 34
    CONFIG.NUM_HEADS = 2
    CONFIG.DIM_FEEDFORWARD = 256

CONFIG.NUM_ENCODER_LAYERS = 2
CONFIG.LAYER_NORM_EPS = 1e-5
CONFIG.NUM_CLASS = 4
CONFIG.USE_CRF = True
CONFIG.ADD_NOISE = False

CONFIG.HIDDEN_CHANNEL = 32
CONFIG.OUT_CHANNEL = 128 
# ========Hyperparameters============
# CONFIG.LR = 2e-5
CONFIG.LR = 1e-4
CONFIG.EVAL_STEPS = 1000
CONFIG.SAVE_STEPS = 2000
CONFIG.NUM_EPOCHS = 1000
CONFIG.BATCH_SIZE = 128
# ========Path=======
# There are three case
# 1. Train HumanML3D : Pretrained = False, Finetune = False
# 2. Finetune to Skating : Pretrained = True, Finetune = True
# 3. Train Skating : Pretrained = False, Finetune = True

CONFIG.Pretrained = False
CONFIG.Finetune = True

CONFIG.weight_path = '/home/weihsin/projects/MotionExpert/models_local_new/Local_epoch50.pt'

if CONFIG.Finetune == False:
    CONFIG.data        = '/home/weihsin/datasets/FigureSkate/HumanML3D_l/local_human_train.pkl'
    CONFIG.out_dir     = './models_local_new'
    CONFIG.prefix      = 'Local'
    CONFIG.test_data   = '/home/weihsin/datasets/FigureSkate/HumanML3D_l/local_human_test.pkl'
    CONFIG.result_dir  = 'STAGCN_output_local_new'

else:
    CONFIG.data        = '/home/weihsin/datasets/VQA/train_local_with_standard.pkl'
    CONFIG.out_dir     = './models_finetune_new2'
    CONFIG.prefix      = 'Finetune'
    CONFIG.test_data   = '/home/weihsin/datasets/VQA/test_local.pkl'
    CONFIG.result_dir  = 'STAGCN_output_finetune_new2'