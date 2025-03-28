import torch, torch.nn as nn, torch.nn.functional as F
from .graph_convolution import Stgc_block

class Perception_branch(nn.Module) :
    def __init__(self, config, num_class, num_att_A, s_kernel_size, t_kernel_size, dropout, residual, A_size,
                 PRETRAIN_SETTING, PRETRAIN, lora_config) :
        super().__init__()
        self.PRETRAIN_SETTING = PRETRAIN_SETTING
        self.PRETRAIN = PRETRAIN
        kwargs = dict(s_kernel_size = s_kernel_size,
                      t_kernel_size = t_kernel_size,
                      dropout = dropout,
                      residual = residual,
                      A_size = A_size,
                      PRETRAIN_SETTING = self.PRETRAIN_SETTING,
                      use_att_A = True,
                      num_att_A = num_att_A,
                      PRETRAIN = self.PRETRAIN,
                      lora_config = lora_config)

        # The Pose Attention in Human Pose Perception and the perception branch of STA-GCN will both
        # used this module. However, the architechtrue of the STGC block depends on the
        # PRETRAIN_SETTING : the former will use Attention, while the latter will use STAGCN.
        self.stgc_block0 = Stgc_block(config[0][0], config[0][1], config[0][2], **kwargs)
        self.stgc_block1 = Stgc_block(config[1][0], config[1][1], config[1][2], **kwargs)
        self.stgc_block2 = Stgc_block(config[2][0], config[2][1], config[2][2], **kwargs)
        self.stgc_block3 = Stgc_block(config[3][0], config[3][1], config[3][2], **kwargs)
        self.stgc_block4 = Stgc_block(config[4][0], config[4][1], config[4][2], **kwargs)

    def forward(self, x, A, att_A) :
        x = self.stgc_block0(x, A, att_A)
        x = self.stgc_block1(x, A, att_A)
        x = self.stgc_block2(x, A, att_A)
        x = self.stgc_block3(x, A, att_A)
        x = self.stgc_block4(x, A, att_A)
        return x