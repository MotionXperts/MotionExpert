import torch
import torch.nn as nn
import torch.nn.functional as F
from net.Utils_attention.graph_convolution import Stgc_block
class Attention_branch(nn.Module) :
    def __init__(self, config, num_class, num_att_A, s_kernel_size, t_kernel_size, dropout, residual, A_size,
                 PRETRAIN_SETTING, PRETRAIN, lora_config) :
        super().__init__()

        # STGC Block
        self.PRETRAIN_SETTING = PRETRAIN_SETTING
        self.PRETRAIN = PRETRAIN
        kwargs = dict(s_kernel_size = s_kernel_size,
                      t_kernel_size = t_kernel_size,
                      dropout = dropout,
                      residual = residual,
                      A_size = A_size,
                      PRETRAIN_SETTING = self.PRETRAIN_SETTING,
                      PRETRAIN = self.PRETRAIN,
                      lora_config = lora_config)

        if self.PRETRAIN_SETTING == 'STAGCN' :
            self.stgc_block0 = Stgc_block(config[0][0], config[0][1], config[0][2], **kwargs)
            self.stgc_block1 = Stgc_block(config[1][0], config[1][1], config[1][2], **kwargs)
            self.stgc_block2 = Stgc_block(config[2][0], config[2][1], config[2][2], **kwargs)
            self.stgc_block3 = Stgc_block(config[3][0], config[3][1], config[3][2], **kwargs)
            self.stgc_block4 = Stgc_block(config[4][0], config[4][1], config[4][2], **kwargs)

        # Layers that process the input embeddings and output embeddings used to generate attention information.
        self.att_bn0 = nn.BatchNorm2d(config[-1][1])
        self.att_conv = nn.Conv2d(config[-1][1], num_class, kernel_size = 1, padding = 0, stride = 1, bias = False)

        # Mechanism that aims to generate attention joint.
        self.att_node_conv = nn.Conv2d(num_class, 1, kernel_size = 1, padding = 0, stride = 1, bias = False)
        self.att_node_bn = nn.BatchNorm2d(1)
        self.sigmoid = nn.Sigmoid()

        # Mechanism that aims to generate attention graph.
        self.num_att_A = num_att_A
        self.att_A_conv = nn.Conv2d(num_class, num_att_A * A_size[2], kernel_size = 1, padding = 0, stride = 1, bias = False)
        self.att_A_bn = nn.BatchNorm2d(num_att_A * A_size[2])
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

    def forward(self, x, A) :
        N, C, T, V = x.size()
        
        if self.PRETRAIN_SETTING == 'STAGCN' :
            # ST-GC Block.
            x = self.stgc_block0(x, A, None)
            x = self.stgc_block1(x, A, None)
            x = self.stgc_block2(x, A, None)
            x = self.stgc_block3(x, A, None)
            x = self.stgc_block4(x, A, None)

        # Pose Extraction of Human Pose Perception generates the attention joint and graph.
        x_att = self.att_bn0(x) 
        x_att = self.att_conv(x_att)
        # Generate attention joint.
        x_node = self.att_node_conv(x_att)
        x_node = self.att_node_bn(x_node)
        x_node = F.interpolate(x_node, size = (T, V))
        att_node = self.sigmoid(x_node)
        # Generate attention graph.
        x_A = F.avg_pool2d(x_att, (x_att.size()[2], 1))
        x_A = self.att_A_conv(x_A)
        x_A = self.att_A_bn(x_A)
        x_A = x_A.view(N, self.num_att_A, V, V)
        x_A = self.tanh(x_A)
        att_A = self.relu(x_A)

        # STA-GCN.
        if self.PRETRAIN_SETTING == 'STAGCN' :
            return x , att_node, att_A
        # Human Pose Perception.
        else :
            # Pose Extraction of Human Pose Perception is responsible only for generating the attention
            # joints and the graph.
            return att_node, att_A