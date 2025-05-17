import torch
import torch.nn as nn
import torch.nn.functional as F
from hpp.graph_convolution import Stgc_block
class PoseExtraction(nn.Module) :
    def __init__(self, config, num_class, num_att_graph, s_kernel_size, t_kernel_size, dropout, residual, A_size,
                 hpp_way, pretrain, lora_config) :
        super().__init__()

        self.hpp_way = hpp_way
        self.pretrain = pretrain
        kwargs = dict(s_kernel_size = s_kernel_size,
                      t_kernel_size = t_kernel_size,
                      dropout = dropout,
                      residual = residual,
                      A_size = A_size,
                      hpp_way = hpp_way,
                      pretrain = pretrain,
                      lora_config = lora_config)

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
        self.num_att_graph = num_att_graph
        self.att_A_conv = nn.Conv2d(num_class, num_att_graph * A_size[2], kernel_size = 1, padding = 0, stride = 1, bias = False)
        self.att_A_bn = nn.BatchNorm2d(num_att_graph * A_size[2])
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

    def forward(self, skeleton_feat, A) :
        N, C, T, V = skeleton_feat.size()
        skeleton_feat = self.stgc_block0(skeleton_feat, A, None)
        skeleton_feat = self.stgc_block1(skeleton_feat, A, None)
        skeleton_feat = self.stgc_block2(skeleton_feat, A, None)
        skeleton_feat = self.stgc_block3(skeleton_feat, A, None)
        skeleton_feat = self.stgc_block4(skeleton_feat, A, None)

        skeleton_att = self.att_bn0(skeleton_feat) 
        skeleton_att = self.att_conv(skeleton_att)

        # Generate attention joint.
        x_node = self.att_node_conv(skeleton_att)
        x_node = self.att_node_bn(x_node)
        x_node = F.interpolate(x_node, size = (T, V))
        att_node = self.sigmoid(x_node)

        # Generate attention graph.
        x_graph = F.avg_pool2d(skeleton_att, (skeleton_att.size()[2], 1))
        x_graph = self.att_A_conv(x_graph)
        x_graph = self.att_A_bn(x_graph)
        x_graph = x_graph.view(N, self.num_att_graph, V, V)
        x_graph = self.tanh(x_graph)
        att_graph = self.relu(x_graph)

        return skeleton_feat, att_node, att_graph