import torch
import torch.nn as nn

# Human Pose Perception
from hpp.pose_understanding import PoseUnderstanding
from hpp.pose_extraction import PoseExtraction
from hpp.pose_attention import PoseAttention
from hpp.graph_convolution import *
from hpp.make_graph import Graph

class HumanPosePerception(nn.Module):
    def __init__(self, num_class, in_channel, residual, dropout,
                 t_kernel_size, layout, strategy, hop_size, num_att_graph,
                 hpp_way, pretrain = True, lora_config = None) :
        super().__init__()
        self.hpp_way = hpp_way
        self.pretrain = pretrain
        # Graph
        graph = Graph(layout=layout, strategy = strategy, hop_size = hop_size)
        spatial_graph = torch.tensor(graph.A, dtype = torch.float32, requires_grad = False)
        self.register_buffer('spatial_graph', spatial_graph)

        # config
        kwargs = dict(s_kernel_size = spatial_graph.size(0),
                      t_kernel_size = t_kernel_size,
                      dropout = dropout,
                      residual = residual,
                      A_size = spatial_graph.size(),
                      hpp_way = self.hpp_way,
                      pretrain = self.pretrain,
                      lora_config = lora_config)

        # The setting of spatial-temporal attention graph convolutional networks (STA-GCN)
        if self.hpp_way == 'STAGCN' :
            f_config = [[in_channel, 32, 1], [32, 32, 1], [32, 32, 1], [32, 64, 2], [64, 64, 1]]
            a_config = [[128, 128, 1], [128, 128, 1], [128, 256, 2], [256, 256, 1], [256, 256, 1]]
            p_config = [[128, 128, 1], [128, 128, 1], [128, 256, 2], [256, 256, 1], [256, 256, 1]]
            self.PoseUnderstanding = PoseUnderstanding(config = f_config, **kwargs)
            self.PoseExtraction = PoseExtraction(config = a_config, num_class = num_class, num_att_graph = num_att_graph, **kwargs)
            self.PoseAttention = PoseAttention(config = p_config, num_att_graph = num_att_graph, **kwargs)
            self.output_channel = p_config[-1][1] + a_config[-1][1]

        # "HPP" - Human Pose Perception
        else :
            understanding_config = [[in_channel, 32, 1], [32, 32, 1], [32, 64, 1], [64, 64, 1], [64, 128, 1]]
            extraction_config = [[128, 128, 1], [128, 128, 1], [128, 256, 1], [256, 256, 1], [256, 256, 1]]
            attention_config = [[128, 128, 1], [128, 128, 1], [128, 256, 1], [256, 256, 1], [256, 256, 1]]
            self.PoseUnderstanding = PoseUnderstanding(config = understanding_config, **kwargs)
            self.PoseExtraction = PoseExtraction(config = extraction_config, num_class = num_class, num_att_graph = num_att_graph, **kwargs)
            self.PoseAttention = PoseAttention(config = attention_config, num_att_graph = num_att_graph, **kwargs)
            self.output_channel = extraction_config[-1][1] + attention_config[-1][1] 

    def forward(self, x):
        # [N : number of attention, c : channels, t : number of frame, v : joints]     
        skeleton_feature = self.PoseUnderstanding(x, self.spatial_graph)
        spatial_feature, att_node, att_graph = self.PoseExtraction(skeleton_feature, self.spatial_graph)

        # Attention Mechanism
        att_x = skeleton_feature * att_node
        if self.hpp_way == 'STAGCN' :
            attention_feature = self.PoseAttention(att_x, self.spatial_graph, att_graph)
        else :
            attention_feature = self.PoseAttention(att_x, None, att_graph)

        spatial_feature = spatial_feature.permute(0, 2, 3, 1)
        attention_feature = attention_feature.permute(0, 2, 3, 1)
        # spatial_feature : [batchsize, number of frame, joints(22), channel(256)]
        # attention_feature : [batchsize, number of frame, joints(22), channel(256)]
        # hpp_feature : [batchsize, number of frame, joints(22), channel(512)]
        hpp_feature = torch.cat([attention_feature, spatial_feature], dim = -1)
        return hpp_feature, att_node, att_graph