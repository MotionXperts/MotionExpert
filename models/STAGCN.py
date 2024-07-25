import torch
import torch.nn as nn

###### STA-GCN ######
from net.Utils_attention.attention_branch import *
from net.Utils_attention.perception_branch import *
from net.Utils_attention.feature_extractor import *
from net.Utils_attention.graph_convolution import *
from net.Utils_attention.make_graph import Graph

class STA_GCN(nn.Module):
    def __init__(self, num_class, in_channels,
                 residual, dropout, num_person,
                 t_kernel_size, layout, strategy, hop_size, num_att_A):
        super().__init__()

        # Graph
        graph = Graph(layout=layout, strategy=strategy, hop_size=hop_size)
        A = torch.tensor(graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)

        # config
        kwargs = dict(s_kernel_size=A.size(0),
                      t_kernel_size=t_kernel_size,
                      dropout=dropout,
                      residual=residual,
                      A_size=A.size())

        # Feature Extractor
        f_config = [[in_channels, 32, 1], [32, 32, 1], [32, 64, 1], [64, 64, 1], [64, 128, 1], [128, 128, 1], [128, 128, 1], [128, 256, 1], [256, 256, 1], [256, 256, 1]]
        self.feature_extractor = Feature_extractor(config=f_config, **kwargs)

        # Attention Branch
        a_config = [[128, 128, 1], [128, 128, 1], [128, 256, 1], [256, 256, 1], [256, 256, 1]]
        self.attention_branch = Attention_branch(config=f_config,num_class=num_class, num_att_A=num_att_A, **kwargs)

        # Perception Branch
        p_config = [[128, 128, 1], [128, 128, 1], [128, 256, 1], [256, 256, 1], [256, 256, 1]]
        self.perception_branch = Perception_branch(config=p_config,num_class=num_class, num_att_A=num_att_A, **kwargs)

        self.output_channel = p_config[-1][1] + a_config[-1][1] ## output channel is concatnating these 2 embeddings


    def forward(self, x):
        N, c, t, v = x.size() # N : number of attention, c : channel, t : time, v : vertex

        # Feature Extractor
        feature,feature_last = self.feature_extractor(x, self.A)
        # Attention Branch
        att_node, att_A = self.attention_branch(feature_last, self.A)
        # Attention Mechanism
        att_x = feature * att_node
        # Perception Branch
        perception_last = self.perception_branch(att_x, self.A, att_A, N)

        perception_last = perception_last.permute(0,2,3,1) # batchsize, channel, seq_length, vertex
        feature_last    = feature_last.permute(0,2,3,1)   # batchsize, channel, seq_length, vertex

        # perception_last torch.Size : [ batchsize, seq_length, vertex(22), channel(256) ]
        # attention_last torch.Size : [ batchsize, seq_length, vertex(22), channel(256) ]
        PA_embedding = torch.cat([perception_last, feature_last], dim=-1) # [ batchsize, seq_length, vertex(22), channel(512) ]
        return PA_embedding, att_node, att_A

