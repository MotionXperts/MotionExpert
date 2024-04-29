import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

#from net.utils.tgcn import ConvTemporalGraphical
#from net.utils.graph import Graph
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
        ## in_channel   6 -> 64
        f_config = [[in_channels, 32, 1], [32, 32, 1], [32, 32, 1], [32, 64, 2], [64, 64, 1]]
        self.feature_extractor = Feature_extractor(config=f_config, num_person=num_person, **kwargs)

        # Attention Branch
        a_config = [[128, 128, 1], [128, 128, 1], [128, 256, 2], [256, 256, 1], [256, 256, 1]]
        self.attention_branch = Attention_branch(config=a_config,num_class=num_class, num_att_A=num_att_A, **kwargs)

        # Perception Branch
        p_config = [[128, 128, 1], [128, 128, 1], [128, 256, 2], [256, 256, 1], [256, 256, 1]]
        self.perception_branch = Perception_branch(config=p_config,num_class=num_class, num_att_A=num_att_A, **kwargs)

    def forward(self, x):
        N, c, t, v = x.size() # N : number of attention, c : channel, t : time, v : vertex

        # Feature Extractor
        feature = self.feature_extractor(x, self.A)

        # Attention Branch
        attention_last , att_node, att_A = self.attention_branch(feature, self.A, N)
        # Attention Mechanism
        att_x = feature * att_node
        # Perception Branch
        perception_last = self.perception_branch(att_x, self.A, att_A, N)

        perception_last = perception_last.permute(0,2,3,1) # batchsize, channel, seq_length, vertex
        attention_last = attention_last.permute(0,2,3,1)   # batchsize, channel, seq_length, vertex

        # perception_last torch.Size : ([8, 118, 22, 256]) // batchsize, seq_length, vertex, channel 
        # attention_last torch.Size : ([8, 118, 22, 256])  // batchsize, seq_length, vertex, channel 
        PA_embedding = torch.cat([perception_last, attention_last], dim=-1) # ([8, 118, 22, 512])

        self.embedding = nn.Sequential(nn.Linear(11264,5632),  # 22 * 512 -> 22 * 768
                                       nn.ReLU(),nn.Linear(5632,768)).to(PA_embedding.get_device()) # 22 * 256 -> 22 * 1024
        
        # batch_size, seq_length, feature_dim = input_ids.shape
        # 512 -> 768/22
        # 22 , 768/22
        output_embedding = self.embedding(PA_embedding.view(-1,11264)).view(N, -1, 768)

        # att_node torch.Size([8, 1, 235, 22])
        # att_A torch.Size([8, 4, 22, 22])  
        return output_embedding, att_node, att_A
