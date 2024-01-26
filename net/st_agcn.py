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
'''
class Model(nn.Module):
    r"""Spatial temporal graph convolutional networks.

    Args:
        in_channels (int): Number of channels in the input data
        num_class (int): Number of classes for the classification task
        graph_args (dict): The arguments for building the graph
        edge_importance_weighting (bool): If ``True``, adds a learnable
            importance weighting to the edges of the graph
        **kwargs (optional): Other parameters for graph convolution units

    Shape:
        - Input: :math:`(N, in_channels, T_{in}, V_{in}, M_{in})`
        - Output: :math:`(N, num_class)` where
            :math:`N` is a batch size,
            :math:`T_{in}` is a length of input sequence,
            :math:`V_{in}` is the number of graph nodes,
            :math:`M_{in}` is the number of instance in a frame.
    """

    def __init__(self, in_channels, num_class, graph_args,t_kernel_size, residual,num_person,num_att_A,
                 edge_importance_weighting, dropout, **kwargs):
        super().__init__()

        # load graph
        self.graph = Graph(**graph_args)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)

        # build networks
        spatial_kernel_size = A.size(0)
        temporal_kernel_size = 9
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))
        kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}

        
        ###### STA-GCN ######

        # config
        kwargs = dict(s_kernel_size=A.size(0),
                      t_kernel_size=t_kernel_size,
                      dropout=dropout,
                      residual=residual,
                      A_size=A.size())

        # Feature Extractor
        f_config = [[in_channels, 32, 1], [32, 32, 1], [32, 32, 1], [32, 64, 2], [64, 64, 1]]
        self.feature_extractor = Feature_extractor(config=f_config, num_person=num_person, **kwargs)

        # Attention Branch
        a_config = [[128, 128, 1], [128, 128, 1], [128, 256, 2], [256, 256, 1], [256, 256, 1]]
        self.attention_branch = Attention_branch(config=a_config, num_class=num_class, num_att_A=num_att_A, **kwargs)

        # Perception Branch
        p_config = [[128, 128, 1], [128, 128, 1], [128, 256, 2], [256, 256, 1], [256, 256, 1]]
        self.perception_branch = Perception_branch(config=p_config, num_class=num_class, num_att_A=num_att_A, **kwargs)

    def forward(self, x):

        N, c, t, v, M = x.size() # BatchSize : 128, channel = 3, t = 150 , v = 18 , M = 2
        
        print("X first",x.size())
        # Feature Extractor
        feature = self.feature_extractor(x, self.A)
        print("feature",feature.shape)
        # Attention Branch
        output_ab, att_node, att_A = self.attention_branch(feature, self.A, N, M)
        print("output_ab",output_ab.shape)
        print("att_node",att_node.shape)
        print("att_A",att_A.shape)
        # Attention Mechanism
        att_x = feature * att_node
        print("att_x",att_x.shape)
        # Perception Branch
        output_pb = self.perception_branch(att_x, self.A, att_A, N, M)
        print("output_pb",output_pb.shape)
        return output_ab, output_pb, att_node, att_A
'''
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
        f_config = [[in_channels, 32, 1], [32, 32, 1], [32, 32, 1], [32, 64, 2], [64, 64, 1]]
        self.feature_extractor = Feature_extractor(config=f_config, num_person=num_person, **kwargs)

        # Attention Branch
        a_config = [[128, 128, 1], [128, 128, 1], [128, 256, 2], [256, 256, 1], [256, 256, 1]]
        self.attention_branch = Attention_branch(config=a_config, num_att_A=num_att_A, **kwargs)

        # Perception Branch
        p_config = [[128, 128, 1], [128, 128, 1], [128, 256, 2], [256, 256, 1], [256, 256, 1]]
        self.perception_branch = Perception_branch(config=p_config, num_att_A=num_att_A, **kwargs)

    def forward(self, x):
        # FIXME: weihsin
        print("x",x.size())
        N, c, t, v = x.size()
        # Feature Extractor
        feature = self.feature_extractor(x, self.A)

        # Attention Branch
        # output_ab, att_node, att_A = self.attention_branch(feature, self.A, N, M)
        attention_last , att_node, att_A = self.attention_branch(feature, self.A, N)
        # Attention Mechanism
        att_x = feature * att_node

        # Perception Branch
        perception_last = self.perception_branch(att_x, self.A, att_A, N)

        # perception_last torch.Size([8, 256, 118, 22])
        # attention_last torch.Size([8, 256, 118, 22])
        perception_last = perception_last.permute(0,2,3,1)
        attention_last = attention_last.permute(0,2,3,1)
        PA_embedding = torch.cat([perception_last, attention_last], dim=-1)
        # embedding torch.Size([8, 118, 22, 512])
        self.embedding = nn.Sequential(nn.Linear(11264,5632),  # 22 * 512 -> 22 * 256
                                       nn.ReLU(),nn.Linear(5632,768)).to(PA_embedding.get_device()) # 22 * 256 -> 22 * 1024
        
        # batch_size, seq_length, feature_dim = input_ids.shape
        output_embedding = self.embedding(PA_embedding.view(-1,11264)).view(N, -1, 768)

        # att_node torch.Size([8, 1, 235, 22])
        # att_A torch.Size([8, 4, 22, 22])  
        return output_embedding, att_node, att_A
