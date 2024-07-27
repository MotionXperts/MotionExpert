import torch
import torch.nn as nn

###### STA-GCN ######
from net.Utils_attention.attention_branch import *
from net.Utils_attention.perception_branch import *
from net.Utils_attention.feature_extractor import *
from net.Utils_attention.graph_convolution import *
from net.Utils_attention.make_graph import Graph

class STA_GCN(nn.Module):
    def __init__(self, num_class, in_channels, residual, dropout, 
                 t_kernel_size, layout, strategy, hop_size, num_att_A, PRETRAIN_SETTING):
        super().__init__()
        self.PRETRAIN_SETTING = PRETRAIN_SETTING
        # Graph
        graph = Graph(layout=layout, strategy=strategy, hop_size=hop_size)
        A = torch.tensor(graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)

        # config
        kwargs = dict(s_kernel_size=A.size(0),
                      t_kernel_size=t_kernel_size,
                      dropout=dropout,
                      residual=residual,
                      A_size=A.size(),
                      PRETRAIN_SETTING = self.PRETRAIN_SETTING)


        
        if self.PRETRAIN_SETTING == 'STAGCN' :
            f_config = [[in_channels, 32, 1],   [32, 32, 1],    [32, 32, 1],    [32, 64, 2],    [64, 64, 1]]
            a_config = [[128, 128, 1],          [128, 128, 1],  [128, 256, 2],  [256, 256, 1],  [256, 256, 1]]
            p_config = [[128, 128, 1],          [128, 128, 1],  [128, 256, 2],  [256, 256, 1],  [256, 256, 1]]

        # PRETRAIN_SETTING : 'Attention'
        else : 
            f_config = [[in_channels, 32, 1],   [32, 32, 1],    [32, 64, 1],    [64, 64, 1],    [64, 128, 1], 
                        [128, 128, 1],          [128, 128, 1],  [128, 256, 1],  [256, 256, 1],  [256, 256, 1]]
            a_config = [[256, 256, 1]]
            p_config = [[128, 128, 1],          [128, 128, 1],  [128, 256, 1],  [256, 256, 1],  [256, 256, 1]]

        self.feature_extractor = Feature_extractor(config=f_config, **kwargs)
        self.attention_branch = Attention_branch(config=a_config,num_class=num_class, num_att_A=num_att_A, **kwargs)
        self.perception_branch = Perception_branch(config=p_config,num_class=num_class, num_att_A=num_att_A, **kwargs)
        self.output_channel = p_config[-1][1] + a_config[-1][1] 

    def forward(self, x):
        # [N : number of attention, c : channels, t : number of frame, v : joints]
        N, c, t, v = x.size() 
 
        if self.PRETRAIN_SETTING == 'STAGCN' :
            feature = self.feature_extractor(x, self.A)

            attention_last , att_node, att_A = self.attention_branch(feature, self.A)

        else :
            feature,feature_last = self.feature_extractor(x, self.A)

            att_node, att_A = self.attention_branch(feature_last, self.A)

        # Attention Mechanism
        att_x = feature * att_node

        perception_last = self.perception_branch(att_x, self.A, att_A)

        '''
            The dimension of perception_last is [batchsize, channel(256), number of frame, joints(22)]
            The dimension of attention_last is [batchsize, channel(256), number of frame, joints(22)]
            The dimension of feature_last is [batchsize, channel(256), number of frame, joints(22)]
            Convert to [batchsize, number of frame, joints(22), channel(256)]
            Concatenate two embeding to [batchsize, number of frame, joints(22), channel(512)]
        '''
        perception_last = perception_last.permute(0,2,3,1)  
        if self.PRETRAIN_SETTING == 'STAGCN' :
            attention_last = attention_last.permute(0,2,3,1) 
            concatenate_embedding = torch.cat([perception_last, attention_last], dim=-1)
        else :
            feature_last   = feature_last.permute(0,2,3,1)     
            concatenate_embedding = torch.cat([perception_last, feature_last], dim=-1) 

        return concatenate_embedding, att_node, att_A