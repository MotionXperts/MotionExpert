import torch
import torch.nn as nn

from .graph_convolution import Stgc_block

class Feature_extractor(nn.Module):
    def __init__(self,
                 config,
                 num_person,
                 s_kernel_size,
                 t_kernel_size,
                 dropout,
                 residual,
                 A_size):
        super().__init__()

        # Batch Normalization
        self.bn = nn.BatchNorm1d(config[0][0] * A_size[2] * num_person)

        # STGC-Block config
        kwargs = dict(s_kernel_size=s_kernel_size,
                      t_kernel_size=t_kernel_size,
                      dropout=dropout,
                      residual=residual,
                      A_size=A_size)

        # branch1
        self.stgc_block1_0 = Stgc_block(in_channels=3,
                                        out_channels=config[0][1],
                                        stride=config[0][2],
                                        s_kernel_size=s_kernel_size,
                                        t_kernel_size=t_kernel_size,
                                        dropout=0,
                                        residual=False,
                                        A_size=A_size)
        self.stgc_block1_1 = Stgc_block(config[1][0], config[1][1], config[1][2], **kwargs)
        self.stgc_block1_2 = Stgc_block(config[2][0], config[2][1], config[2][2], **kwargs)
        self.stgc_block1_3 = Stgc_block(config[3][0], config[3][1], config[3][2], **kwargs)
        self.stgc_block1_4 = Stgc_block(config[4][0], config[4][1], config[4][2], **kwargs)

        # branch2
        self.stgc_block2_0 = Stgc_block(in_channels=3,
                                        out_channels=config[0][1],
                                        stride=config[0][2],
                                        s_kernel_size=s_kernel_size,
                                        t_kernel_size=t_kernel_size,
                                        dropout=0,
                                        residual=False,
                                        A_size=A_size)
        self.stgc_block2_1 = Stgc_block(config[1][0], config[1][1], config[1][2], **kwargs)
        self.stgc_block2_2 = Stgc_block(config[2][0], config[2][1], config[2][2], **kwargs)
        self.stgc_block2_3 = Stgc_block(config[3][0], config[3][1], config[3][2], **kwargs)
        self.stgc_block2_4 = Stgc_block(config[4][0], config[4][1], config[4][2], **kwargs)

    def forward(self, x, A):
        # Batch Normalization
        N, C, T, V = x.size()
        x = x.permute(0, 3, 1, 2).contiguous().view(N, V * C, T)
        x = self.bn(x)
        x = x.view(N, V, C, T).permute(0, 2, 3,1).contiguous().view(N, C, T, V)
       
        x1 = x[:, :3, :, :] # bones
        x2 = x[:, 3:, :, :] # joints

        # branch1 : bone
        x1 = self.stgc_block1_0(x1, A, None)
        x1 = self.stgc_block1_1(x1, A, None)
        x1 = self.stgc_block1_2(x1, A, None)
        x1 = self.stgc_block1_3(x1, A, None)
        x1 = self.stgc_block1_4(x1, A, None)  # N*M, C=64, T, V

        # branch2 : joints
        x2 = self.stgc_block2_0(x2, A, None)
        x2 = self.stgc_block2_1(x2, A, None)
        x2 = self.stgc_block2_2(x2, A, None)
        x2 = self.stgc_block2_3(x2, A, None)
        x2 = self.stgc_block2_4(x2, A, None) # N*M, C=64, T, V
 
        # concat
        feature = torch.cat([x1, x2], dim=1) # N*M, C=128, T, V

        return feature
