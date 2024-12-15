import torch
import torch.nn as nn

from .graph_convolution import Stgc_block

class Feature_extractor(nn.Module):
    def __init__(self, config, s_kernel_size, t_kernel_size, dropout, residual, A_size, PRETRAIN_SETTING, PRETRAIN):
        super().__init__()

        self.bn = nn.BatchNorm1d(config[0][0] * A_size[2])
        self.PRETRAIN_SETTING = PRETRAIN_SETTING
        self.PRETRAIN = PRETRAIN
        kwargs = dict(s_kernel_size=s_kernel_size,
                      t_kernel_size=t_kernel_size,
                      dropout=dropout,
                      residual=residual,
                      A_size=A_size,
                      PRETRAIN_SETTING = self.PRETRAIN_SETTING)
        
        if self.PRETRAIN_SETTING == 'STAGCN' :

            '''Branch1 is for joints coordinates''' 
            self.stgc_block1_0 = Stgc_block(in_channels=3,
                                            out_channels=config[0][1],
                                            stride=config[0][2],
                                            s_kernel_size=s_kernel_size,
                                            t_kernel_size=t_kernel_size,
                                            dropout=0,
                                            residual=False,
                                            A_size=A_size,
                                            PRETRAIN_SETTING = self.PRETRAIN_SETTING,
                                            PRETAIN = self.PRETRAIN)

            self.stgc_block1_1 = Stgc_block(config[1][0], config[1][1], config[1][2], **kwargs)
            self.stgc_block1_2 = Stgc_block(config[2][0], config[2][1], config[2][2], **kwargs)
            self.stgc_block1_3 = Stgc_block(config[3][0], config[3][1], config[3][2], **kwargs)
            self.stgc_block1_4 = Stgc_block(config[4][0], config[4][1], config[4][2], **kwargs)

            '''Branch2 is for bone coordinates'''
            self.stgc_block2_0 = Stgc_block(in_channels=3,
                                            out_channels=config[0][1],
                                            stride=config[0][2],
                                            s_kernel_size=s_kernel_size,
                                            t_kernel_size=t_kernel_size,
                                            dropout=0,
                                            residual=False,
                                            A_size=A_size,
                                            PRETRAIN_SETTING = self.PRETRAIN_SETTING)
            self.stgc_block2_1 = Stgc_block(config[1][0], config[1][1], config[1][2], **kwargs)
            self.stgc_block2_2 = Stgc_block(config[2][0], config[2][1], config[2][2], **kwargs)
            self.stgc_block2_3 = Stgc_block(config[3][0], config[3][1], config[3][2], **kwargs)
            self.stgc_block2_4 = Stgc_block(config[4][0], config[4][1], config[4][2], **kwargs)

        else :
            self.stgc_block0 = Stgc_block(  in_channels=config[0][0],
                                            out_channels=config[0][1],
                                            stride=config[0][2],
                                            s_kernel_size=s_kernel_size,
                                            t_kernel_size=t_kernel_size,
                                            dropout=0,
                                            residual=False,
                                            A_size=A_size,
                                            PRETRAIN_SETTING = self.PRETRAIN_SETTING)
            self.stgc_block1 = Stgc_block(config[1][0], config[1][1], config[1][2], **kwargs)
            self.stgc_block2 = Stgc_block(config[2][0], config[2][1], config[2][2], **kwargs)
            self.stgc_block3 = Stgc_block(config[3][0], config[3][1], config[3][2], **kwargs)
            self.stgc_block4 = Stgc_block(config[4][0], config[4][1], config[4][2], **kwargs)
            self.stgc_block5 = Stgc_block(config[5][0], config[5][1], config[5][2], **kwargs)
            self.stgc_block6 = Stgc_block(config[6][0], config[6][1], config[6][2], **kwargs)
            self.stgc_block7 = Stgc_block(config[7][0], config[7][1], config[7][2], **kwargs)
            self.stgc_block8 = Stgc_block(config[8][0], config[8][1], config[8][2], **kwargs)
            self.stgc_block9 = Stgc_block(config[9][0], config[9][1], config[9][2], **kwargs)

    def forward(self, x, A):
        # Batch Normalization
        N, C, T, V = x.size()
        x = x.permute(0, 3, 1, 2).contiguous().view(N, V * C, T)
        x = self.bn(x)
        x = x.view(N, V, C, T).permute(0, 2, 3,1).contiguous().view(N, C, T, V)
        if self.PRETRAIN_SETTING == 'STAGCN' :
            x1 = x[:, :3, :, :] # bones
            x2 = x[:, 3:, :, :] # joints
            
            # branch1 : bone
            x1 = self.stgc_block1_0(x1, A, None)
            x1 = self.stgc_block1_1(x1, A, None)
            x1 = self.stgc_block1_2(x1, A, None)
            x1 = self.stgc_block1_3(x1, A, None)
            x1 = self.stgc_block1_4(x1, A, None)  

            # branch2 : joints
            x2 = self.stgc_block2_0(x2, A, None)
            x2 = self.stgc_block2_1(x2, A, None)
            x2 = self.stgc_block2_2(x2, A, None)
            x2 = self.stgc_block2_3(x2, A, None)
            x2 = self.stgc_block2_4(x2, A, None) 

            # Concatenate
            feature = torch.cat([x1, x2], dim=1)
            return feature
        
        else :
            x = self.stgc_block0(x, A, None)
            x = self.stgc_block1(x, A, None)
            x = self.stgc_block2(x, A, None)
            x = self.stgc_block3(x, A, None)
            x = self.stgc_block4(x, A, None)  
            x_last = self.stgc_block5(x, A, None)
            x_last = self.stgc_block6(x_last, A, None)
            x_last = self.stgc_block7(x_last, A, None)
            x_last = self.stgc_block8(x_last, A, None)
            x_last = self.stgc_block9(x_last, A, None)

            return x, x_last
