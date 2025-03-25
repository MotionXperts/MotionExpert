import torch
import torch.nn as nn
from .graph_convolution import Stgc_block
class Feature_extractor(nn.Module) :
    def __init__(self, config, s_kernel_size, t_kernel_size, dropout, residual, A_size,
                 PRETRAIN_SETTING, PRETRAIN, lora_config):
        super().__init__()

        self.bn = nn.BatchNorm1d(config[0][0] * A_size[2])
        self.PRETRAIN_SETTING = PRETRAIN_SETTING
        self.PRETRAIN = PRETRAIN
        kwargs = dict(s_kernel_size = s_kernel_size,
                      t_kernel_size = t_kernel_size,
                      dropout = dropout,
                      residual = residual,
                      A_size = A_size,
                      PRETRAIN_SETTING = self.PRETRAIN_SETTING,
                      lora_config = lora_config)

        # STA-GCN.
        if self.PRETRAIN_SETTING == 'STAGCN' :
            # The skeleton attributes joint and bone didn't share weights during training.
            # STGC Block for joints coordinates
            self.stgc_block1_0 = Stgc_block(in_channels = 3,
                                            out_channels = config[0][1],
                                            stride = config[0][2],
                                            s_kernel_size = s_kernel_size,
                                            t_kernel_size = t_kernel_size,
                                            dropout = 0,
                                            residual = False,
                                            A_size = A_size,
                                            PRETRAIN_SETTING = self.PRETRAIN_SETTING,
                                            PRETAIN = self.PRETRAIN,
                                            lora_config = lora_config)

            self.stgc_block1_1 = Stgc_block(config[1][0], config[1][1], config[1][2], **kwargs)
            self.stgc_block1_2 = Stgc_block(config[2][0], config[2][1], config[2][2], **kwargs)
            self.stgc_block1_3 = Stgc_block(config[3][0], config[3][1], config[3][2], **kwargs)
            self.stgc_block1_4 = Stgc_block(config[4][0], config[4][1], config[4][2], **kwargs)

            # STGC Block for bone coordinates
            self.stgc_block2_0 = Stgc_block(in_channels = 3,
                                            out_channels = config[0][1],
                                            stride = config[0][2],
                                            s_kernel_size = s_kernel_size,
                                            t_kernel_size = t_kernel_size,
                                            dropout = 0,
                                            residual = False,
                                            A_size = A_size,
                                            PRETRAIN_SETTING = self.PRETRAIN_SETTING)
            self.stgc_block2_1 = Stgc_block(config[1][0], config[1][1], config[1][2], **kwargs)
            self.stgc_block2_2 = Stgc_block(config[2][0], config[2][1], config[2][2], **kwargs)
            self.stgc_block2_3 = Stgc_block(config[3][0], config[3][1], config[3][2], **kwargs)
            self.stgc_block2_4 = Stgc_block(config[4][0], config[4][1], config[4][2], **kwargs)

        # Human Pose Perception.
        else :
            # The skeleton attributes joint and bone share weights during training.
            self.stgc_block0 = Stgc_block(in_channels = config[0][0],
                                          out_channels = config[0][1],
                                          stride = config[0][2],
                                          s_kernel_size = s_kernel_size,
                                          t_kernel_size = t_kernel_size,
                                          dropout = 0,
                                          residual = False,
                                          A_size = A_size,
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

    def forward(self, x, A) :
        N, C, T, V = x.size()
        x = x.permute(0, 3, 1, 2).contiguous().view(N, V * C, T)
        # Batch Normalization.
        x = self.bn(x)
        x = x.view(N, V, C, T).permute(0, 2, 3, 1).contiguous().view(N, C, T, V)
        # STA-GCN.
        if self.PRETRAIN_SETTING == 'STAGCN' :
            # Bone coordinate.
            bone = x[:, : 3, :, :]
            # Joint coordinate.
            joint = x[:, 3 :, :, :]
            
            # The feature of the bone attribute is trained independently in a separate branch.
            bone = self.stgc_block1_0(bone, A, None)
            bone = self.stgc_block1_1(bone, A, None)
            bone = self.stgc_block1_2(bone, A, None)
            bone = self.stgc_block1_3(bone, A, None)
            bone = self.stgc_block1_4(bone, A, None)

            # The feature of the joint attribute is trained independently in a separate branch.
            joint = self.stgc_block2_0(joint, A, None)
            joint = self.stgc_block2_1(joint, A, None)
            joint = self.stgc_block2_2(joint, A, None)
            joint = self.stgc_block2_3(joint, A, None)
            joint = self.stgc_block2_4(joint, A, None)

            # Concatenate the bone feature and joint feature.
            feature = torch.cat([bone, joint], dim=1)
            return feature

        # Human Pose Perception.
        else :
            # The Pose Understanding of Human Pose Perception.
            # The feature of the bone attribute is trained in the same neural network with
            # shared parameters.
            x = self.stgc_block0(x, A, None)
            x = self.stgc_block1(x, A, None)
            x = self.stgc_block2(x, A, None)
            x = self.stgc_block3(x, A, None)
            # x will then be taken as the input of Pose Attention.
            x = self.stgc_block4(x, A, None)

            # The Pose Extraction of Human Pose Perception.
            # x_last will then be used in Pose Extraction to generate both the attention
            # joint and the attention graph.
            x_last = self.stgc_block5(x, A, None)
            x_last = self.stgc_block6(x_last, A, None)
            x_last = self.stgc_block7(x_last, A, None)
            x_last = self.stgc_block8(x_last, A, None)
            x_last = self.stgc_block9(x_last, A, None)

            return x, x_last
