import torch
import torch.nn as nn
from .graph_convolution import Stgc_block
class PoseUnderstanding(nn.Module) :
    def __init__(self, config, s_kernel_size, t_kernel_size, dropout, residual, A_size,
                 hpp_way, pretrain, lora_config):
        super().__init__()

        self.bn = nn.BatchNorm1d(config[0][0] * A_size[2])
        self.hpp_way = hpp_way
        self.pretrain = pretrain
        kwargs = dict(s_kernel_size = s_kernel_size,
                      t_kernel_size = t_kernel_size,
                      dropout = dropout,
                      residual = residual,
                      A_size = A_size,
                      hpp_way = hpp_way,
                      lora_config = lora_config)

        if hpp_way == 'STAGCN' :
            # The skeleton attributes joint and bone didn't share weights during training.
            # joints coordinates
            self.stgc_block1_0 = Stgc_block(in_channels = 3,
                                            out_channels = config[0][1],
                                            stride = config[0][2],
                                            s_kernel_size = s_kernel_size,
                                            t_kernel_size = t_kernel_size,
                                            dropout = 0,
                                            residual = False,
                                            A_size = A_size,
                                            hpp_way = hpp_way,
                                            pretrain = pretrain,
                                            lora_config = lora_config)
            self.stgc_block1_1 = Stgc_block(config[1][0], config[1][1], config[1][2], **kwargs)
            self.stgc_block1_2 = Stgc_block(config[2][0], config[2][1], config[2][2], **kwargs)
            self.stgc_block1_3 = Stgc_block(config[3][0], config[3][1], config[3][2], **kwargs)
            self.stgc_block1_4 = Stgc_block(config[4][0], config[4][1], config[4][2], **kwargs)

            # bone coordinates
            self.stgc_block2_0 = Stgc_block(in_channels = 3,
                                            out_channels = config[0][1],
                                            stride = config[0][2],
                                            s_kernel_size = s_kernel_size,
                                            t_kernel_size = t_kernel_size,
                                            dropout = 0,
                                            residual = False,
                                            A_size = A_size,
                                            hpp_way = hpp_way)
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
                                          hpp_way = hpp_way)
            self.stgc_block1 = Stgc_block(config[1][0], config[1][1], config[1][2], **kwargs)
            self.stgc_block2 = Stgc_block(config[2][0], config[2][1], config[2][2], **kwargs)
            self.stgc_block3 = Stgc_block(config[3][0], config[3][1], config[3][2], **kwargs)
            self.stgc_block4 = Stgc_block(config[4][0], config[4][1], config[4][2], **kwargs)

    def forward(self, skeleton_coords, A) :
        N, C, T, V = skeleton_coords.size()
        skeleton_coords = skeleton_coords.permute(0, 3, 1, 2).contiguous().view(N, V * C, T)
        # Batch Normalization.
        skeleton_coords = self.bn(skeleton_coords)
        skeleton_coords = skeleton_coords.view(N, V, C, T).permute(0, 2, 3, 1).contiguous().view(N, C, T, V)

        if self.hpp_way == 'STAGCN' :
            joint = skeleton_coords[:, : 3, :, :]
            bone = skeleton_coords[:, 3 :, :, :]
            
            # The feature of the joint attribute is trained independently in a separate branch.
            joint = self.stgc_block1_0(joint, A, None)
            joint = self.stgc_block1_1(joint, A, None)
            joint = self.stgc_block1_2(joint, A, None)
            joint = self.stgc_block1_3(joint, A, None)
            joint = self.stgc_block1_4(joint, A, None)

            # The feature of the bone attribute is trained independently in a separate branch.
            bone = self.stgc_block2_0(bone, A, None)
            bone = self.stgc_block2_1(bone, A, None)
            bone = self.stgc_block2_2(bone, A, None)
            bone = self.stgc_block2_3(bone, A, None)
            bone = self.stgc_block2_4(bone, A, None)

            # Concatenate the bone feature and joint feature.
            skeleton_feat = torch.cat([joint, bone], dim=1)

        # Human Pose Perception.
        else :
            # The Pose Understanding of Human Pose Perception learn the feature of skeleton attributes (joint
            # and bone) in the same neural network with shared parameters.
            skeleton = self.stgc_block0(skeleton_coords, A, None)
            skeleton = self.stgc_block1(skeleton, A, None)
            skeleton = self.stgc_block2(skeleton, A, None)
            skeleton = self.stgc_block3(skeleton, A, None)
            skeleton_feat = self.stgc_block4(skeleton, A, None)
        return skeleton_feat
