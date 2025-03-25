import torch, torch.nn as nn, loralib as lora

# Spatial Temporal Graph Convolution Block.
class Stgc_block(nn.Module) :
    def __init__(self, in_channels, out_channels, stride, s_kernel_size, t_kernel_size, dropout,
                 residual, A_size, PRETRAIN_SETTING, bias = True, use_att_A = False, num_att_A = 0,
                 PRETRAIN = True, lora_config = None) :
        super().__init__()
        self.PRETRAIN_SETTING = PRETRAIN_SETTING
        self.PRETRAIN = PRETRAIN
        self.use_att_A = use_att_A
        # Spatial Graph Convolution.
        if not use_att_A :
            # S_GC Block do convolution on spatial graph.
            self.sgc = S_GC(in_channels = in_channels,
                            out_channels = out_channels,
                            s_kernel_size = s_kernel_size,
                            bias = bias,
                            PRETRAIN = self.PRETRAIN,
                            lora_config = lora_config)
        else :
            # STA-GCN.
            if self.PRETRAIN_SETTING == 'STAGCN' :
                # S_GC_att_A do convolution on attention graph that is generated from attention branch
                # of STA-GCN. This Block will be employed in perception branch of STA-GCN.
                self.sgc = S_GC_att_A(in_channels = in_channels,
                                      out_channels = out_channels,
                                      s_kernel_size = s_kernel_size,
                                      num_att_A = num_att_A,
                                      bias = bias,
                                      PRETRAIN = self.PRETRAIN,
                                      lora_config = lora_config)
            # Human Pose Perception.
            else :
                # A_GC Block do convolution on attention graph that is generated from Pose Extraction.
                # This Block will be employed in Pose Attention.
                self.sgc = A_GC(in_channels = in_channels,
                                out_channels = out_channels,
                                s_kernel_size = s_kernel_size,
                                num_att_A = num_att_A,
                                bias = bias,
                                PRETRAIN = self.PRETRAIN,
                                lora_config = lora_config)
        # Learnable weight matrix M.
        self.M = nn.Parameter(torch.ones(A_size))
        # Temporal Graph Convolution.
        self.tgc = nn.Sequential(nn.BatchNorm2d(out_channels),
                                 nn.ReLU(),
                                 nn.Conv2d(out_channels,
                                           out_channels,
                                           # (temporal kernel size dimension, spatial kernel size dimension)
                                           (t_kernel_size, 1),
                                           # (temporal stride dimension, spatial stride dimension)
                                           (stride, 1),
                                           # (temporal padding kernel size, spatial padding kernel size)
                                           ((t_kernel_size - 1) // 2, 0),
                                           bias = bias),
                                 nn.BatchNorm2d(out_channels),
                                 nn.Dropout(dropout),
                                 nn.ReLU())

        # If residual is False, then self.residual is set to a lambda function that maps the input x
        # to a zero vector. This means if residual connections are not used, self.residual won't have
        # any effect.

        # If input channels is equal to the number of output channels and the stride is set to 1, then
        # self.residual is set to a lambda function that directly. Finally it returns the input x,
        # which implying that if the input and output sizes are the same and no downsampling is 
        # performed, the residual connection remains identity.

        # kernel_size is set to 1. This means the kernel size of the convolution is 1, indicating that
        # the width and height in spatial dimensions (typically two-dimensional) are both 1. Such a 
        # kernel is referred to as a 1x1 convolutional kernel.
        # The stride is set to (stride, 1). This indicates that the convolution operation has a stride
        # of stride in the spatial dimensions, while the stride in the time dimension (if it's a time
        # series data) is 1. Setting the stride this way allows the convolution to move in the spatial
        # dimensions while keeping it stationary in the time dimension,

        if not residual :
            self.residual = lambda x : 0
        elif(in_channels == out_channels) and (stride == 1) :
            self.residual = lambda x : x
        else :
            if self.PRETRAIN :
                self.residual = nn.Sequential(nn.Conv2d(in_channels,
                                                        out_channels,
                                                        kernel_size = 1,
                                                        stride = (stride, 1),
                                                        bias = bias),
                                              nn.BatchNorm2d(out_channels))
            else :
                # Lora Adapter
                self.residual = nn.Sequential(lora.Conv2d(in_channels,
                                                          out_channels,
                                                          kernel_size = 1,
                                                          stride = (stride, 1),
                                                          r = lora_config["r"],
                                                          lora_alpha = lora_config["lora_alpha"],
                                                          lora_dropout = lora_config["lora_dropout"]),
                                              nn.BatchNorm2d(out_channels))
        self.relu = nn.ReLU()
    def forward(self, x, A, att_A) :
        # The dimension of x is [batch_size, in_channels, num_frames, num_nodes].
        # The demension of A is [number of graphs, num_nodes, num_nodes].
        sgc_out = self.sgc(x, A * self.M, att_A)
        x0 = self.tgc(sgc_out)
        x = x0 + self.residual(x)
        return x

# Spatial Graph Convolution.
class S_GC(nn.Module) :
    def __init__(self, in_channels, out_channels, s_kernel_size, bias, PRETRAIN, lora_config = None) :
        super().__init__()
        self.s_kernel_size = s_kernel_size
        if PRETRAIN :
            self.conv = nn.Conv2d(in_channels = in_channels,
                                  out_channels = out_channels * s_kernel_size,
                                  kernel_size = (1, 1),
                                  padding = (0, 0),
                                  stride = (1, 1),
                                  dilation = (1, 1),
                                  bias = bias)
        else :
            # Lora Adapter
            self.conv = lora.Conv2d(in_channels = in_channels,
                                    out_channels = out_channels * s_kernel_size,
                                    kernel_size = (1, 1),
                                    padding = (0, 0),
                                    stride = (1, 1),
                                    dilation = (1, 1),
                                    r = lora_config["r"],
                                    lora_alpha = lora_config["lora_alpha"],
                                    lora_dropout = lora_config["lora_dropout"])
    def forward(self, x, A, att_A) :
        x = self.conv(x)  
        n, kc, t, v = x.size()
        x = x.view(n, self.s_kernel_size, kc//self.s_kernel_size, t, v)
        # The demension of A is [number of spatial graphs (7), number of joints (22), number of joints (22)].
        x = torch.einsum('nkctv,kvw->nctw', (x, A))
        return x.contiguous()

# STA-GCN.
class S_GC_att_A(nn.Module) :
    def __init__(self, in_channels, out_channels, s_kernel_size, num_att_A, bias, PRETRAIN, lora_config = None) :
        super().__init__()
        self.num_att_A = num_att_A
        # Apply both spatial graph and attention graph on convolution of perception branch of STA-GCN.
        self.s_kernel_size = s_kernel_size + num_att_A
        if PRETRAIN :
            self.conv = nn.Conv2d(in_channels = in_channels,
                                  out_channels = out_channels * self.s_kernel_size,
                                  kernel_size = (1, 1),
                                  padding = (0, 0),
                                  stride = (1, 1),
                                  dilation = (1, 1))
        else :
            # Lora Adapter
            self.conv = lora.Conv2d(in_channels = in_channels,
                                    out_channels = out_channels * self.s_kernel_size,
                                    kernel_size = (1, 1),
                                    padding = (0, 0),
                                    stride = (1, 1),
                                    dilation = (1, 1),
                                    r = lora_config["r"],
                                    lora_alpha = lora_config["lora_alpha"],
                                    lora_dropout = lora_config["lora_dropout"])
    def forward(self, x, A, att_A) :
        x = self.conv(x)
        n, kc, t, v = x.size()
        x = x.view(n, self.s_kernel_size, kc // self.s_kernel_size, t, v)
        # The dimension of x1 is [batchsize, number of spatial graph(7), channel, sequence length, vertex].
        x1 = x[:, : self.s_kernel_size - self.num_att_A, :, :, :]
        # The dimension of x2 is [batchsize, number of attention graph(4), channel, sequence length, vertex].
        x2 = x[:, -self.num_att_A :, :, :, :]
        x1 = torch.einsum('nkctv,kvw->nctw', (x1, A))
        x2 = torch.einsum('nkctv,nkvw->nctw', (x2, att_A))
        x_sum = x1 + x2
        return x_sum.contiguous()

# Human Pose Perception.
class A_GC(nn.Module) :
    def __init__(self, in_channels, out_channels, s_kernel_size, num_att_A, bias, PRETRAIN, lora_config = None) :
        super().__init__()
        self.num_att_A = num_att_A
        # Apply only attention graph on convolution of Pose Attention of Human Pose Perception.
        self.s_kernel_size = num_att_A
        if PRETRAIN :
            self.conv = nn.Conv2d(in_channels = in_channels,
                                  out_channels = out_channels * self.s_kernel_size,
                                  kernel_size = (1, 1),
                                  padding = (0, 0),
                                  stride = (1, 1),
                                  dilation = (1, 1),
                                  bias = bias)
        else :
            # Lora Adapter
            self.conv = lora.Conv2d(in_channels = in_channels,
                                    out_channels = out_channels * self.s_kernel_size,
                                    kernel_size = 1,
                                    padding = 0,
                                    stride = 1,
                                    dilation = 1,
                                    r = lora_config["r"], 
                                    lora_alpha = lora_config["lora_alpha"],
                                    lora_dropout = lora_config["lora_dropout"])
    def forward(self, x, A, att_A) :
        x = self.conv(x)
        n, kc, t, v = x.size()
        x = x.view(n, self.s_kernel_size, kc // self.s_kernel_size, t, v)
        x = torch.einsum('nkctv,nkvw->nctw', (x, att_A))
        return x.contiguous()