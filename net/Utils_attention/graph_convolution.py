import torch
import torch.nn as nn
import loralib as lora
''' Spatial Temporal Graph Convolution Block '''
class Stgc_block(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 s_kernel_size,
                 t_kernel_size,
                 dropout, residual,
                 A_size,
                 PRETRAIN_SETTING,
                 bias=True,
                 use_att_A=False,
                 num_att_A=0,
                 PRETRAIN = True,
                 lora_config = None):
        super().__init__()
        self.PRETRAIN_SETTING = PRETRAIN_SETTING
        self.PRETRAIN = PRETRAIN
        # Spatial Graph Convolution
        if not use_att_A:
            self.sgc = S_GC(in_channels=in_channels,
                            out_channels=out_channels,
                            s_kernel_size=s_kernel_size,
                            bias=bias,
                            PRETRAIN=self.PRETRAIN,
                            lora_config=lora_config)
        else:
            if self.PRETRAIN_SETTING == 'STAGCN':
                self.sgc = S_GC_att_A(  in_channels=in_channels,
                                        out_channels=out_channels,
                                        s_kernel_size=s_kernel_size,
                                        num_att_A=num_att_A,
                                        bias=bias,
                                        PRETRAIN=self.PRETRAIN,
                                        lora_config=lora_config)

            # PRETRAIN_SETTING : 'Attention':
            else : 
                self.sgc = A_GC(    in_channels=in_channels,
                                    out_channels=out_channels,
                                    s_kernel_size=s_kernel_size,
                                    num_att_A=num_att_A,
                                    bias=bias,
                                    PRETRAIN=self.PRETRAIN,
                                    lora_config=lora_config)

        # Learnable weight matrix M
        self.M = nn.Parameter(torch.ones(A_size))

        # Temporal Graph Convolution unit
        if self.PRETRAIN :
            self.tgc = nn.Sequential(nn.BatchNorm2d(out_channels),
                                    nn.ReLU(),
                                    nn.Conv2d(out_channels, out_channels,
                                            # (temporal kernel size dimension, spatial kernel size dimension)
                                            (t_kernel_size, 1),
                                            # (temporal stride dimension, spatial stride dimension)
                                            (stride, 1),
                                            # (temporal padding kernel size, spatial padding kernel size)
                                            ((t_kernel_size - 1) // 2, 0),
                                            bias=bias),
                                    nn.BatchNorm2d(out_channels),
                                    nn.Dropout(dropout),
                                    nn.ReLU())
        else:
            # Lora Adapter
            self.tgc = nn.Sequential(nn.BatchNorm2d(out_channels),
                                    nn.ReLU(),
                                    lora.Conv2d(in_channels = out_channels,
                                                out_channels = out_channels,
                                                # (temporal kernel size dimension, spatial kernel size dimension)
                                                kernel_size = t_kernel_size,
                                                # (temporal stride dimension, spatial stride dimension)
                                                stride = stride,
                                                # (temporal padding kernel size, spatial padding kernel size)
                                                padding = (t_kernel_size - 1) // 2,
                                                r = lora_config["r"], 
                                                lora_alpha =  lora_config["lora_alpha"],
                                                lora_dropout = lora_config["lora_dropout"]),
                                    nn.BatchNorm2d(out_channels),
                                    nn.Dropout(dropout),
                                    nn.ReLU())
        ''' 
            Residual
            If residual is False, then self.residual is set to a lambda function that maps the input x 
            to a zero vector. This means if residual connections are not used, self.residual won't have 
            any effect.

            If @input channels (in_channels) is equal to the @number of output channels (out_channels), 
            and the @stride (stride) is 1, then self.residual is set to a lambda function that directly 
            returns the input x, implying that if the input and output sizes are the same and no 
            downsampling is performed, the residual connection remains identity.

            @kernel_size = 1: 
            This means the kernel size of the convolution is 1, indicating that the width and height in 
            spatial dimensions (typically two-dimensional) are both 1. Such a kernel is referred to as a 
            1x1 convolutional kernel. 
            @stride=(stride, 1): 
            This indicates that the convolution operation has a stride of stride in the spatial dimensions,
            while the stride in the time dimension (if it's a time series data) is 1. 
            Setting the stride this way allows the convolution to move in the spatial dimensions while 
            keeping it stationary in the time dimension,
        '''
        if not residual:
            self.residual = lambda x: 0
        elif(in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            if self.PRETRAIN :
                self.residual = nn.Sequential(nn.Conv2d(in_channels,
                                                        out_channels,
                                                        kernel_size=1,
                                                        stride=(stride, 1),
                                                        bias=bias),
                                                        nn.BatchNorm2d(out_channels))
            else :
                # Lora Adapter
                self.residual = nn.Sequential(lora.Conv2d(in_channels,
                                                        out_channels,
                                                        kernel_size=1,
                                                        stride=(stride, 1),
                                                        r = lora_config["r"], 
                                                        lora_alpha =  lora_config["lora_alpha"],
                                                        lora_dropout = lora_config["lora_dropout"]),
                                                        nn.BatchNorm2d(out_channels))
        self.relu = nn.ReLU()

    def forward(self, x, A, att_A):
        '''
        @x : [batch_size, in_channels, num_frames, num_nodes]
        @A : [multihead_STGCN, num_nodes, num_nodes]
        '''
        sgc_out = self.sgc(x, A * self.M, att_A) # x, A, att_A 
        x0 = self.tgc(sgc_out)
        x = x0 + self.residual(x)
        return x

''' Spatial Graph Convolution '''
class S_GC(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 s_kernel_size,
                 bias,
                 PRETRAIN,
                 lora_config = None):
        super().__init__()

        self.s_kernel_size = s_kernel_size
        if PRETRAIN :
            self.conv = nn.Conv2d(in_channels=in_channels,
                                out_channels=out_channels * s_kernel_size,
                                kernel_size=(1, 1),
                                padding=(0, 0),
                                stride=(1, 1),
                                dilation=(1, 1),
                                bias=bias)
        else :
            # Lora Adapter
            self.conv = lora.Conv2d(in_channels=in_channels,
                                    out_channels=out_channels * s_kernel_size,
                                    kernel_size=(1, 1),
                                    padding=(0, 0),
                                    stride=(1, 1),
                                    dilation=(1, 1),
                                    r = lora_config["r"], 
                                    lora_alpha = lora_config["lora_alpha"],
                                    lora_dropout = lora_config["lora_dropout"])
    def forward(self, x, A, att_A):
        x = self.conv(x)  
        n, kc, t, v = x.size()
        x = x.view(n, self.s_kernel_size, kc//self.s_kernel_size, t, v)
        x = torch.einsum('nkctv,kvw->nctw', (x, A)) # 7 * 22 * 22
        return x.contiguous()
    
''' 
    PRETRAIN_SETTING : 'STAGCN' 
    Spatial Graph and Attention Graph Convolution
'''
class S_GC_att_A(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 s_kernel_size,
                 num_att_A,
                 bias,
                 PRETRAIN,
                 lora_config = None):
        super().__init__()
        self.num_att_A = num_att_A                     
 
        self.s_kernel_size = s_kernel_size + num_att_A  
        if PRETRAIN :
            self.conv = nn.Conv2d(in_channels=in_channels,
                                out_channels=out_channels * self.s_kernel_size,
                                kernel_size=(1, 1),
                                padding=(0, 0),
                                stride=(1, 1),
                                dilation=(1, 1))
        else :
            # Lora Adapter
            self.conv = lora.Conv2d(in_channels=in_channels,
                                    out_channels=out_channels * self.s_kernel_size,
                                    kernel_size=(1, 1),
                                    padding=(0, 0),
                                    stride=(1, 1),
                                    dilation=(1, 1),
                                    r = lora_config["r"], 
                                    lora_alpha = lora_config["lora_alpha"],
                                    lora_dropout = lora_config["lora_dropout"])

    def forward(self, x, A, att_A):
        x = self.conv(x)
        n, kc, t, v = x.size()
        x = x.view(n, self.s_kernel_size, kc//self.s_kernel_size, t, v)
        
        x1 = x[:, :self.s_kernel_size-self.num_att_A, :, :, :]
        x2 = x[:, -self.num_att_A:, :, :, :]
        '''
            x1 : [batchsize, # of multihead_STGCN , channel, sequence length, vertex]
            x2 : [batchsize, # of attention_head, channel, sequence length, vertex]
        '''
        x1 = torch.einsum('nkctv,kvw->nctw', (x1, A))       
        x2 = torch.einsum('nkctv,nkvw->nctw', (x2, att_A))  
        x_sum = x1 + x2
        return x_sum.contiguous()

''' 
    PRETRAIN_SETTING : 'Attention' 
    Attention Graph Convolution
'''
class A_GC(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 s_kernel_size,
                 num_att_A,
                 bias,
                 PRETRAIN,
                 lora_config = None):
        super().__init__()
                                                        
        self.num_att_A = num_att_A                      

        self.s_kernel_size = num_att_A  
        if PRETRAIN :
            self.conv = nn.Conv2d(in_channels=in_channels,
                                out_channels=out_channels * self.s_kernel_size,
                                kernel_size=(1, 1),
                                padding=(0, 0),
                                stride=(1, 1),
                                dilation=(1, 1),
                                bias=bias)
        else :
            # Lora Adapter
            self.conv = lora.Conv2d(in_channels=in_channels,
                                    out_channels=out_channels * self.s_kernel_size,
                                    kernel_size=1,
                                    padding=0,
                                    stride=1,
                                    dilation=1,
                                    r = lora_config["r"], 
                                    lora_alpha = lora_config["lora_alpha"],
                                    lora_dropout = lora_config["lora_dropout"])

    def forward(self, x, A, att_A):
        x = self.conv(x)
        n, kc, t, v = x.size()
        x = x.view(n, self.s_kernel_size, kc//self.s_kernel_size, t, v)
        x = torch.einsum('nkctv,nkvw->nctw', (x, att_A)) 

        return x.contiguous()
