import torch
import torch.nn as nn


# =============== Spatial Temporal Graph Convolution Block ===============
class Stgc_block(nn.Module):
    def __init__(self, 
                 in_channels,
                 out_channels,
                 stride,
                 s_kernel_size,
                 t_kernel_size,
                 dropout,
                 residual,
                 A_size,
                 bias=True,
                 use_att_A=False,
                 num_att_A=0):
        super().__init__()

        # Spatial Graph Convolution
        if not use_att_A:
            self.sgc = S_GC(in_channels=in_channels,
                            out_channels=out_channels,
                            s_kernel_size=s_kernel_size,
                            bias=bias)
        else:
            self.sgc = S_GC_att_A(in_channels=in_channels,
                                  out_channels=out_channels,
                                  s_kernel_size=s_kernel_size,
                                  num_att_A=num_att_A,
                                  bias=bias)

        # Learnable weight matrix M
        self.M = nn.Parameter(torch.ones(A_size))

        # Temporal Graph Convolution unit
        self.tgc = nn.Sequential(nn.BatchNorm2d(out_channels),
                                 nn.ReLU(),
                                 nn.Conv2d(out_channels,
                                           out_channels,
                                           (t_kernel_size, 1),
                                           (stride, 1),
                                           ((t_kernel_size - 1) // 2, 0),
                                           bias=bias),
                                 nn.BatchNorm2d(out_channels),
                                 nn.Dropout(dropout),
                                 nn.ReLU())

        # Residual
        if not residual:
            self.residual = lambda x: 0

        elif(in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = nn.Sequential(nn.Conv2d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=(stride, 1),
                                                    bias=bias),
                                                    nn.BatchNorm2d(out_channels))

        self.relu = nn.ReLU()

    def forward(self, x, A, att_A):
        #In STAGCN : Stgc_block x torch.Size([256, 2, 150, 18])                                                                                                         
        #In STAGCN : Stgc_block A torch.Size([3, 18, 18])   
      
        sgc_out = self.sgc(x, A * self.M, att_A)
        x = self.tgc(sgc_out) + self.residual(x)
        return x


# =============== Spatial Graph Convolution ===============
class S_GC(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 s_kernel_size,
                 bias):
        super().__init__()

        self.s_kernel_size = s_kernel_size
        # FIXME: change in_channels
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels * s_kernel_size,
                              kernel_size=(1, 1),
                              padding=(0, 0),
                              stride=(1, 1),
                              dilation=(1, 1),
                              bias=bias)

    def forward(self, x, A, att_A):
        #In S_GC 1 : x torch.Size([256, 2, 150, 18])                                                                                                                                                              
        #In S_GC 1 : A torch.Size([3, 18, 18])                                                                                                                                                                    
        #In S_GC 1 : att_A None        

        x = self.conv(x)
        n, kc, t, v = x.size()
        x = x.view(n, self.s_kernel_size, kc//self.s_kernel_size, t, v)
        x = torch.einsum('nkctv,kvw->nctw', (x, A))

        return x.contiguous()


# =============== Spatial Graph Convolution ===============
class S_GC_att_A(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 s_kernel_size,
                 num_att_A,
                 bias):
        super().__init__()

        self.num_att_A = num_att_A
        self.s_kernel_size = s_kernel_size + num_att_A

        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels * self.s_kernel_size,
                              kernel_size=(1, 1),
                              padding=(0, 0),
                              stride=(1, 1),
                              dilation=(1, 1),
                              bias=bias)

    def forward(self, x, A, att_A):
        
        x = self.conv(x)
        
        n, kc, t, v = x.size()
        x = x.view(n, self.s_kernel_size, kc//self.s_kernel_size, t, v)

        x1 = x[:, :self.s_kernel_size-self.num_att_A, :, :, :]
        x2 = x[:, -self.num_att_A:, :, :, :]
        x1 = torch.einsum('nkctv,kvw->nctw', (x1, A))
        x2 = torch.einsum('nkctv,nkvw->nctw', (x2, att_A))
        x_sum = x1 + x2

        return x_sum.contiguous()
