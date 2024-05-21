import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class dim_conv(nn.Module):

    def __init__(self, alignment=True):
        super().__init__()
        self.alig = alignment
        self.embedding = nn.Sequential(
            nn.Linear(512, 768),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        
    '''
    # @ x : input tensor
    # x is the output embedding of alignment network 
    # dimension is [ batchsize, vertex(22), seq_length (T), channel (512)] 
    # @ return : batchsize, vertex(22), 768
    '''
    def forward(self, x):
        '''   
        # if alignment the input x is [ batchsize, vertex ,seq_length, channel]
        # if not alignment the input x is [ batchsize, seq_length, vertex, channel]
        '''
        if self.alig == False:
            x = x.permute(0, 3, 1, 2)

        # use kernel model of size (seq_length, 1) to get the global feature
        x = F.max_pool2d(x, (x.size(2), 1)).squeeze(2)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        output_embedding = self.embedding(x.to(device))

        return output_embedding