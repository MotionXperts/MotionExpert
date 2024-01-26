from .layer.stgcn import ConvTemporalGraphical
from .utils.graph import Graph
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch import Tensor
import torch.nn.functional as F
from typing import Optional
import numpy as np
import math
from torchcrf import CRF

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=300):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        # output channel is odd
        if d_model%2 != 0:  pe[:, 1::2] = torch.cos(position * div_term)[:,0:-1]
        # output channel is even
        else:               pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # make embeddings relatively larger
        x = x * math.sqrt(self.d_model)
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class STGCN_Transformer(nn.Module):
    def __init__(self,
                 hidden_channel,
                 out_channel,
                 nhead, 
                 num_encoder_layers,
                 dim_feedforward,
                 dropout,
                 batch_first,
                 num_class,
                 use_crf,
                 layer_norm_eps = 1e-5,
                 activation = 'relu',
                 norm_first = False,
                 device=None, dtype=None):
        super(STGCN_Transformer, self).__init__()
        ## ST-GCN's GCN block
        # TODO: out_channel not give
        self.out_channel = out_channel
        self.num_node = 22
        self.self_link = [(i, i) for i in range(self.num_node)]
        self.inward = [(0,  1),  (0, 2), ( 0,  3), (1,  4 ), ( 2, 5 ), ( 3, 6 ), ( 4,  7), ( 5, 8 ), ( 6,  9), ( 7, 10), (8, 11), 
                       (9, 12), (9, 13), ( 9, 14), (12, 15), (13, 16), (14, 17), (16, 18), (17, 19), (18, 20), (19, 21)]
        self.outward = [(j, i) for (i, j) in self.inward]
        self.neighbor = self.inward + self.outward
        graph_args = {
            'num_node': self.num_node,
            'self_link': self.self_link,
            'inward': self.inward,
            'outward': self.outward,
            'neighbor': self.neighbor
        }
        self.graph = Graph(**graph_args)
        self.A = torch.from_numpy(self.graph.A.astype(np.float32))

        # TODO: hidden_channel not give  out_channel not give
        self.gcn1 = ConvTemporalGraphical(in_channels=2, out_channels=hidden_channel, kernel_size=self.A.shape[0])
        self.gcn2 = ConvTemporalGraphical(in_channels=hidden_channel, out_channels=out_channel, kernel_size=self.A.shape[0])
        ########################################################
        self.data_bn = nn.BatchNorm1d(self.num_node * 2)
        self.edge_importance = nn.Parameter(torch.ones(self.A.shape)) # 都給1
        ## Encoder
        # TODO: device, dtype not give
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.d_model = self.out_channel
        # TODO: dropout not give
        self.pos_encoder = PositionalEncoding(d_model=self.d_model, dropout=dropout, max_len=300)
        # TODO: nhead not give, dim_feedforward not give, ,batch_first not give
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model,
                                                   nhead =nhead, 
                                                   dim_feedforward = dim_feedforward, 
                                                   dropout = dropout,
                                                   activation = activation, 
                                                   layer_norm_eps = layer_norm_eps, 
                                                   batch_first = batch_first, 
                                                   **factory_kwargs)
        encoder_norm = nn.LayerNorm(normalized_shape=self.d_model, 
                                    eps=layer_norm_eps, 
                                    **factory_kwargs)
        # TODO: num_encoder_layers not give
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
        # TODO: num_class not give
        self.out = nn.Linear(out_channel, num_class)
        self.num_class = num_class
        self.softmax = nn.Softmax(dim=-1)
        self.use_crf = use_crf
        self.crf = CRF(num_tags=num_class, batch_first=batch_first)

    def _get_embeddings(self, src):
        A = self.A.cuda(src.get_device())
        ## TODO: Batch normalization
        ## src = [batch_size, seq_len, 34] --> [batch_size, 34, seq_len]
        src = src.permute(0, 2, 1)
        src = self.data_bn(src)
        ## src = [batch_size, 34, seq_len] --> [batch_size, seq_len, 34]
        src = src.permute(0, 2, 1)
        ## TODO: ST-GCN embeddings
        N, T, _ = src.size()
        
        # input src = [batch_size, seq_len, 34] --> [batch_size, 2, seq_len, 17]
        src = src.view(N, T, self.num_node, 2).permute(0, 3, 1, 2)
        src, A = self.gcn1(src, A * self.edge_importance)
        src, A = self.gcn2(src, A * self.edge_importance)

        ## output of agcn = [batch_size, out_channel, seq_len, 17] --> [batch_size, seq_len, out_channel]
        # src = src.mean(dim=3)
        src, _ = torch.max(src, 3)
        src = src.permute(0, 2, 1)
        return src
    
    def _extend_cls_token(self, src):
        cls_token = nn.Parameter(torch.zeros(1, 1,34))
        cls_token = cls_token.expand(src.size()[0], -1, -1).to("cuda")
        new_src = torch.cat([cls_token, src], dim = 1)
        return new_src

    def _extend_padding_mask(self, mask):
        n_batches = mask.shape[0]
        return torch.cat([torch.full((n_batches, 1), True).to("cuda"), mask], dim = 1)

    def _get_encoder_feature(self, src, mask):
        src  = self._extend_cls_token(src)
        mask = self._extend_padding_mask(mask)
        src  = self._get_embeddings(src)
        ### True: not attend, False: attend        
        self.src_key_padding_mask = ~mask

        # [batch_size, seq_len, 272]
        src = self.pos_encoder(src)
        # output --> [batch_size, seq_len, 272]
        output = self.encoder(src, mask=None, src_key_padding_mask=self.src_key_padding_mask)
        # output --> [batch_size, seq_len, 32]
        #if not self.use_crf:
        #    # output --> [batch_size * seq_len, 32]
        #    output = output.contiguous().view(N * T, -1)
        # output --> [batch_size, seq_len, 4] or [batch_size * seq_len, 4]
        output = self.out(output)
        return output

    def forward(self, src, mask):
        output = self._get_encoder_feature(src, mask)
        if self.use_crf:
            ### type: List (for prediction)
            return self.crf.decode(output, mask)
        else:
            return self.softmax(output)[:, :1, :].view(-1, self.num_class)
    
    def loss_fn(self, src, target, mask):
        pred = self._get_encoder_feature(src, mask)
        return -self.crf.forward(pred, target, mask, reduction='mean')

if __name__ == '__main__':
    dataset = IceSkatingDataset(pkl_file='/home/lin10/projects/SkatingJumpClassifier/data/loop/alphapose/test.pkl',
                                tag_mapping_file='/home/lin10/projects/SkatingJumpClassifier/data/tag2idx.json',
                                use_crf=True,
                                add_noise=False,
                                subtract_feature=False)
    dataloader = DataLoader(dataset,batch_size=2,
                        shuffle=False, num_workers=1, collate_fn=dataset.collate_fn)

    model = AGCN_Transformer(
                    out_channel = 16,
                    nhead = 2, 
                    num_encoder_layers = 2,
                    dim_feedforward = 128,
                    dropout = 0.1,
                    batch_first = True,
                    num_class = 4,
                    use_crf = True).to('cuda')
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.999))

    for e in range(2):
        for i_batch, sample in enumerate(dataloader):
            keypoints, labels = sample['keypoints'].to('cuda'), sample['output'].to('cuda')
            mask = sample['mask'].to('cuda')
            output = model(keypoints, mask)
            print(output)
            print(model.state_dict())
            exit()
