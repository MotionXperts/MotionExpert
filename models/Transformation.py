import torch.nn.functional as F
import torch.nn as nn
import loralib as lora
class Transformation(nn.Module):
    def __init__(self,cfg,in_channel,t5_channel):
        super().__init__()
        self.fc_layer = []
        drop_rate = .1
        self.change_layer = []
        self.cfg =cfg
        
        for _ in range(1):
            self.change_layer.append(nn.Dropout(drop_rate))
            ## Linear Layer
            if cfg.TASK.PRETRAIN:
                self.change_layer.append(nn.Linear(in_channel,512))
            else:
                self.change_layer.append(lora.Linear(in_channel,512, r = 32, lora_alpha = 64, lora_dropout = 0.1))

            self.change_layer.append(nn.BatchNorm1d(512))
            self.change_layer.append(nn.ReLU(True))
            in_channel = 512
        self.change_layer = nn.Sequential(*self.change_layer)

        for _ in range(2):
            self.fc_layer.append(nn.Dropout(drop_rate))
            ## Linear Layer
            if cfg.TASK.PRETRAIN:
                self.fc_layer.append(nn.Linear(in_channel,512))
            else:
                self.fc_layer.append(lora.Linear(in_channel,512, r = 32, lora_alpha = 64, lora_dropout = 0.1))

            self.fc_layer.append(nn.BatchNorm1d(512))
            self.fc_layer.append(nn.ReLU(True))
            in_channel = 512
        self.fc_layer = nn.Sequential(*self.fc_layer)

        self.t5_channel = t5_channel
        ## Linear Layer
        if cfg.TASK.PRETRAIN:
            self.video_emb= nn.Linear(512,self.t5_channel)
        else:
            self.video_emb= lora.Linear(512,self.t5_channel, r = 32, lora_alpha = 64, lora_dropout = 0.1)

    def forward(self,x):
        B,T,V,C = x.size()
        ## Either aggregate time and skeleton dimension, avg pool skeleton dimension, or max pool time dimension
        if   self.cfg.TRANSFORMATION.REDUCTION_POLICY == 'TIME_POOL': ## Cindy's method
            x = x.permute(0,2,1,3)
            x = F.max_pool2d(x,(x.size(2),1)).squeeze(2)
        elif self.cfg.TRANSFORMATION.REDUCTION_POLICY == 'SKELETON_POOL':
            ## first convert node dimension to the third dim so that pool2d can work on
            x = F.max_pool2d(x,(x.size(2),1)).squeeze(2)
        x = x.reshape(-1,C)
        x = self.change_layer(x)
        x = self.fc_layer(x)
        x = self.video_emb(x)
        x = x.reshape(B,-1,self.t5_channel)
        return x
