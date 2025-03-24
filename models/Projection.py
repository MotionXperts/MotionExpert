import torch.nn.functional as F
import torch.nn as nn
class Transformation(nn.Module) :
    def __init__(self, cfg, in_channel, t5_channel) :
        super().__init__()
        self.fc_layer = []
        drop_rate = 0.1
        self.change_layer = []
        self.cfg = cfg

        self.change_layer.append(nn.Dropout(drop_rate))
        self.change_layer.append(nn.Linear(in_channel, 512))
        self.change_layer.append(nn.BatchNorm1d(512))
        self.change_layer.append(nn.ReLU(True))
        in_channel = 512
        self.change_layer = nn.Sequential(*self.change_layer)

        for _ in range(2) :
            self.fc_layer.append(nn.Dropout(drop_rate))
            self.fc_layer.append(nn.Linear(in_channel, 512))
            self.fc_layer.append(nn.BatchNorm1d(512))
            self.fc_layer.append(nn.ReLU(True))
            in_channel = 512
        self.fc_layer = nn.Sequential(*self.fc_layer)

        self.t5_channel = t5_channel
        self.video_emb = nn.Linear(512, self.t5_channel)

    def forward(self, x) :
        B, T, V, C = x.size()
        # Either aggregate time and skeleton dimension, avg pool skeleton dimension, or
        # max pool time dimension
        if self.cfg.TRANSFORMATION.REDUCTION_POLICY == 'TIME_POOL' :
            x = x.permute(0, 2, 1, 3)
            x = F.max_pool2d(x, (x.size(2), 1)).squeeze(2)
        elif self.cfg.TRANSFORMATION.REDUCTION_POLICY == 'SKELETON_POOL' :
            # First convert node dimension to the third dimension so that pool2d can
            # work.
            x = F.max_pool2d(x, (x.size(2), 1)).squeeze(2)
        x = x.reshape(-1, C)
        x = self.change_layer(x)
        x = self.fc_layer(x)
        x = self.video_emb(x)
        x = x.reshape(B, -1, self.t5_channel)
        return x
