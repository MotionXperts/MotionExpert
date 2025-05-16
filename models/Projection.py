import torch.nn.functional as F
import torch.nn as nn
import loralib as lora
class Projection(nn.Module) :
    def __init__(self, pretrain, proj_strategy, in_channel, t5_channel, lora_config = None) :
        super().__init__()
        self.fc_layer = []
        drop_rate = 0.1
        self.change_layer = []
        self.proj_strategy = proj_strategy

        for _ in range(1) :
            self.change_layer.append(nn.Dropout(drop_rate))
            # Linear Layer
            if pretrain :
                self.change_layer.append(nn.Linear(in_channel, 512))
            else:
                self.change_layer.append(lora.Linear(in_channel,
                                                     512,
                                                     r = lora_config["r"],
                                                     lora_alpha = lora_config["lora_alpha"],
                                                     lora_dropout = lora_config["lora_dropout"]))

            self.change_layer.append(nn.BatchNorm1d(512))
            self.change_layer.append(nn.ReLU(True))
            in_channel = 512
        self.change_layer = nn.Sequential(*self.change_layer)

        for _ in range(2) :
            self.fc_layer.append(nn.Dropout(drop_rate))
            # Linear Layer
            if pretrain :
                self.fc_layer.append(nn.Linear(in_channel, 512))
            else :
                self.fc_layer.append(lora.Linear(in_channel,
                                                 512,
                                                 r = lora_config["r"],
                                                 lora_alpha = lora_config["lora_alpha"],
                                                 lora_dropout = lora_config["lora_dropout"]))

            self.fc_layer.append(nn.BatchNorm1d(512))
            self.fc_layer.append(nn.ReLU(True))
            in_channel = 512
        self.fc_layer = nn.Sequential(*self.fc_layer)

        self.t5_channel = t5_channel
        # Linear Layer
        if pretrain :
            self.video_emb = nn.Linear(512, self.t5_channel)
        else :
            self.video_emb = lora.Linear(512,
                                         self.t5_channel,
                                         r = lora_config["r"],
                                         lora_alpha = lora_config["lora_alpha"],
                                         lora_dropout = lora_config["lora_dropout"])

    def forward(self, x) :
        B, T, V, C = x.size()
        # If the strategy is to aggregate along the time dimension, apply max pooling over time.
        # If the strategy is to aggregate along the skeleton dimension, apply average pooling over skeletons.
        if self.proj_strategy == 'TIME_POOL' :
            x = x.permute(0, 2, 1, 3)
            x_ = F.avg_pool2d(x, kernel_size = (1, x.size(3)))
            x_, max_indices = F.max_pool2d(x_, kernel_size = (x_.size(2), 1), return_indices = True)
            x = F.max_pool2d(x, (x.size(2), 1))
            x = x.squeeze(2)
        elif self.proj_strategy == 'SKELETON_POOL' :
            # Convert node dimension to the third dimension so that pool2d operation can work.
            x, max_indices = F.max_pool2d(x, (x.size(2), 1), return_indices = True)
            x = x.squeeze(2)

        x = x.reshape(-1, C)
        x = self.change_layer(x)
        x = self.fc_layer(x)
        x = self.video_emb(x)
        x = x.reshape(B, -1, self.t5_channel)
        return x, max_indices
