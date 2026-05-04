import numpy as np
import torch
import torch.nn as nn

class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, expand_ratio=2): # 伪图像较稀疏，expand_ratio 设为 2 比较稳妥
        super().__init__()
        hidden_dim = int(round(in_channels * expand_ratio))
        self.use_res_connect = stride == 1 and in_channels == out_channels

        layers = []
        if expand_ratio != 1:
            # 1. Pointwise Conv (Expand)
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(hidden_dim, eps=1e-3, momentum=0.01),
                nn.SiLU()  # ✨ 保持你之前验证过的绝佳激活函数
            ])
            
        # 2. Depthwise Conv
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1, 
                      groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim, eps=1e-3, momentum=0.01),
            nn.SiLU()
        ])
        
        # 3. Pointwise Conv (Project) -> 注意这里没有激活函数，这是 MobileNetV2 的核心思想 Linear Bottleneck
        layers.extend([
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.01)
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileBEVBackbone(nn.Module):
    def __init__(self, model_cfg, input_channels):
        super().__init__()
        self.model_cfg = model_cfg

        layer_nums = self.model_cfg.LAYER_NUMS
        layer_strides = self.model_cfg.LAYER_STRIDES
        num_filters = self.model_cfg.NUM_FILTERS
        num_upsample_filters = self.model_cfg.NUM_UPSAMPLE_FILTERS
        upsample_strides = self.model_cfg.UPSAMPLE_STRIDES

        num_levels = len(layer_nums)
        c_in_list = [input_channels, *num_filters[:-1]]
        
        self.blocks = nn.ModuleList()
        self.deblocks = nn.ModuleList()
        
        for idx in range(num_levels):
            # Block 的第一层：负责下采样或改变通道数
            cur_layers = [
                InvertedResidual(c_in_list[idx], num_filters[idx], stride=layer_strides[idx], expand_ratio=2)
            ]
            # 后续层：保持尺寸和通道数不变
            for k in range(layer_nums[idx] - 1):
                cur_layers.append(
                    InvertedResidual(num_filters[idx], num_filters[idx], stride=1, expand_ratio=2)
                )
            self.blocks.append(nn.Sequential(*cur_layers))
            
            # FPN 反卷积层 (保持原样，这部分计算量不大，主要用于特征融合)
            if len(upsample_strides) > 0:
                stride = upsample_strides[idx]
                if stride >= 1:
                    self.deblocks.append(nn.Sequential(
                        nn.ConvTranspose2d(
                            num_filters[idx], num_upsample_filters[idx],
                            upsample_strides[idx],
                            stride=upsample_strides[idx], bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.SiLU()
                    ))
                else:
                    stride = np.round(1 / stride).astype(np.int)
                    self.deblocks.append(nn.Sequential(
                        nn.Conv2d(
                            num_filters[idx], num_upsample_filters[idx],
                            stride,
                            stride=stride, bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.SiLU()
                    ))

        c_in = sum(num_upsample_filters)
        self.num_bev_features = c_in

    def forward(self, data_dict):
        spatial_features = data_dict['spatial_features']
        ups = []
        ret_dict = {}
        x = spatial_features
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
            stride = int(spatial_features.shape[2] / x.shape[2])
            ret_dict['spatial_features_%dx' % stride] = x
            ups.append(self.deblocks[i](x))

        x = torch.cat(ups, dim=1)
        data_dict['spatial_features_2d'] = x
        return data_dict