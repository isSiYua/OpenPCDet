import numpy as np
import torch
import torch.nn as nn


class InvertedResidualLite(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, expand_ratio=2, act_layer=nn.SiLU):
        super().__init__()
        assert stride in [1, 2]

        hidden_dim = int(round(in_channels * expand_ratio))
        self.use_res_connect = (stride == 1 and in_channels == out_channels)

        layers = []

        # 1x1 expand
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(hidden_dim, eps=1e-3, momentum=0.01),
                act_layer()
            ])
        else:
            hidden_dim = in_channels

        # 3x3 depthwise
        layers.extend([
            nn.Conv2d(
                hidden_dim, hidden_dim,
                kernel_size=3, stride=stride, padding=1,
                groups=hidden_dim, bias=False
            ),
            nn.BatchNorm2d(hidden_dim, eps=1e-3, momentum=0.01),
            act_layer()
        ])

        # 1x1 project
        layers.extend([
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.01)
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        if self.use_res_connect:
            out = x + out
        return out


class MobileBEVBackboneV2(nn.Module):
    """
    A conservative MobileNet-style BEV backbone:
    - keep the first block of each stage as standard 3x3 conv
    - replace following repeated conv blocks with lightweight inverted residual blocks
    - keep output channels / deblocks unchanged for fair comparison
    """
    def __init__(self, model_cfg, input_channels):
        super().__init__()
        self.model_cfg = model_cfg

        layer_nums = self.model_cfg.LAYER_NUMS
        layer_strides = self.model_cfg.LAYER_STRIDES
        num_filters = self.model_cfg.NUM_FILTERS

        upsample_strides = self.model_cfg.UPSAMPLE_STRIDES
        num_upsample_filters = self.model_cfg.NUM_UPSAMPLE_FILTERS

        expand_ratio = self.model_cfg.get('EXPAND_RATIO', 2)
        act_name = self.model_cfg.get('ACT_FN', 'SiLU')
        act_layer = nn.SiLU if act_name.lower() == 'silu' else nn.ReLU

        num_levels = len(layer_nums)
        c_in_list = [input_channels, *num_filters[:-1]]

        self.blocks = nn.ModuleList()
        self.deblocks = nn.ModuleList()

        for idx in range(num_levels):
            cur_layers = []

            # First block in each stage: keep standard conv
            cur_layers.extend([
                nn.ZeroPad2d(1),
                nn.Conv2d(
                    c_in_list[idx], num_filters[idx],
                    kernel_size=3,
                    stride=layer_strides[idx],
                    padding=0,
                    bias=False
                ),
                nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                act_layer()
            ])

            # Replace repeated blocks with lightweight blocks
            for _ in range(layer_nums[idx]):
                cur_layers.append(
                    InvertedResidualLite(
                        in_channels=num_filters[idx],
                        out_channels=num_filters[idx],
                        stride=1,
                        expand_ratio=expand_ratio,
                        act_layer=act_layer
                    )
                )

            self.blocks.append(nn.Sequential(*cur_layers))

            stride = upsample_strides[idx]
            if stride >= 1:
                self.deblocks.append(nn.Sequential(
                    nn.ConvTranspose2d(
                        num_filters[idx], num_upsample_filters[idx],
                        upsample_strides[idx],
                        stride=upsample_strides[idx],
                        bias=False
                    ),
                    nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                    act_layer()
                ))
            else:
                stride = np.round(1 / stride).astype(np.int32)
                self.deblocks.append(nn.Sequential(
                    nn.Conv2d(
                        num_filters[idx], num_upsample_filters[idx],
                        stride, stride=stride, bias=False
                    ),
                    nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                    act_layer()
                ))

        c_in = sum(num_upsample_filters)
        if len(upsample_strides) > num_levels:
            self.deblocks.append(nn.Sequential(
                nn.ConvTranspose2d(
                    c_in, c_in,
                    upsample_strides[-1],
                    stride=upsample_strides[-1],
                    bias=False
                ),
                nn.BatchNorm2d(c_in, eps=1e-3, momentum=0.01),
                act_layer()
            ))

        self.num_bev_features = c_in

    def forward(self, data_dict):
        spatial_features = data_dict['spatial_features']
        ups = []
        ret_dict = {}
        x = spatial_features

        for i in range(len(self.blocks)):
            x = self.blocks[i](x)

            stride = int(spatial_features.shape[2] / x.shape[2])
            ret_dict[f'spatial_features_{stride}x'] = x

            ups.append(self.deblocks[i](x))

        if len(ups) > 1:
            x = torch.cat(ups, dim=1)
        else:
            x = ups[0]

        if len(self.deblocks) > len(self.blocks):
            x = self.deblocks[-1](x)

        data_dict['spatial_features_2d'] = x
        return data_dict