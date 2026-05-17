import torch
from thop import profile
from pcdet.models.backbones_2d.base_bev_backbone import BaseBEVBackbone
from pcdet.models.backbones_2d.mobile_bev_backbone import MobileBEVBackbone
import yaml
from easydict import EasyDict

# 1. 构造一个假的 BEV 特征图输入 [Batch, Channels, H, W]
# KITTI PointPillars 默认的 BEV 尺寸大概是 496x432，64通道
dummy_input = torch.randn(1, 64, 496, 432).cuda()
data_dict = {'spatial_features': dummy_input}

# 2. 测算原版 V7 Backbone
class DummyModelBase(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 填入原版的 yaml 配置
        cfg = EasyDict({'LAYER_NUMS': [3, 5, 5], 'LAYER_STRIDES': [2, 2, 2], 
                        'NUM_FILTERS': [64, 128, 256], 'UPSAMPLE_STRIDES': [1, 2, 4], 
                        'NUM_UPSAMPLE_FILTERS': [128, 128, 128]})
        self.net = BaseBEVBackbone(cfg, input_channels=64).cuda()
    def forward(self, x):
        return self.net({'spatial_features': x})['spatial_features_2d']

# 3. 测算轻量化 Mobile Backbone
class DummyModelMobile(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 还原成你实际 YAML 里的满血通道数！
        cfg = EasyDict({
            'LAYER_NUMS': [3, 5, 5], 
            'LAYER_STRIDES': [2, 2, 2], 
            'NUM_FILTERS': [64, 128, 256],      # ✨ 恢复满血！
            'UPSAMPLE_STRIDES': [1, 2, 4], 
            'NUM_UPSAMPLE_FILTERS': [128, 128, 128] # ✨ 恢复满血！
        })
        self.net = MobileBEVBackbone(cfg, input_channels=64).cuda()
        self.net.eval()
    def forward(self, x):
        return self.net({'spatial_features': x})['spatial_features_2d']

if __name__ == '__main__':
    # 需要先 pip install thop
    macs_base, params_base = profile(DummyModelBase(), inputs=(dummy_input, ))
    macs_mobile, params_mobile = profile(DummyModelMobile(), inputs=(dummy_input, ))
    
    print(f"原版 BaseBEV:   FLOPs: {macs_base/1e9:.2f} G, Params: {params_base/1e6:.2f} M")
    print(f"轻量 MobileBEV: FLOPs: {macs_mobile/1e9:.2f} G, Params: {params_mobile/1e6:.2f} M")