from .base_bev_backbone import BaseBEVBackbone, BaseBEVBackboneV1, BaseBEVResBackbone
from .mobile_bev_backbone import MobileBEVBackbone  # ✨ 新增导入
from .mobile_bev_backbone_v2 import MobileBEVBackboneV2 # mobilenet v2

__all__ = {
    'BaseBEVBackbone': BaseBEVBackbone,
    'BaseBEVBackboneV1': BaseBEVBackboneV1,
    'BaseBEVResBackbone': BaseBEVResBackbone,
    'MobileBEVBackboneV2': MobileBEVBackboneV2,
    'MobileBEVBackbone': MobileBEVBackbone # ✨ 新增注册
}
