from .base_bev_backbone import BaseBEVBackbone, BaseBEVBackboneV1, BaseBEVResBackbone
from .mobile_bev_backbone import MobileBEVBackbone  # ✨ 新增导入

__all__ = {
    'BaseBEVBackbone': BaseBEVBackbone,
    'BaseBEVBackboneV1': BaseBEVBackboneV1,
    'BaseBEVResBackbone': BaseBEVResBackbone,
    'MobileBEVBackbone': MobileBEVBackbone # ✨ 新增注册
}
