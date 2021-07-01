from addict import Dict
from torch import nn
import torch.nn.functional as F

from detection.model.backbone import build_backbone
from detection.model.neck import build_neck
from detection.model.head import build_head


class Model(nn.Module):
    def __init__(self, model_config: dict):
        super().__init__()
        model_config = Dict(model_config)
        backbone_type = model_config.backbone.pop('type')
        neck_type = model_config.neck.pop('type')
        head_type = model_config.head.pop('type')
        self.backbone = build_backbone(backbone_type, **model_config.backbone)
        self.neck = build_neck(neck_type, in_channels=self.backbone.out_channels, **model_config.neck)
        self.head = build_head(head_type, in_channels=self.neck.out_channels, **model_config.head)
        self.name = f'{backbone_type}_{neck_type}_{head_type}'

    def forward(self, x):
        _, _, H, W = x.size()
        backbone_out = self.backbone(x)
        neck_out = self.neck(backbone_out)
        y = self.head(neck_out)
        y = F.interpolate(y, size=(H, W), mode='bilinear', align_corners=True)
        return y
