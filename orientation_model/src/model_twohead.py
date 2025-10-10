# src/model_twohead.py
from __future__ import annotations
import torch
from torch import nn
from torchvision import models

class TwoHeadMobileNet(nn.Module):
    """
    MobileNetV3-Small backbone with a classifier projection (576→1024) and two heads:
      - family:   2-class (portrait vs landscape)
      - polarity: 2-class (up vs down within family)
    """
    def __init__(self, pretrained: bool = True, drop_p: float = 0.2):
        super().__init__()
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        base = models.mobilenet_v3_small(weights=weights)

        self.features = base.features
        self.pool = nn.AdaptiveAvgPool2d(1)

        # Projection head (mirror torchvision: Linear→Hardswish→Dropout)
        in_feat = base.classifier[0].in_features   # 576
        hid = base.classifier[0].out_features      # 1024
        self.proj = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(in_feat, hid),
            nn.Hardswish(),
            nn.Dropout(p=drop_p),
        )

        self.head_family   = nn.Linear(hid, 2)   # 0: portrait, 1: landscape
        self.head_polarity = nn.Linear(hid, 2)   # 0: up, 1: down

    def forward(self, x: torch.Tensor):
        x = self.features(x)
        x = self.pool(x)
        x = self.proj(x)                         # [B,1024]
        fam_logits = self.head_family(x)         # [B,2]
        pol_logits = self.head_polarity(x)       # [B,2]
        return fam_logits, pol_logits

    @staticmethod
    def combine_to_fourway(fam_logits: torch.Tensor, pol_logits: torch.Tensor) -> torch.Tensor:
        """
        Return 4-class LOG-PROBS in fixed order [0, 90, 180, 270].
          fam: 0=portrait, 1=landscape
          pol: 0=up,       1=down
        """
        fam_lp = torch.log_softmax(fam_logits, dim=1)
        pol_lp = torch.log_softmax(pol_logits, dim=1)
        PORTRAIT, LANDSCAPE = 0, 1
        UP, DOWN = 0, 1
        b = fam_lp.size(0)
        four_lp = torch.empty(b, 4, device=fam_lp.device, dtype=fam_lp.dtype)
        four_lp[:, 0] = fam_lp[:, PORTRAIT]  + pol_lp[:, UP]    # 0
        four_lp[:, 2] = fam_lp[:, PORTRAIT]  + pol_lp[:, DOWN]  # 180
        four_lp[:, 1] = fam_lp[:, LANDSCAPE] + pol_lp[:, UP]    # 90
        four_lp[:, 3] = fam_lp[:, LANDSCAPE] + pol_lp[:, DOWN]  # 270
        return four_lp


