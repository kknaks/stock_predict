"""CNN Base Learner — 1D Multi-Scale Convolution (분리 학습용).

다중 커널 크기(3, 5, 7)의 1D Convolution으로
서로 다른 시간 스케일의 패턴을 캡처한다.

Layer 구조 (kernel_sizes=[3,5,7], num_filters=64):
    transpose → 3 Branches 병렬:
      Conv1d(33,64,k=3) → BN → ReLU → masked max pool → (batch, 64)
      Conv1d(33,64,k=5) → BN → ReLU → masked max pool → (batch, 64)
      Conv1d(33,64,k=7) → BN → ReLU → masked max pool → (batch, 64)
    → Concat (batch, 192)
    → Linear(192→64) → ReLU → Dropout
    → Output: (batch, 64)
"""

import torch
import torch.nn as nn
from omegaconf import DictConfig

from .base import BaseModel, ModeType


class CNNPredictor(BaseModel):
    """1D Multi-Scale CNN encoder + fusion + mode별 head.

    Args:
        input_size: 시계열 피처 수 (33).
        cfg: model config (kernel_sizes, num_filters, dropout, fusion_dim).
        mode: 출력 모드 ('cls', 'high', 'low', 'all').
    """

    PROJECTION_DIM = 64

    def __init__(self, input_size: int, cfg: DictConfig, mode: ModeType = "all"):
        super().__init__(self.PROJECTION_DIM, cfg, mode)

        num_filters = cfg.num_filters
        kernel_sizes = list(cfg.kernel_sizes)

        # Multi-scale conv branches
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for k in kernel_sizes:
            pad = k // 2
            self.convs.append(
                nn.Conv1d(input_size, num_filters, kernel_size=k, padding=pad)
            )
            self.bns.append(nn.BatchNorm1d(num_filters))

        # Projection: (num_filters * n_branches) → PROJECTION_DIM
        self.projection = nn.Sequential(
            nn.Linear(num_filters * len(kernel_sizes), self.PROJECTION_DIM),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
        )

    def encode(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """(batch, seq_len, input_size) → (batch, 64)"""
        # (batch, seq_len, C) → (batch, C, seq_len)
        x_t = x.transpose(1, 2)
        seq_len = x.size(1)

        # 패딩 마스크: (batch, 1, seq_len) — conv 출력과 broadcasting
        mask = torch.arange(seq_len, device=x.device).unsqueeze(0) < lengths.unsqueeze(1)
        mask_1d = mask.unsqueeze(1)  # (batch, 1, seq_len)

        pooled_list = []
        for conv, bn in zip(self.convs, self.bns):
            h = conv(x_t)  # (batch, num_filters, seq_len)
            h = bn(h)
            h = torch.relu(h)
            # 패딩 위치를 -inf로 설정하여 max pool에서 무시
            h = h.masked_fill(~mask_1d, float("-inf"))
            pooled = h.max(dim=2)[0]  # (batch, num_filters)
            pooled_list.append(pooled)

        # (batch, num_filters * n_branches)
        concat = torch.cat(pooled_list, dim=1)
        return self.projection(concat)
