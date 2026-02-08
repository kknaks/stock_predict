"""MLP Base Learner — Aggregation 기반 (분리 학습용).

시계열을 통계량(mean, std, last, min, max)으로 요약한 후
MLP로 처리. 시간 순서를 무시하는 관점을 제공하여
앙상블 다양성을 확보한다.

Layer 구조 (hidden_dims=[128, 64]):
    Aggregation (5 stats × 33 feat) → (batch, 165)
    + normalized_seq_length         → (batch, 166)
    → Linear(166→128) → ReLU → Dropout
    → Linear(128→64)  → ReLU → Dropout
    → Output: (batch, 64)
"""

import torch
import torch.nn as nn
from omegaconf import DictConfig

from .base import BaseModel, ModeType


class MLPPredictor(BaseModel):
    """Aggregation 기반 MLP encoder + fusion + mode별 head.

    Args:
        input_size: 시계열 피처 수 (33).
        cfg: model config (hidden_dims, dropout, fusion_dim).
        mode: 출력 모드 ('cls', 'high', 'low', 'all').
    """

    def __init__(self, input_size: int, cfg: DictConfig, mode: ModeType = "all"):
        n_aggregations = 5  # mean, std, last, min, max
        agg_dim = input_size * n_aggregations + 1  # +1 for normalized seq length
        encoder_dim = cfg.hidden_dims[-1]
        super().__init__(encoder_dim, cfg, mode)

        layers: list[nn.Module] = []
        in_dim = agg_dim
        for h_dim in cfg.hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.ReLU(),
                nn.Dropout(cfg.dropout),
            ])
            in_dim = h_dim
        self.encoder = nn.Sequential(*layers)

    def encode(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """(batch, seq_len, input_size) → (batch, hidden_dims[-1])"""
        batch_size, seq_len, _ = x.shape

        # 유효 마스크: (batch, seq_len, 1)
        mask = torch.arange(seq_len, device=x.device).unsqueeze(0) < lengths.unsqueeze(1)
        mask_expanded = mask.unsqueeze(2)

        # lengths를 float로, 0 방지
        lengths_f = lengths.unsqueeze(1).float().clamp(min=1)

        # mean: 유효 구간 평균
        x_masked = x * mask_expanded
        feat_mean = x_masked.sum(dim=1) / lengths_f  # (batch, n_feat)

        # std: 유효 구간 표준편차
        x_centered = (x - feat_mean.unsqueeze(1)) * mask_expanded
        feat_std = (x_centered.pow(2).sum(dim=1) / lengths_f).sqrt()  # (batch, n_feat)

        # last: 마지막 유효 타임스텝
        last_idx = (lengths - 1).long()
        feat_last = x[torch.arange(batch_size, device=x.device), last_idx]  # (batch, n_feat)

        # min: 유효 구간 최솟값 (패딩을 inf로 마스킹)
        x_for_min = x.masked_fill(~mask_expanded, float("inf"))
        feat_min = x_for_min.min(dim=1)[0]  # (batch, n_feat)

        # max: 유효 구간 최댓값 (패딩을 -inf로 마스킹)
        x_for_max = x.masked_fill(~mask_expanded, float("-inf"))
        feat_max = x_for_max.max(dim=1)[0]  # (batch, n_feat)

        # normalized sequence length
        norm_len = (lengths.float() / seq_len).unsqueeze(1)  # (batch, 1)

        # concat: (batch, n_feat*5 + 1)
        aggregated = torch.cat(
            [feat_mean, feat_std, feat_last, feat_min, feat_max, norm_len], dim=1
        )

        return self.encoder(aggregated)
