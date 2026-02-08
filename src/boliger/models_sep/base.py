"""Base model — 분리 학습용 공통 인터페이스.

mode 파라미터로 출력 head를 선택:
- 'cls': prob만 출력 (분류 모델 학습용)
- 'high': high_return만 출력 (High 회귀 모델 학습용)
- 'low': low_return만 출력 (Low 회귀 모델 학습용)
- 'high_bin': high_return 구간 분류 (Binned Classification)
- 'low_bin': low_return 구간 분류 (Binned Classification)
- 'all': 전부 출력 (호환성 유지)
"""

from abc import abstractmethod
from typing import Literal

import torch
import torch.nn as nn
from omegaconf import DictConfig

# [d_open_scaled, prev_close_scaled, 6개 횡보 aggregate, 4개 RF top features]
OPEN_INFO_DIM = 12

# Binned Classification 구간 정의
HIGH_BINS = [0.01, 0.02, 0.03]  # 0~1%, 1~2%, 2~3%, 3%+
LOW_BINS = [-0.01, -0.02, -0.03]  # 0~-1%, -1~-2%, -2~-3%, -3%+
N_BINS = 4  # 구간 수

ModeType = Literal["cls", "high", "low", "high_bin", "low_bin", "all"]


class BaseModel(nn.Module):
    """분리 학습을 위한 베이스 모델.

    mode에 따라 출력이 달라짐:
    - 'cls': prob만 출력 (분류 모델 학습용)
    - 'high': high_return만 출력 (High 회귀 모델 학습용)
    - 'low': low_return만 출력 (Low 회귀 모델 학습용)
    - 'all': 기존처럼 전부 출력 (호환성 유지)

    Args:
        encoder_dim: Encoder 출력 차원.
        cfg: model config (fusion_dim, dropout 포함).
        mode: 출력 모드.
    """

    def __init__(self, encoder_dim: int, cfg: DictConfig, mode: ModeType = "all"):
        super().__init__()
        self.mode = mode
        fusion_dim = cfg.fusion_dim

        self.fusion = nn.Sequential(
            nn.Linear(encoder_dim + OPEN_INFO_DIM, fusion_dim),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
        )

        # mode에 따라 head 선택적 생성
        if mode in ("cls", "all"):
            self.head_prob = nn.Linear(fusion_dim, 1)
        if mode in ("high", "all"):
            self.head_high = nn.Linear(fusion_dim, 1)
        if mode in ("low", "all"):
            self.head_low = nn.Linear(fusion_dim, 1)
        if mode == "high_bin":
            self.head_high_bin = nn.Linear(fusion_dim, N_BINS)
        if mode == "low_bin":
            self.head_low_bin = nn.Linear(fusion_dim, N_BINS)

    @abstractmethod
    def encode(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """시계열 입력을 고정 길이 벡터로 인코딩.

        Args:
            x: (batch, seq_len, input_size) 패딩된 시계열.
            lengths: (batch,) 각 샘플의 실제 시퀀스 길이.

        Returns:
            (batch, encoder_dim) 인코딩 벡터.
        """
        ...

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
        open_info: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """모델 순전파.

        Args:
            x: (batch, seq_len, input_size) 패딩된 시계열.
            lengths: (batch,) 각 샘플의 실제 시퀀스 길이.
            open_info: (batch, 12) 오픈 정보 및 횡보 aggregate 피처.

        Returns:
            mode에 따라 다른 출력:
            - 'cls': {'logit': (B,1), 'prob': (B,1)}
            - 'high': {'high_return': (B,1)}
            - 'low': {'low_return': (B,1)}
            - 'high_bin': {'logits': (B, N_BINS), 'probs': (B, N_BINS)}
            - 'low_bin': {'logits': (B, N_BINS), 'probs': (B, N_BINS)}
            - 'all': {'logit', 'prob', 'high_return', 'low_return'}
        """
        h = self.encode(x, lengths)  # (batch, encoder_dim)
        fused = self.fusion(torch.cat([h, open_info], dim=1))  # (batch, fusion_dim)

        result = {}
        if self.mode in ("cls", "all"):
            logit = self.head_prob(fused)
            result["logit"] = logit
            result["prob"] = torch.sigmoid(logit)
        if self.mode in ("high", "all"):
            result["high_return"] = self.head_high(fused)
        if self.mode in ("low", "all"):
            result["low_return"] = self.head_low(fused)
        if self.mode == "high_bin":
            logits = self.head_high_bin(fused)
            result["logits"] = logits
            result["probs"] = torch.softmax(logits, dim=1)
        if self.mode == "low_bin":
            logits = self.head_low_bin(fused)
            result["logits"] = logits
            result["probs"] = torch.softmax(logits, dim=1)

        return result
