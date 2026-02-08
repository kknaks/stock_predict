"""LSTM Base Learner — 분리 학습용.

nn.ModuleList로 레이어별 hidden size가 다른 적층 LSTM.

Layer 구조 (hidden_sizes=[128, 64]):
    LSTM(33→128) → Dropout → LSTM(128→64) → last timestep → (batch, 64)
"""

import torch
import torch.nn as nn
from omegaconf import DictConfig

from .base import BaseModel, ModeType


class LSTMPredictor(BaseModel):
    """적층 LSTM encoder + fusion + mode별 head.

    Args:
        input_size: 시계열 피처 수 (33).
        cfg: model config (hidden_sizes, dropout, bidirectional, fusion_dim).
        mode: 출력 모드 ('cls', 'high', 'low', 'all').
    """

    def __init__(self, input_size: int, cfg: DictConfig, mode: ModeType = "all"):
        encoder_dim = cfg.hidden_sizes[-1] * (2 if cfg.bidirectional else 1)
        super().__init__(encoder_dim, cfg, mode)

        self.bidirectional = cfg.bidirectional

        self.lstm_layers = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        in_size = input_size
        for i, hidden_size in enumerate(cfg.hidden_sizes):
            self.lstm_layers.append(
                nn.LSTM(
                    input_size=in_size,
                    hidden_size=hidden_size,
                    batch_first=True,
                    bidirectional=cfg.bidirectional,
                )
            )
            # 마지막 레이어를 제외하고 Dropout 적용
            if i < len(cfg.hidden_sizes) - 1:
                self.dropouts.append(nn.Dropout(cfg.dropout))
            in_size = hidden_size * (2 if cfg.bidirectional else 1)

    def encode(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """(batch, seq_len, input_size) → (batch, hidden_sizes[-1])

        pack_padded_sequence 없이 전체 시퀀스 처리 후 유효 타임스텝 추출.
        MPS/CUDA에서 훨씬 빠름.
        """
        batch_size = x.size(0)

        out = x
        for i, lstm in enumerate(self.lstm_layers):
            out, _ = lstm(out)
            if i < len(self.dropouts):
                out = self.dropouts[i](out)

        # 마지막 유효 타임스텝 추출 (lengths 기반)
        idx = (lengths - 1).long()
        if self.bidirectional:
            # bidirectional: forward의 마지막 + backward의 첫번째
            hidden_size = out.size(-1) // 2
            forward_out = out[torch.arange(batch_size, device=x.device), idx, :hidden_size]
            backward_out = out[:, 0, hidden_size:]
            h = torch.cat([forward_out, backward_out], dim=-1)
        else:
            h = out[torch.arange(batch_size, device=x.device), idx]
        return h
