"""PyTorch Dataset 클래스.

v3.2: 변곡점 기준 스케일링으로 변경
- 변곡점 Train 데이터 기준 mean/std 사용 (Tree 모델과 동일)
- scaler.npz에서 로드하여 NN/Tree 일관성 유지
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

# Scaler 저장 경로 (기본값)
DEFAULT_SCALER_PATH = Path("data/processed_sep/scaler.npz")

FEATURE_COLS = [
    # === 가격 (1) ===
    "close",
    # === 볼린저 밴드 (6) — rsi, cum_area 제거 (다중공선성) ===
    "bb_upper",
    "bb_mid",
    "bb_lower",
    "bb_width_ratio",
    "bb_position",
    "d1",
    "d2",
    # === 거래량 (2) ===
    "volume",
    "volume_ratio",
    # === 기술적 지표 (1) — macd_signal 제거 (다중공선성) ===
    "macd",
    # === 이동평균 (7) ===
    "ma_5",
    "ma_20",
    "ma_50",
    "above_ma5",
    "above_ma20",
    "above_ma50",
    "ma5_ma20_cross",
    # === ATR (2) ===
    "atr_14",
    "atr_ratio",
    # === 시장 (1) ===
    "prev_market_return",
    # === 전일 패턴 (4) ===
    "prev_return",
    "prev_range_pct",
    "prev_upper_shadow",
    "prev_lower_shadow",
    # === 시간 피처 (5) ===
    "day_of_week",
    "month",
    "is_month_start",
    "is_month_end",
    "is_quarter_end",
    # === 모멘텀 (3) ===
    "return_5d",
    "return_20d",
    "consecutive_up_days",
]

# 횡보 특화 aggregate 피처 (Input2에 포함)
SIDEWAYS_AGG_COLS = [
    "sw_bb_width_slope",
    "sw_volume_slope",
    "sw_rsi_slope",
    "sw_price_range",
    "sw_close_position",
    "sw_d1_slope",
]

# RF Feature Importance 상위 피처 (Input2에 추가, 변곡점 직전 값)
RF_TOP_FEATURES = [
    "bb_position",      # RF importance 0.2342
    "atr_ratio",        # RF importance 0.1377
    "prev_range_pct",   # RF importance 0.0876
    "bb_width_ratio",   # RF importance 0.0726
]

TARGET_COLS = ["day_positive", "day_high_return", "day_low_return"]


class StockDataset(Dataset):
    """변곡점 샘플 기반 시계열 Dataset (사전 패딩 + 텐서 변환).

    초기화 시 모든 샘플을 텐서로 변환하고 max_seq_len으로 패딩하여
    학습 루프에서 CPU 연산을 최소화한다.
    """

    def __init__(
        self,
        samples: list[tuple[np.ndarray, ...]],
        max_seq_len: int = 50,
    ):
        if not samples:
            self.input1 = torch.empty(0)
            self.lengths = torch.empty(0, dtype=torch.long)
            self.input2 = torch.empty(0)
            self.targets = torch.empty(0)
            self.flat_features = torch.empty(0)
            self._targets = np.array([])
            return

        n_samples = len(samples)
        n_features = samples[0][0].shape[1]
        has_flat = len(samples[0]) >= 4

        # 미리 텐서 할당 (메모리 효율)
        self.input1 = torch.zeros(n_samples, max_seq_len, n_features, dtype=torch.float32)
        self.lengths = torch.zeros(n_samples, dtype=torch.long)
        self.input2 = torch.zeros(n_samples, samples[0][1].shape[0], dtype=torch.float32)
        self.targets = torch.zeros(n_samples, samples[0][2].shape[0], dtype=torch.float32)

        # 변곡점 당일 피처 (Tree 모델용)
        if has_flat:
            self.flat_features = torch.zeros(n_samples, n_features, dtype=torch.float32)
        else:
            self.flat_features = torch.empty(0)

        # 한 번에 변환 + 패딩
        for i, sample in enumerate(samples):
            inp1, inp2, tgt = sample[0], sample[1], sample[2]
            seq_len = min(inp1.shape[0], max_seq_len)
            self.input1[i, :seq_len, :] = torch.from_numpy(inp1[:seq_len])
            self.lengths[i] = seq_len
            self.input2[i] = torch.from_numpy(inp2)
            self.targets[i] = torch.from_numpy(tgt)
            if has_flat:
                self.flat_features[i] = torch.from_numpy(sample[3])

        # targets numpy 버전 (필터링용)
        self._targets = self.targets.numpy()

    def __len__(self) -> int:
        return len(self.lengths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """사전 변환된 텐서 반환 (CPU 연산 없음)."""
        return self.input1[idx], self.lengths[idx], self.input2[idx], self.targets[idx]

    def get_targets(self) -> np.ndarray:
        """모든 샘플의 targets 반환 (필터링용)."""
        return self._targets


def collate_fn(
    batch: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """사전 패딩된 텐서를 단순 stack (CPU 연산 최소화).

    Returns:
        input1: (batch, max_seq_len, features), 이미 패딩됨.
        lengths: (batch,), 각 샘플의 실제 시퀀스 길이.
        input2: (batch, 8).
        targets: (batch, 3).
    """
    input1, lengths, input2, targets = zip(*batch)
    return torch.stack(input1), torch.stack(lengths), torch.stack(input2), torch.stack(targets)


def _extract_samples_for_stock(
    args: tuple,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """단일 종목의 샘플 추출 (전역 스케일링).

    v3.1 변경사항:
    - 스케일링: 횡보 구간 내 → 전역 StandardScaler (피처 상관관계 보존)
    - Input2: [gap_ratio, 6개 횡보 aggregate, 4개 RF top features (원본값)]
    """
    stock_df, inflection_indices, max_seq_len, global_mean, global_std = args

    # DataFrame → numpy로 미리 변환 (속도 향상)
    stock_df = stock_df.sort_values("date").reset_index(drop=True)
    feature_arr = stock_df[FEATURE_COLS].values  # (N, n_features)
    target_arr = stock_df[TARGET_COLS].values    # (N, 3)
    open_arr = stock_df["open"].values
    close_arr = stock_df["close"].values
    sideways_arr = stock_df["sideways_days"].values

    # 횡보 aggregate 피처 (변곡점 행에만 유효)
    has_sw_cols = all(col in stock_df.columns for col in SIDEWAYS_AGG_COLS)
    if has_sw_cols:
        sw_agg_arr = stock_df[SIDEWAYS_AGG_COLS].values  # (N, 6)
    else:
        sw_agg_arr = None

    # RF top features 인덱스 (변곡점 직전 값 사용)
    rf_feature_indices = [FEATURE_COLS.index(f) for f in RF_TOP_FEATURES]

    samples = []
    for idx in inflection_indices:
        raw_seq_len = int(sideways_arr[idx])

        # 시퀀스 길이 제한: 최근 N일만 사용
        seq_len = min(raw_seq_len, max_seq_len)

        # 충분한 데이터가 있는지 체크 (제한된 길이 기준)
        if seq_len == 0 or idx < seq_len:
            continue

        # Input1: 횡보 구간 (최근 seq_len일)
        window_features = feature_arr[idx - seq_len : idx]

        # NaN 체크 (빠른 버전)
        if np.isnan(window_features).any():
            continue

        # 전역 스케일링 (global mean/std 사용)
        input1 = (window_features - global_mean) / global_std
        if not np.isfinite(input1).all():
            continue

        # Input2: gap_ratio + 횡보 aggregate 피처 + RF top features (원본값)
        prev_close = close_arr[idx - 1]
        d_open = open_arr[idx]
        if prev_close == 0 or np.isnan(prev_close) or np.isnan(d_open):
            continue

        # gap_ratio (스케일링 없이 원본값 사용)
        gap_ratio = (d_open - prev_close) / prev_close
        if np.isnan(gap_ratio) or not np.isfinite(gap_ratio):
            continue
        gap_ratio = np.clip(gap_ratio, -0.3, 0.3)  # 극단값 clip

        # RF top features: 변곡점 직전(idx-1)의 원본 값 (전역 스케일링)
        rf_features = feature_arr[idx - 1, rf_feature_indices]
        if np.isnan(rf_features).any():
            continue
        # RF features를 전역 통계로 정규화
        rf_means = global_mean[rf_feature_indices]
        rf_stds = global_std[rf_feature_indices]
        rf_features_scaled = (rf_features - rf_means) / rf_stds
        if not np.isfinite(rf_features_scaled).all():
            continue

        # 횡보 aggregate 피처 추가
        if sw_agg_arr is not None:
            sw_agg = sw_agg_arr[idx]
            if np.isnan(sw_agg).any():
                continue
            input2 = np.array([
                gap_ratio,            # gap_ratio (원본)
                0.0,                  # placeholder (기존 prev_close_scaled 자리)
                sw_agg[0],  # sw_bb_width_slope
                sw_agg[1],  # sw_volume_slope
                sw_agg[2],  # sw_rsi_slope
                sw_agg[3],  # sw_price_range
                sw_agg[4],  # sw_close_position
                sw_agg[5],  # sw_d1_slope
                rf_features_scaled[0],  # bb_position
                rf_features_scaled[1],  # atr_ratio
                rf_features_scaled[2],  # prev_range_pct
                rf_features_scaled[3],  # bb_width_ratio
            ])
        else:
            input2 = np.array([
                gap_ratio,
                0.0,
                0.0, 0.0, 0.0, 0.0, 0.5, 0.0,  # 횡보 aggregate 기본값
                rf_features_scaled[0],  # bb_position
                rf_features_scaled[1],  # atr_ratio
                rf_features_scaled[2],  # prev_range_pct
                rf_features_scaled[3],  # bb_width_ratio
            ])

        # Targets
        targets = target_arr[idx]
        if np.isnan(targets).any():
            continue

        # 변곡점 당일 피처 (Tree 모델용, 스케일링 적용)
        inflection_features = feature_arr[idx]
        if np.isnan(inflection_features).any():
            inflection_features = np.nan_to_num(inflection_features, nan=0.0)
        inflection_features_scaled = (inflection_features - global_mean) / global_std
        if not np.isfinite(inflection_features_scaled).all():
            inflection_features_scaled = np.nan_to_num(
                inflection_features_scaled, nan=0.0, posinf=0.0, neginf=0.0
            )

        samples.append((input1, input2, targets, inflection_features_scaled))

    return samples


def _compute_global_statistics(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """전역 스케일링 통계 계산 (deprecated).

    Note:
        이 함수 대신 load_scaler()를 사용하세요.
        변곡점 기준 scaler가 더 좋은 성능을 보입니다.

    Args:
        df: 학습 데이터 DataFrame.

    Returns:
        (global_mean, global_std): 각 피처의 전역 평균/표준편차.
    """
    feature_data = df[FEATURE_COLS].values

    # NaN 제외하고 통계 계산
    global_mean = np.nanmean(feature_data, axis=0)
    global_std = np.nanstd(feature_data, axis=0) + 1e-8

    return global_mean, global_std


def load_scaler(scaler_path: Optional[Path] = None) -> tuple[np.ndarray, np.ndarray]:
    """저장된 scaler 로드.

    변곡점 Train 데이터 기준으로 계산된 mean/std를 로드합니다.
    NN과 Tree 모델 모두 동일한 scaler를 사용해야 합니다.

    Args:
        scaler_path: scaler.npz 파일 경로 (기본: data/processed_sep/scaler.npz).

    Returns:
        (mean, std): 각 피처의 평균/표준편차.

    Raises:
        FileNotFoundError: scaler.npz 파일이 없는 경우.
    """
    if scaler_path is None:
        scaler_path = DEFAULT_SCALER_PATH

    if not scaler_path.exists():
        raise FileNotFoundError(
            f"Scaler not found: {scaler_path}\n"
            "먼저 02_dataset.ipynb를 실행하여 scaler를 저장하세요."
        )

    loaded = np.load(scaler_path, allow_pickle=True)
    mean = loaded["mean"]
    std = loaded["std"]

    print(f"[Scaler] Loaded from {scaler_path}")
    print(f"  Mean range: [{mean.min():.4f}, {mean.max():.4f}]")
    print(f"  Std range: [{std.min():.4f}, {std.max():.4f}]")

    return mean, std


def _extract_samples(
    df: pd.DataFrame,
    max_seq_len: int = 20,
    global_mean: Optional[np.ndarray] = None,
    global_std: Optional[np.ndarray] = None,
) -> list[tuple[np.ndarray, ...]]:
    """변곡점 샘플 추출 (변곡점 기준 스케일링).

    v3.2: 변곡점 Train 기준 scaler 사용 (Tree 모델과 동일).
    scaler.npz가 있으면 로드, 없으면 fallback으로 현재 데이터에서 계산.

    Args:
        df: 피처가 계산된 DataFrame (is_inflection, sideways_days, 횡보 aggregate 포함).
        max_seq_len: 시퀀스 최대 길이 (config에서 설정).
        global_mean: 전역 평균 (외부에서 전달 시 사용).
        global_std: 전역 표준편차 (외부에서 전달 시 사용).

    Returns:
        samples 리스트.
    """
    # 저장된 scaler 로드 시도 (변곡점 기준)
    if global_mean is None or global_std is None:
        try:
            global_mean, global_std = load_scaler()
        except FileNotFoundError:
            print("  [Warning] scaler.npz not found, computing from current data...")
            global_mean, global_std = _compute_global_statistics(df)

    # 종목별로 그룹화
    grouped = df.groupby("InfoCode")
    total_stocks = len(grouped)
    print(f"  Processing {total_stocks} stocks... (max_seq_len={max_seq_len}, global_scaling=True)")

    # 순차 처리 (메모리 안전, numpy 최적화로 속도 보완)
    all_samples = []
    valid_stocks = 0
    for i, (info_code, stock_df) in enumerate(grouped):
        # 변곡점 필터링
        stock_inflection_mask = (
            stock_df["is_inflection"]
            & stock_df["day_high_return"].notna()
            & stock_df["day_low_return"].notna()
        )
        if stock_inflection_mask.sum() == 0:
            continue

        # 복사 및 인덱스 리셋
        stock_df = stock_df.copy().reset_index(drop=True)
        inflection_indices = stock_df[stock_inflection_mask.values].index.tolist()

        args = (stock_df, inflection_indices, max_seq_len, global_mean, global_std)
        samples = _extract_samples_for_stock(args)
        all_samples.extend(samples)
        valid_stocks += 1

        # 진행률 (500개마다)
        if (i + 1) % 500 == 0:
            print(f"    {i + 1}/{total_stocks} stocks, {len(all_samples)} samples")

    print(f"  Extracted {len(all_samples)} samples from {valid_stocks} stocks")

    return all_samples


def create_datasets(cfg: DictConfig) -> tuple[StockDataset, StockDataset, StockDataset]:
    """시간 기준 분할로 train/val/test Dataset 생성.

    v3.2 변경사항:
    - 변곡점 기준 스케일링: scaler.npz에서 로드 (Tree 모델과 동일)
    - NN/Tree 일관성 유지로 앙상블 성능 향상

    Args:
        cfg: data config.

    Returns:
        (train_dataset, val_dataset, test_dataset).
    """
    processed_path = Path(cfg.processed_dir) / "features.parquet"
    df = pd.read_parquet(processed_path)
    df["date"] = pd.to_datetime(df["date"])

    train_end = pd.Timestamp(cfg.train_end)
    val_end = pd.Timestamp(cfg.val_end)

    train_df = df[df["date"] <= train_end]
    val_df = df[(df["date"] > train_end) & (df["date"] <= val_end)]
    test_df = df[df["date"] > val_end]

    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # 저장된 scaler 로드 (변곡점 Train 기준)
    print("\n[Scaler] Loading from scaler.npz (inflection-based)...")
    try:
        global_mean, global_std = load_scaler()
    except FileNotFoundError:
        print("  [Warning] scaler.npz not found, computing from training data...")
        global_mean, global_std = _compute_global_statistics(train_df)
        print(f"  Features: {len(global_mean)}, mean range: [{global_mean.min():.4f}, {global_mean.max():.4f}]")

    max_seq_len = cfg.get("max_seq_len", 20)

    # 샘플 추출 (동일한 전역 통계 사용)
    print("\n[Train]")
    train_samples = _extract_samples(train_df, max_seq_len=max_seq_len,
                                      global_mean=global_mean, global_std=global_std)
    print("\n[Val]")
    val_samples = _extract_samples(val_df, max_seq_len=max_seq_len,
                                    global_mean=global_mean, global_std=global_std)
    print("\n[Test]")
    test_samples = _extract_samples(test_df, max_seq_len=max_seq_len,
                                     global_mean=global_mean, global_std=global_std)

    print(
        f"\nSamples - Train: {len(train_samples)}, Val: {len(val_samples)}, "
        f"Test: {len(test_samples)}"
    )

    # 시퀀스 길이 통계
    if train_samples:
        lens = [s[0].shape[0] for s in train_samples]
        print(f"Seq lengths - min: {min(lens)}, max: {max(lens)}, mean: {np.mean(lens):.0f}")

    # 타겟 분포 확인 (base rate)
    if train_samples:
        targets = np.array([s[2] for s in train_samples])
        print(
            f"Train day_positive base rate: {targets[:, 0].mean():.4f} "
            f"(redefined: day_high_return > threshold)"
        )

    return (
        StockDataset(train_samples, max_seq_len=max_seq_len),
        StockDataset(val_samples, max_seq_len=max_seq_len),
        StockDataset(test_samples, max_seq_len=max_seq_len),
    )
