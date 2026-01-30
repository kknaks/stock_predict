"""
재학습 후 백테스트 실행

기존 src/backtesting/simulator.py + metric.py 활용
노트북 07_backtesting.ipynb과 동일한 로직
"""

import logging
from typing import Dict, Any, Optional, List

import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from src.backtesting.simulator import GapTradingSimulator, BacktestConfig
from src.backtesting.metric import calculate_all_metrics

from src.models.base import FEATURE_COLS

logger = logging.getLogger(__name__)


def run_backtest(
    model_path: str,
    df_model: pd.DataFrame,
    df_raw: Optional[pd.DataFrame] = None,
    threshold: float = 0.4,
    features: Optional[List[str]] = None,
) -> Optional[Dict[str, Any]]:
    """
    학습된 모델로 백테스트 실행

    Args:
        model_path: pkl 모델 경로
        df_model: 전처리된 학습 데이터 (features + targets, OHLC 없을 수 있음)
        df_raw: 원본 병합 데이터 (OHLC 포함). None이면 df_model에서 OHLC를 찾음
        threshold: 분류 threshold
        features: feature 컬럼 목록 (None이면 pkl에서 추출)

    Returns:
        {
            "metrics": dict,
            "trades_df": DataFrame,
            "equity_df": DataFrame,
            "config": BacktestConfig,
        }
    """
    logger.info(f"Loading model for backtest: {model_path}")
    model_data = joblib.load(model_path)

    stacking_clf = model_data["stacking_clf"]
    stacking_reg_up = model_data["stacking_reg_up"]
    stacking_reg_down = model_data["stacking_reg_down"]
    stacking_reg_up_max = model_data.get("stacking_reg_up_max", None)

    if features is None:
        features = model_data.get("features", FEATURE_COLS)

    features = [f for f in features if f != "target_max_return"]

    # df_model에 OHLC가 없으면 df_raw에서 join
    ohlc_cols = ["open", "high", "low", "close"]
    if df_raw is not None and not all(c in df_model.columns for c in ohlc_cols):
        # df_raw에서 OHLC + date를 df_model의 index 기준으로 join
        raw_cols = [c for c in ohlc_cols + ["date", "symbol", "InfoCode"] if c in df_raw.columns]
        df_raw_subset = df_raw[raw_cols]

        # date + InfoCode/symbol 기준 merge
        merge_key = []
        if "date" in df_model.columns and "date" in df_raw_subset.columns:
            merge_key.append("date")
        if "InfoCode" in df_model.columns and "InfoCode" in df_raw_subset.columns:
            merge_key.append("InfoCode")

        if merge_key:
            df_merged = df_model.merge(df_raw_subset, on=merge_key, how="left", suffixes=("", "_raw"))
            # 이미 있는 컬럼은 덮어쓰지 않음
            for col in ohlc_cols:
                if col not in df_model.columns and col in df_merged.columns:
                    df_model = df_merged
                    break
                elif col + "_raw" in df_merged.columns and col not in df_model.columns:
                    df_merged[col] = df_merged[col + "_raw"]
            if any(c not in df_model.columns for c in ohlc_cols):
                df_model = df_merged
            logger.info(f"Merged OHLC from raw data: {df_model.shape}")
        else:
            logger.warning("Cannot merge OHLC: no common keys (date, InfoCode)")

    # 필요한 컬럼 확인
    required_data_cols = ["date", "open", "high", "low", "close",
                          "target_direction", "target_return", "target_max_return"]
    missing = [c for c in ohlc_cols if c not in df_model.columns]
    if missing:
        logger.warning(f"Missing OHLC columns for backtest: {missing}")
        return None

    available_cols = list(dict.fromkeys(
        [c for c in features + required_data_cols if c in df_model.columns]
    ))
    df = df_model[available_cols].dropna(subset=[f for f in features if f in df_model.columns]).copy()

    # NaN 처리
    for col in features:
        if col in df.columns:
            na_count = df[col].isna().sum()
            if na_count > 0:
                df[col] = df[col].fillna(df[col].median())

    if len(df) < 100:
        logger.warning(f"Not enough data for backtest: {len(df)} rows")
        return None

    # Train/Test 분할
    X = df[features]
    y = df["target_direction"]

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42, stratify=y
    )

    test_idx = X_test.index
    df_test = df.loc[test_idx].copy()

    logger.info(f"Backtest test set: {len(df_test)} samples")

    # 모델 예측
    prob_up = stacking_clf.predict_proba(X_test)[:, 1]
    prob_down = 1 - prob_up
    return_if_up = stacking_reg_up.predict(X_test)
    return_if_down = stacking_reg_down.predict(X_test)
    expected_return = (prob_up * return_if_up) + (prob_down * return_if_down)

    predicted_high = None
    if stacking_reg_up_max is not None:
        predicted_high = stacking_reg_up_max.predict(X_test)

    predictions = pd.DataFrame({
        "prob_up": prob_up,
        "prob_down": prob_down,
        "return_if_up": return_if_up,
        "return_if_down": return_if_down,
        "expected_return": expected_return,
        "predicted_direction": (prob_up >= threshold).astype(int),
        "predicted_high": predicted_high,
    }, index=X_test.index)

    # 백테스트 실행 (노트북 07 기본 전략과 동일)
    config = BacktestConfig(
        initial_capital=100000.0,
        max_positions=20,
        position_sizing="equal",
        commission_rate=0.001,
        slippage_rate=0.0005,
        min_expected_return=1.0,
        min_prob_up=0.4,
        take_profit_strategy="model_2_1",
        take_profit_ratio=0.8,
        stop_loss_strategy="model_3",
        stop_loss_ratio=0.5,
    )

    simulator = GapTradingSimulator(config=config)
    results = simulator.run(data=df_test, predictions=predictions)

    if results["trades"].empty:
        logger.warning("No trades executed in backtest")
        return None

    metrics = calculate_all_metrics(
        trades_df=results["trades"],
        equity_df=results["equity"],
        initial_capital=config.initial_capital,
    )

    logger.info(
        f"Backtest complete: {metrics.get('n_trades', 0)} trades, "
        f"return={metrics.get('total_return_pct', 0):+.2f}%, "
        f"sharpe={metrics.get('sharpe_ratio', 0):.3f}"
    )

    return {
        "metrics": metrics,
        "trades_df": results["trades"],
        "equity_df": results["equity"],
        "config": config,
    }
