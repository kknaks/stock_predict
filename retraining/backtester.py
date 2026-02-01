"""
재학습 후 백테스트 실행

기존 src/backtesting/simulator.py + metric.py 활용
노트북 07_backtesting.ipynb과 동일한 로직

두 가지 손절 전략을 비교:
  - model_3: Model 3 예측 손실의 50%에서 손절
  - fixed_1pct: 고정 -1% 손절
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


# 백테스트 전략 정의
BACKTEST_STRATEGIES: Dict[str, BacktestConfig] = {
    "model_3_stop": BacktestConfig(
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
    ),
    "fixed_1pct_stop": BacktestConfig(
        initial_capital=100000.0,
        max_positions=20,
        position_sizing="equal",
        commission_rate=0.001,
        slippage_rate=0.0005,
        min_expected_return=1.0,
        min_prob_up=0.4,
        take_profit_strategy="model_2_1",
        take_profit_ratio=0.8,
        stop_loss_strategy="fixed",
        fixed_stop_loss_pct=-1.0,
    ),
}


def _prepare_backtest_data(
    model_path: str,
    df_model: pd.DataFrame,
    df_raw: Optional[pd.DataFrame] = None,
    threshold: float = 0.4,
    features: Optional[List[str]] = None,
) -> Optional[Dict[str, Any]]:
    """
    백테스트용 데이터 + 예측 결과 준비 (공통 로직)

    Returns:
        {"df_test": DataFrame, "predictions": DataFrame, "features": list}
        또는 None (데이터 부족 등)
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
        raw_cols = [c for c in ohlc_cols + ["date", "symbol", "InfoCode"] if c in df_raw.columns]
        df_raw_subset = df_raw[raw_cols]

        merge_key = []
        if "date" in df_model.columns and "date" in df_raw_subset.columns:
            merge_key.append("date")
        if "InfoCode" in df_model.columns and "InfoCode" in df_raw_subset.columns:
            merge_key.append("InfoCode")

        if merge_key:
            df_merged = df_model.merge(df_raw_subset, on=merge_key, how="left", suffixes=("", "_raw"))
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

    missing = [c for c in ohlc_cols if c not in df_model.columns]
    if missing:
        logger.warning(f"Missing OHLC columns for backtest: {missing}")
        return None

    required_data_cols = ["date", "open", "high", "low", "close",
                          "target_direction", "target_return", "target_max_return"]
    available_cols = list(dict.fromkeys(
        [c for c in features + required_data_cols if c in df_model.columns]
    ))
    df = df_model[available_cols].dropna(subset=[f for f in features if f in df_model.columns]).copy()

    for col in features:
        if col in df.columns:
            na_count = df[col].isna().sum()
            if na_count > 0:
                df[col] = df[col].fillna(df[col].median())

    if len(df) < 100:
        logger.warning(f"Not enough data for backtest: {len(df)} rows")
        return None

    X = df[features]
    y = df["target_direction"]

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42, stratify=y
    )

    test_idx = X_test.index
    df_test = df.loc[test_idx].copy()

    logger.info(f"Backtest test set: {len(df_test)} samples")

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

    return {"df_test": df_test, "predictions": predictions, "features": features}


def _run_single_strategy(
    name: str,
    config: BacktestConfig,
    df_test: pd.DataFrame,
    predictions: pd.DataFrame,
) -> Optional[Dict[str, Any]]:
    """단일 전략 백테스트 실행"""
    simulator = GapTradingSimulator(config=config)
    results = simulator.run(data=df_test, predictions=predictions)

    if results["trades"].empty:
        logger.warning(f"[{name}] No trades executed")
        return None

    metrics = calculate_all_metrics(
        trades_df=results["trades"],
        equity_df=results["equity"],
        initial_capital=config.initial_capital,
    )

    logger.info(
        f"[{name}] Backtest complete: {metrics.get('n_trades', 0)} trades, "
        f"return={metrics.get('total_return_pct', 0):+.2f}%, "
        f"sharpe={metrics.get('sharpe_ratio', 0):.3f}"
    )

    return {
        "metrics": metrics,
        "trades_df": results["trades"],
        "equity_df": results["equity"],
        "config": config,
    }


def run_backtest(
    model_path: str,
    df_model: pd.DataFrame,
    df_raw: Optional[pd.DataFrame] = None,
    threshold: float = 0.4,
    features: Optional[List[str]] = None,
) -> Optional[Dict[str, Any]]:
    """
    학습된 모델로 백테스트 실행 (두 가지 손절 전략 비교)

    Returns:
        {
            "strategies": {
                "model_3_stop": {"metrics", "trades_df", "equity_df", "config"},
                "fixed_1pct_stop": {"metrics", "trades_df", "equity_df", "config"},
            },
            # 하위 호환: 기본 전략(model_3_stop) 결과를 최상위에도 유지
            "metrics": dict,
            "trades_df": DataFrame,
            "equity_df": DataFrame,
            "config": BacktestConfig,
        }
    """
    prepared = _prepare_backtest_data(
        model_path=model_path,
        df_model=df_model,
        df_raw=df_raw,
        threshold=threshold,
        features=features,
    )
    if prepared is None:
        return None

    df_test = prepared["df_test"]
    predictions = prepared["predictions"]

    strategies_results = {}
    for name, config in BACKTEST_STRATEGIES.items():
        result = _run_single_strategy(name, config, df_test, predictions)
        if result is not None:
            strategies_results[name] = result

    if not strategies_results:
        logger.warning("No strategy produced trades")
        return None

    # 하위 호환: model_3_stop을 기본으로, 없으면 첫 번째 전략
    default_key = "model_3_stop" if "model_3_stop" in strategies_results else next(iter(strategies_results))
    default = strategies_results[default_key]

    return {
        "strategies": strategies_results,
        "metrics": default["metrics"],
        "trades_df": default["trades_df"],
        "equity_df": default["equity_df"],
        "config": default["config"],
    }
