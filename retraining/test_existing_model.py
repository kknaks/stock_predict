"""
기존 모델 평가 스크립트

이미 학습된 pkl 모델을 로드하여 test_recent=True/False 조건으로
분류 메트릭 + 백테스트를 실행

Usage:
    python -m retraining.test_existing_model --model_path models/stacking/v1.0.0/stacking_hybrid_model.pkl
    python -m retraining.test_existing_model --model_path models/stacking/v1.0.0/stacking_hybrid_model.pkl --test_recent
"""

import sys
import logging
import argparse

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_absolute_error, mean_squared_error, r2_score,
)

from retraining.config import settings
from retraining.database import get_session_factory
from retraining.data_pipeline import build_training_data
from retraining.backtester import run_backtest
from src.models.base import FEATURE_COLS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def evaluate_existing_model(
    model_path: str,
    test_recent: bool = True,
    test_size: float = 0.1,
    threshold: float = 0.4,
):
    logger.info(f"=== Evaluate Existing Model ===")
    logger.info(f"Model: {model_path}")
    logger.info(f"test_recent: {test_recent}, test_size: {test_size}, threshold: {threshold}")

    # 1. 모델 로드 (노트북에서 저장된 pkl은 __main__.StackingHybridPredictor 참조)
    from src.prediction.predictor import HybridPredictor
    import types
    if "__main__" not in sys.modules or not hasattr(sys.modules["__main__"], "StackingHybridPredictor"):
        main_module = sys.modules.get("__main__", types.ModuleType("__main__"))
        main_module.StackingHybridPredictor = HybridPredictor
        sys.modules["__main__"] = main_module
    model_data = joblib.load(model_path)
    stacking_clf = model_data["stacking_clf"]
    stacking_reg_up = model_data["stacking_reg_up"]
    stacking_reg_down = model_data["stacking_reg_down"]
    stacking_reg_up_max = model_data.get("stacking_reg_up_max")
    features = model_data.get("features", FEATURE_COLS)
    features = [f for f in features if f != "target_max_return"]

    logger.info(f"Features: {len(features)}개")

    # 2. 데이터 로드
    session_factory = get_session_factory()
    with session_factory() as session:
        result = build_training_data(
            session=session,
            parquet_path=settings.parquet_path,
        )
        if len(result) == 3:
            df_model, total_samples, merged_raw_df = result
        else:
            df_model, total_samples = result
            merged_raw_df = None

    logger.info(f"Data: {len(df_model)} samples, {df_model['date'].min()} ~ {df_model['date'].max()}")

    # 3. 테스트셋 분할
    if test_recent:
        df_sorted = df_model.sort_values("date").reset_index(drop=True)
        n = len(df_sorted)
        n_test = int(n * test_size)
        df_test = df_sorted.iloc[n - n_test:].copy()
        df_train = df_sorted.iloc[:n - n_test].copy()
        logger.info(f"test_recent=True: test={len(df_test)}, date range: {df_test['date'].min()} ~ {df_test['date'].max()}")
    else:
        from sklearn.model_selection import train_test_split
        y = df_model["target_direction"]
        _, test_idx = train_test_split(
            df_model.index, test_size=test_size, random_state=42, stratify=y
        )
        df_test = df_model.loc[test_idx].copy()
        logger.info(f"test_recent=False: test={len(df_test)} (random split)")

    # NaN 처리
    for col in features:
        if col in df_test.columns:
            na_count = df_test[col].isna().sum()
            if na_count > 0:
                df_test[col] = df_test[col].fillna(df_test[col].median())

    X_test = df_test[[f for f in features if f in df_test.columns]]
    y_test = df_test["target_direction"]

    # 4. 분류 모델 평가
    logger.info("")
    logger.info("=== Classifier Metrics ===")
    prob_up = stacking_clf.predict_proba(X_test)[:, 1]
    y_pred = (prob_up >= threshold).astype(int)

    clf_metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, prob_up),
        "f1_score": f1_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
    }
    for k, v in clf_metrics.items():
        logger.info(f"  {k}: {v:.4f}")

    # 5. 회귀 모델 평가
    # Regressor Up (상승 케이스만)
    up_mask = y_test == 1
    if up_mask.sum() > 0:
        X_up = X_test[up_mask]
        y_up_actual = df_test.loc[up_mask, "target_return"]

        logger.info("")
        logger.info("=== Regressor Up (Close Return) ===")
        pred_up = stacking_reg_up.predict(X_up)
        logger.info(f"  MAE: {mean_absolute_error(y_up_actual, pred_up):.4f}")
        logger.info(f"  RMSE: {np.sqrt(mean_squared_error(y_up_actual, pred_up)):.4f}")
        logger.info(f"  R2: {r2_score(y_up_actual, pred_up):.4f}")

        if stacking_reg_up_max and "target_max_return" in df_test.columns:
            logger.info("")
            logger.info("=== Regressor High (Max Return) ===")
            y_high_actual = df_test.loc[up_mask, "target_max_return"]
            pred_high = stacking_reg_up_max.predict(X_up)
            logger.info(f"  MAE: {mean_absolute_error(y_high_actual, pred_high):.4f}")
            logger.info(f"  RMSE: {np.sqrt(mean_squared_error(y_high_actual, pred_high)):.4f}")
            logger.info(f"  R2: {r2_score(y_high_actual, pred_high):.4f}")

    # Regressor Down (하락 케이스만)
    down_mask = y_test == 0
    if down_mask.sum() > 0:
        X_down = X_test[down_mask]
        y_down_actual = df_test.loc[down_mask, "target_return"]

        logger.info("")
        logger.info("=== Regressor Down (Loss) ===")
        pred_down = stacking_reg_down.predict(X_down)
        logger.info(f"  MAE: {mean_absolute_error(y_down_actual, pred_down):.4f}")
        logger.info(f"  RMSE: {np.sqrt(mean_squared_error(y_down_actual, pred_down)):.4f}")
        logger.info(f"  R2: {r2_score(y_down_actual, pred_down):.4f}")

    # 6. 백테스트
    logger.info("")
    logger.info("=== Backtest ===")
    backtest_result = run_backtest(
        model_path=model_path,
        df_model=df_test,
        df_raw=merged_raw_df if 'merged_raw_df' in dir() else None,
        threshold=threshold,
        features=features,
    )

    if backtest_result:
        metrics = backtest_result["metrics"]
        for k, v in metrics.items():
            if isinstance(v, float):
                logger.info(f"  {k}: {v:.4f}")
            else:
                logger.info(f"  {k}: {v}")
    else:
        logger.warning("  Backtest returned no results")

    return clf_metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate existing model")
    parser.add_argument("--model_path", required=True, help="Path to pkl file")
    parser.add_argument("--test_recent", action="store_true", help="Use recent data as test set")
    parser.add_argument("--test_size", type=float, default=0.1)
    parser.add_argument("--threshold", type=float, default=0.4)
    args = parser.parse_args()

    evaluate_existing_model(
        model_path=args.model_path,
        test_recent=args.test_recent,
        test_size=args.test_size,
        threshold=args.threshold,
    )


if __name__ == "__main__":
    main()
