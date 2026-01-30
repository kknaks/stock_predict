"""
데이터 파이프라인 - parquet(과거) + DB(신규) 병합

1. Parquet에서 과거 학습 데이터 로드
2. DB에서 신규 StockPrices + MarketIndices 데이터 로드
3. DB 데이터를 parquet 스키마에 맞춰 피처 + 타겟 생성
4. 중복 제거 후 병합
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import numpy as np
from sqlalchemy import and_
from sqlalchemy.orm import Session

from app.database.database.stocks import StockPrices, MarketIndices

from src.models.base import (
    FEATURE_COLS,
    TARGET_DIRECTION,
    TARGET_RETURN,
    TARGET_MAX_RETURN,
    CORE_FEATURES,
    get_available_features,
    prepare_data_for_training,
)

logger = logging.getLogger(__name__)


def load_parquet_data(parquet_path: str) -> pd.DataFrame:
    """
    Parquet에서 과거 학습 데이터 로드

    Args:
        parquet_path: parquet 파일 경로

    Returns:
        과거 학습 데이터 DataFrame
    """
    path = Path(parquet_path)
    if not path.exists():
        raise FileNotFoundError(f"Parquet file not found: {path}")

    df = pd.read_parquet(path)

    # Ticker → symbol 통일 (Refinitiv 레거시 InfoCode 제거)
    if "Ticker" in df.columns and "symbol" not in df.columns:
        df = df.rename(columns={"Ticker": "symbol"})
    if "InfoCode" in df.columns:
        df = df.drop(columns=["InfoCode"])

    logger.info(f"Loaded parquet: shape={df.shape}, "
                f"date range={df['date'].min()} ~ {df['date'].max()}")
    return df


def load_db_data(
    session: Session,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    DB에서 StockPrices + MarketIndices 데이터 로드 및 피처/타겟 생성

    Args:
        session: DB 세션
        start_date: 시작일 (YYYY-MM-DD)
        end_date: 종료일 (YYYY-MM-DD)

    Returns:
        학습 가능한 DataFrame (parquet 스키마와 호환)
    """
    # StockPrices 쿼리
    query = session.query(StockPrices)
    if start_date:
        query = query.filter(StockPrices.date >= start_date)
    if end_date:
        query = query.filter(StockPrices.date <= end_date)

    stock_prices = query.order_by(StockPrices.date.asc()).all()

    if not stock_prices:
        logger.warning("No stock prices found in DB for given date range")
        return pd.DataFrame()

    # ORM → DataFrame
    rows = []
    for sp in stock_prices:
        row = {
            "date": sp.date,
            "symbol": sp.symbol,
            "open": float(sp.open),
            "high": float(sp.high),
            "low": float(sp.low),
            "close": float(sp.close),
            "volume": int(sp.volume),
            # Pre-computed features from DB
            "gap_pct": float(sp.gap_pct) if sp.gap_pct is not None else None,
            "prev_return": float(sp.prev_return) if sp.prev_return is not None else None,
            "prev_range_pct": float(sp.prev_range_pct) if sp.prev_range_pct is not None else None,
            "prev_upper_shadow": float(sp.prev_upper_shadow) if sp.prev_upper_shadow is not None else None,
            "prev_lower_shadow": float(sp.prev_lower_shadow) if sp.prev_lower_shadow is not None else None,
            "volume_ratio": float(sp.volume_ratio) if sp.volume_ratio is not None else None,
            "rsi_14": float(sp.rsi_14) if sp.rsi_14 is not None else None,
            "atr_14": float(sp.atr_14) if sp.atr_14 is not None else None,
            "atr_ratio": float(sp.atr_ratio) if sp.atr_ratio is not None else None,
            "bollinger_position": float(sp.bollinger_position) if sp.bollinger_position is not None else None,
            "above_ma5": float(sp.above_ma5) if sp.above_ma5 is not None else None,
            "above_ma20": float(sp.above_ma20) if sp.above_ma20 is not None else None,
            "above_ma50": float(sp.above_ma50) if sp.above_ma50 is not None else None,
            "ma5_ma20_cross": float(sp.ma5_ma20_cross) if sp.ma5_ma20_cross is not None else None,
            "return_5d": float(sp.return_5d) if sp.return_5d is not None else None,
            "return_20d": float(sp.return_20d) if sp.return_20d is not None else None,
            "consecutive_up_days": float(sp.consecutive_up_days) if sp.consecutive_up_days is not None else None,
            "market_gap_diff": float(sp.market_gap_diff) if sp.market_gap_diff is not None else None,
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    # 타겟 변수 계산
    df[TARGET_DIRECTION] = (df["close"] > df["open"]).astype(int)
    df[TARGET_RETURN] = (df["close"] - df["open"]) / df["open"] * 100
    df[TARGET_MAX_RETURN] = (df["high"] - df["open"]) / df["open"] * 100

    # 시간 Features
    df["date"] = pd.to_datetime(df["date"])
    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["is_month_start"] = (df["date"].dt.day <= 5).astype(int)
    df["is_month_end"] = (df["date"].dt.day >= 25).astype(int)
    df["is_quarter_end"] = ((df["date"].dt.month.isin([3, 6, 9, 12])) & (df["date"].dt.day >= 25)).astype(int)

    # 갭 상승만 필터 (gap_pct > 0)
    df = df[df["gap_pct"] > 0].copy()

    logger.info(f"Loaded DB data: shape={df.shape}, "
                f"date range={df['date'].min()} ~ {df['date'].max()}")
    return df


def merge_data(
    parquet_df: pd.DataFrame,
    db_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Parquet + DB 데이터 병합 (중복 제거)

    Args:
        parquet_df: Parquet 데이터
        db_df: DB 데이터

    Returns:
        병합된 DataFrame
    """
    if db_df.empty:
        logger.info("DB data empty, using parquet only")
        return parquet_df

    if parquet_df.empty:
        logger.info("Parquet data empty, using DB only")
        return db_df

    # date 타입 통일
    parquet_df = parquet_df.copy()
    db_df = db_df.copy()
    parquet_df["date"] = pd.to_datetime(parquet_df["date"])
    db_df["date"] = pd.to_datetime(db_df["date"])

    # 공통 컬럼만 사용
    common_cols = list(set(parquet_df.columns) & set(db_df.columns))
    parquet_subset = parquet_df[common_cols]
    db_subset = db_df[common_cols]

    # 병합
    merged = pd.concat([parquet_subset, db_subset], ignore_index=True)

    # 중복 제거 (symbol + date 기준, DB 데이터 우선)
    if "symbol" in merged.columns:
        merged = merged.drop_duplicates(subset=["symbol", "date"], keep="last")

    logger.info(f"Merged data: shape={merged.shape}, "
                f"parquet={len(parquet_df)}, db={len(db_df)}, "
                f"after dedup={len(merged)}")
    return merged


def build_training_data(
    session: Session,
    parquet_path: str,
    db_start_date: Optional[str] = None,
    db_end_date: Optional[str] = None,
) -> Tuple[pd.DataFrame, int]:
    """
    학습용 데이터 빌드 (parquet + DB 병합 + 전처리)

    Args:
        session: DB 세션
        parquet_path: parquet 파일 경로
        db_start_date: DB 데이터 시작일
        db_end_date: DB 데이터 종료일

    Returns:
        (전처리된 DataFrame, 전체 샘플 수)
    """
    # 1. Parquet 로드
    parquet_df = load_parquet_data(parquet_path)

    # 2. DB 신규 데이터 로드
    # parquet 마지막 날짜 이후부터 가져오기 (명시적 start_date가 없으면)
    if db_start_date is None and not parquet_df.empty:
        parquet_max_date = pd.to_datetime(parquet_df["date"]).max()
        db_start_date = (parquet_max_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        logger.info(f"Auto db_start_date: {db_start_date} (parquet max + 1)")

    db_df = load_db_data(session, start_date=db_start_date, end_date=db_end_date)

    # 3. 병합
    merged_df = merge_data(parquet_df, db_df)
    total_samples = len(merged_df)

    # 4. 전처리 (src/models/base.py 활용)
    # prepare_data_for_training이 InfoCode를 참조하므로 symbol → InfoCode alias
    if "symbol" in merged_df.columns and "InfoCode" not in merged_df.columns:
        merged_df["InfoCode"] = merged_df["symbol"]
    df_model = prepare_data_for_training(merged_df, include_max_return=True)

    logger.info(f"Training data ready: shape={df_model.shape}, "
                f"total_samples={total_samples}")
    return df_model, total_samples, merged_df
