"""
기술적 지표 Feature 생성 모듈

이동평균, RSI, ATR, Bollinger Bands 등의 기술적 지표를 계산합니다.
"""

import pandas as pd
import numpy as np
from typing import Optional


class TechnicalFeatures:
    """기술적 지표 생성 클래스"""

    @staticmethod
    def calculate_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
        """
        이동평균 Features 계산

        Args:
            df: OHLCV DataFrame (종목별로 정렬되어 있어야 함)

        Returns:
            이동평균 Features가 추가된 DataFrame
        """
        # 5일, 20일, 50일 이동평균 (종가 기준)
        df['ma_5'] = df['close'].rolling(window=5, min_periods=1).mean()
        df['ma_20'] = df['close'].rolling(window=20, min_periods=1).mean()
        df['ma_50'] = df['close'].rolling(window=50, min_periods=1).mean()

        # 전일 종가가 이동평균선 위에 있는지 (Look-ahead bias 방지)
        df['above_ma5'] = (df['prev_close'] > df['ma_5'].shift(1)).astype(int)
        df['above_ma20'] = (df['prev_close'] > df['ma_20'].shift(1)).astype(int)
        df['above_ma50'] = (df['prev_close'] > df['ma_50'].shift(1)).astype(int)

        # 5일선이 20일선 위에 있는지 (골든크로스/데드크로스)
        df['ma5_ma20_cross'] = (df['ma_5'].shift(1) > df['ma_20'].shift(1)).astype(int)

        return df

    @staticmethod
    def calculate_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        RSI (Relative Strength Index) 계산

        Args:
            df: OHLCV DataFrame
            period: RSI 기간 (기본 14일)

        Returns:
            RSI Features가 추가된 DataFrame
        """
        # 일간 가격 변화
        delta = df['close'].diff()

        # 상승/하락 분리
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        # 평균 상승/하락폭 (Wilder's smoothing)
        avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()

        # RS와 RSI 계산
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        # Look-ahead bias 방지: 전일 RSI 사용
        df['rsi_14'] = rsi.shift(1)

        # RSI 카테고리
        df['rsi_category'] = pd.cut(
            df['rsi_14'],
            bins=[0, 30, 70, 100],
            labels=['oversold', 'neutral', 'overbought'],
            include_lowest=True
        )

        return df

    @staticmethod
    def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        ATR (Average True Range) 계산

        Args:
            df: OHLCV DataFrame
            period: ATR 기간 (기본 14일)

        Returns:
            ATR Features가 추가된 DataFrame
        """
        # True Range 계산
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift(1))
        low_close = np.abs(df['low'] - df['close'].shift(1))

        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

        # ATR 계산 (Wilder's smoothing)
        atr = true_range.ewm(alpha=1/period, min_periods=period, adjust=False).mean()

        # Look-ahead bias 방지: 전일 ATR 사용
        df['atr_14'] = atr.shift(1)

        # 갭 크기 대비 ATR 비율
        if 'gap_pct' in df.columns and 'prev_close' in df.columns:
            gap_size = np.abs(df['open'] - df['prev_close'])
            df['atr_ratio'] = gap_size / (df['atr_14'] + 1e-10)  # 0으로 나누기 방지
        else:
            df['atr_ratio'] = np.nan

        return df

    @staticmethod
    def calculate_bollinger_bands(df: pd.DataFrame, period: int = 20, std: float = 2.0) -> pd.DataFrame:
        """
        볼린저 밴드 계산

        Args:
            df: OHLCV DataFrame
            period: 이동평균 기간 (기본 20일)
            std: 표준편차 배수 (기본 2.0)

        Returns:
            볼린저 밴드 Features가 추가된 DataFrame
        """
        # 중심선 (20일 이동평균)
        sma = df['close'].rolling(window=period, min_periods=period).mean()

        # 표준편차
        rolling_std = df['close'].rolling(window=period, min_periods=period).std()

        # 상단/하단 밴드
        upper_band = sma + (rolling_std * std)
        lower_band = sma - (rolling_std * std)

        # 볼린저 밴드 위치 (0~1, 0.5가 중심)
        # Look-ahead bias 방지: 전일 종가와 전일 밴드 사용
        df['bb_upper'] = upper_band.shift(1)
        df['bb_lower'] = lower_band.shift(1)
        df['bb_middle'] = sma.shift(1)

        df['bollinger_position'] = (
            (df['prev_close'] - df['bb_lower']) /
            (df['bb_upper'] - df['bb_lower'] + 1e-10)  # 0으로 나누기 방지
        )

        # 밴드 범위 밖인 경우 처리 (클리핑)
        df['bollinger_position'] = df['bollinger_position'].clip(0, 1)

        return df

    @staticmethod
    def calculate_momentum(df: pd.DataFrame) -> pd.DataFrame:
        """
        추세 및 모멘텀 Features 계산

        Args:
            df: OHLCV DataFrame

        Returns:
            모멘텀 Features가 추가된 DataFrame
        """
        # 5일 수익률
        df['return_5d'] = ((df['close'] - df['close'].shift(5)) / df['close'].shift(5) * 100)

        # 20일 수익률
        df['return_20d'] = ((df['close'] - df['close'].shift(20)) / df['close'].shift(20) * 100)

        # 연속 상승일 수
        # 양수면 상승, 음수면 하락, 0이면 없음
        daily_return = df['close'].pct_change()
        is_up = (daily_return > 0).astype(int)

        # 연속 상승일 카운트
        consecutive = (is_up * (is_up.groupby((is_up != is_up.shift()).cumsum()).cumcount() + 1))

        # Look-ahead bias 방지: 전일까지의 연속 상승일
        df['consecutive_up_days'] = consecutive.shift(1)

        # Look-ahead bias 방지: 전일 기준 수익률로 변경
        df['return_5d'] = df['return_5d'].shift(1)
        df['return_20d'] = df['return_20d'].shift(1)

        return df

    @staticmethod
    def add_all_technical_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        모든 기술적 지표를 한 번에 추가

        Args:
            df: 단일 종목의 OHLCV DataFrame (날짜순 정렬되어 있어야 함)

        Returns:
            모든 기술적 지표가 추가된 DataFrame
        """
        df = TechnicalFeatures.calculate_moving_averages(df)
        df = TechnicalFeatures.calculate_rsi(df)
        df = TechnicalFeatures.calculate_atr(df)
        df = TechnicalFeatures.calculate_bollinger_bands(df)
        df = TechnicalFeatures.calculate_momentum(df)

        return df


def add_technical_features_parallel(
    df: pd.DataFrame,
    max_workers: int = 10
) -> pd.DataFrame:
    """
    종목별로 병렬로 기술적 지표 추가

    Args:
        df: OHLCV DataFrame (InfoCode, date, open, high, low, close, volume 필수)
        max_workers: 최대 워커 수

    Returns:
        기술적 지표가 추가된 DataFrame
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from tqdm import tqdm

    # InfoCode별로 그룹화
    grouped = df.groupby('InfoCode')
    stock_codes = list(grouped.groups.keys())

    print(f"   Adding technical features to {len(stock_codes)} stocks with {max_workers} workers...")

    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # 각 종목에 대해 비동기 실행
        future_to_code = {
            executor.submit(
                TechnicalFeatures.add_all_technical_features,
                group.sort_values('date').copy()
            ): code
            for code, group in grouped
        }

        # 결과 수집
        for future in tqdm(
            as_completed(future_to_code),
            total=len(stock_codes),
            desc="   Technical indicators"
        ):
            try:
                result = future.result()
                if not result.empty:
                    results.append(result)
            except Exception as e:
                code = future_to_code[future]
                print(f"   Error adding technical features for InfoCode {code}: {e}")

    # 모든 결과 병합
    if results:
        return pd.concat(results, ignore_index=True)
    else:
        return pd.DataFrame()
