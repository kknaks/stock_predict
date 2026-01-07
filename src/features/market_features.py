"""
시장 컨텍스트 Feature 생성 모듈

시장 지수(SPY, QQQ, VIX) 관련 Features를 생성합니다.
"""

import pandas as pd
import numpy as np
from typing import Optional


class MarketFeatures:
    """시장 컨텍스트 Feature 생성 클래스"""

    @staticmethod
    def calculate_index_gaps(market_indices: pd.DataFrame) -> pd.DataFrame:
        """
        시장 지수의 갭 계산

        Args:
            market_indices: 시장 지수 DataFrame (date, spy_pi, qqq_pi, vix_pi)

        Returns:
            갭 정보가 추가된 DataFrame
        """
        df = market_indices.copy()
        df = df.sort_values('date')

        # SPY 갭 계산 (Price Index 기준)
        if 'spy_pi' in df.columns:
            df['spy_prev_close'] = df['spy_pi'].shift(1)
            df['spy_gap_pct'] = ((df['spy_pi'] - df['spy_prev_close']) / df['spy_prev_close'] * 100)
        else:
            df['spy_gap_pct'] = np.nan

        # QQQ 갭 계산
        if 'qqq_pi' in df.columns:
            df['qqq_prev_close'] = df['qqq_pi'].shift(1)
            df['qqq_gap_pct'] = ((df['qqq_pi'] - df['qqq_prev_close']) / df['qqq_prev_close'] * 100)
        else:
            df['qqq_gap_pct'] = np.nan

        # VIX 변화율
        if 'vix_pi' in df.columns:
            df['vix_prev'] = df['vix_pi'].shift(1)
            df['vix_change'] = ((df['vix_pi'] - df['vix_prev']) / df['vix_prev'] * 100)
            df['vix_level'] = df['vix_pi']

            # VIX 카테고리
            df['vix_category'] = pd.cut(
                df['vix_level'],
                bins=[0, 15, 25, 1000],
                labels=['low', 'normal', 'high']
            )
        else:
            df['vix_change'] = np.nan
            df['vix_level'] = np.nan
            df['vix_category'] = None

        # 불필요한 컬럼 제거
        cols_to_drop = ['spy_prev_close', 'qqq_prev_close', 'vix_prev']
        df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

        return df

    @staticmethod
    def merge_market_features(
        stock_df: pd.DataFrame,
        market_indices: pd.DataFrame
    ) -> pd.DataFrame:
        """
        종목 데이터에 시장 지수 Features 병합

        Args:
            stock_df: 종목 데이터 DataFrame (date 컬럼 필수)
            market_indices: 시장 지수 DataFrame (갭 계산 완료)

        Returns:
            시장 Features가 병합된 DataFrame
        """
        if market_indices.empty:
            # 시장 지수 데이터가 없으면 NaN으로 채움
            stock_df['spy_gap_pct'] = np.nan
            stock_df['qqq_gap_pct'] = np.nan
            stock_df['vix_level'] = np.nan
            stock_df['vix_change'] = np.nan
            stock_df['vix_category'] = None
            stock_df['market_gap_diff'] = np.nan
            return stock_df

        # 날짜를 datetime으로 통일
        stock_df['date'] = pd.to_datetime(stock_df['date'])
        market_indices['date'] = pd.to_datetime(market_indices['date'])

        # 필요한 컬럼만 선택
        merge_cols = ['date', 'spy_gap_pct', 'qqq_gap_pct', 'vix_level', 'vix_change', 'vix_category']
        available_cols = [c for c in merge_cols if c in market_indices.columns]

        # 병합 (left join)
        result = stock_df.merge(
            market_indices[available_cols],
            on='date',
            how='left'
        )

        # 상대 갭 계산 (종목 갭 - SPY 갭)
        if 'gap_pct' in result.columns and 'spy_gap_pct' in result.columns:
            result['market_gap_diff'] = result['gap_pct'] - result['spy_gap_pct']
        else:
            result['market_gap_diff'] = np.nan

        return result

    @staticmethod
    def add_sector_features(
        df: pd.DataFrame,
        sector_col: str = 'sector',
        industry_col: str = 'industry'
    ) -> pd.DataFrame:
        """
        섹터 Features 추가

        Args:
            df: DataFrame with sector/industry columns
            sector_col: 섹터 컬럼명
            industry_col: 산업 컬럼명

        Returns:
            섹터 Features가 추가된 DataFrame
        """
        # 섹터/산업 정보가 있으면 유지, 없으면 NaN
        if sector_col not in df.columns:
            df[sector_col] = np.nan

        if industry_col not in df.columns:
            df[industry_col] = np.nan

        # 섹터별 평균 갭 계산 (같은 날짜, 같은 섹터)
        if 'gap_pct' in df.columns and df[sector_col].notna().any():
            # 날짜와 섹터별로 평균 갭 계산
            sector_gap = df.groupby(['date', sector_col])['gap_pct'].transform('mean')
            df['sector_gap_pct'] = sector_gap

            # 섹터 대비 상대 갭
            df['sector_relative_gap'] = df['gap_pct'] - df['sector_gap_pct']
        else:
            df['sector_gap_pct'] = np.nan
            df['sector_relative_gap'] = np.nan

        return df


def add_market_context(
    stock_df: pd.DataFrame,
    market_indices: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    종목 데이터에 시장 컨텍스트 Features 추가

    Args:
        stock_df: 종목 데이터 DataFrame
        market_indices: 시장 지수 DataFrame (raw, 갭 미계산)

    Returns:
        시장 컨텍스트 Features가 추가된 DataFrame
    """
    if market_indices is None or market_indices.empty:
        print("   Warning: No market indices data provided")
        # NaN으로 채움
        result = MarketFeatures.merge_market_features(stock_df, pd.DataFrame())
    else:
        # 1. 시장 지수 갭 계산
        print("   Calculating market index gaps...")
        market_with_gaps = MarketFeatures.calculate_index_gaps(market_indices)

        # 2. 종목 데이터에 병합
        print("   Merging market features...")
        result = MarketFeatures.merge_market_features(stock_df, market_with_gaps)

        # 통계 출력
        coverage = result['spy_gap_pct'].notna().sum() / len(result) * 100
        print(f"   ✓ Market data coverage: {coverage:.1f}%")

    # 3. 섹터 Features 추가 (현재는 플레이스홀더)
    result = MarketFeatures.add_sector_features(result)

    return result
