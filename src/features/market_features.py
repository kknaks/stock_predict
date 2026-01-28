"""
시장 컨텍스트 Feature 생성 모듈

시장 지수 관련 Features를 생성합니다.
- US: SPY (S&P 500), QQQ (NASDAQ 100)
- KR: KOSPI200, KOSDAQ
- JP: Nikkei 225
- HK: HSI
"""

import pandas as pd
import numpy as np
from typing import Optional, List


class MarketFeatures:
    """시장 컨텍스트 Feature 생성 클래스"""

    @staticmethod
    def calculate_index_gaps(market_indices: pd.DataFrame) -> pd.DataFrame:
        """
        시장 지수의 갭 계산 (동적 컬럼 처리)

        Args:
            market_indices: 시장 지수 DataFrame (date, *_pi 컬럼들)

        Returns:
            갭 정보가 추가된 DataFrame
        """
        df = market_indices.copy()
        df = df.sort_values('date')

        # 모든 *_pi 컬럼을 찾아서 갭 계산
        pi_cols = [c for c in df.columns if c.endswith('_pi')]
        
        for col in pi_cols:
            name = col.replace('_pi', '')  # spy, qqq, kospi200, kosdaq 등
            
            # 이전 종가
            df[f'{name}_prev_close'] = df[col].shift(1)
            
            # 갭 퍼센트 계산
            df[f'{name}_gap_pct'] = (
                (df[col] - df[f'{name}_prev_close']) / df[f'{name}_prev_close'] * 100
            )
            
            # 임시 컬럼 삭제
            df = df.drop(columns=[f'{name}_prev_close'])

        return df

    @staticmethod
    def get_primary_index_name(market_indices: pd.DataFrame) -> Optional[str]:
        """
        주요 지수 이름 반환 (region에 따라)
        
        Args:
            market_indices: 시장 지수 DataFrame
            
        Returns:
            주요 지수 이름 (spy, kospi200 등)
        """
        # *_gap_pct 컬럼 찾기
        gap_cols = [c for c in market_indices.columns if c.endswith('_gap_pct')]
        
        if not gap_cols:
            return None
        
        # 우선순위: spy > kospi200 > 첫 번째 컬럼
        for priority in ['spy_gap_pct', 'kospi200_gap_pct']:
            if priority in gap_cols:
                return priority.replace('_gap_pct', '')
        
        return gap_cols[0].replace('_gap_pct', '')

    @staticmethod
    def merge_market_features(
        stock_df: pd.DataFrame,
        market_indices: pd.DataFrame
    ) -> pd.DataFrame:
        """
        종목 데이터에 시장 지수 Features 병합 (동적 컬럼 처리)

        Args:
            stock_df: 종목 데이터 DataFrame (date 컬럼 필수)
            market_indices: 시장 지수 DataFrame (갭 계산 완료)

        Returns:
            시장 Features가 병합된 DataFrame
        """
        if market_indices.empty:
            # 시장 지수 데이터가 없으면 빈 컬럼 추가
            stock_df['market_gap_diff'] = np.nan
            return stock_df

        # 날짜를 datetime으로 통일
        stock_df['date'] = pd.to_datetime(stock_df['date'])
        market_indices['date'] = pd.to_datetime(market_indices['date'])

        # *_gap_pct 컬럼들 찾기
        gap_cols = [c for c in market_indices.columns if c.endswith('_gap_pct')]
        
        # 병합할 컬럼 선택 (date + 모든 gap 컬럼)
        merge_cols = ['date'] + gap_cols
        available_cols = [c for c in merge_cols if c in market_indices.columns]

        # 병합 (left join)
        result = stock_df.merge(
            market_indices[available_cols],
            on='date',
            how='left'
        )

        # 주요 지수 대비 상대 갭 계산 (종목 갭 - 주요 지수 갭)
        primary_index = MarketFeatures.get_primary_index_name(market_indices)
        
        if primary_index and 'gap_pct' in result.columns and f'{primary_index}_gap_pct' in result.columns:
            result['market_gap_diff'] = result['gap_pct'] - result[f'{primary_index}_gap_pct']
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

        # 3. 통계 출력 (동적으로 주요 지수 찾기)
        gap_cols = [c for c in result.columns if c.endswith('_gap_pct') and c != 'gap_pct']
        
        if gap_cols:
            primary_col = gap_cols[0]  # 첫 번째 gap 컬럼 사용
            coverage = result[primary_col].notna().sum() / len(result) * 100
            print(f"   ✓ Market data coverage: {coverage:.1f}%")
            print(f"   ✓ Available indices: {', '.join([c.replace('_gap_pct', '') for c in gap_cols])}")

    # 4. 섹터 Features 추가
    result = MarketFeatures.add_sector_features(result)

    return result
