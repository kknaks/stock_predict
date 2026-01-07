"""
데이터 전처리 모듈

Raw 데이터를 전처리하여 preprocessed_df 생성
"""

from typing import Optional, Dict, List
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import pandas as pd
import numpy as np


class DataPreprocessor:
    """데이터 전처리 클래스"""

    def __init__(self, config: dict):
        """
        초기화

        Args:
            config: 설정 딕셔너리
        """
        self.config = config
        self.data_config = config.get('data', {})

    @staticmethod
    def _process_single_stock(stock_data: pd.DataFrame) -> pd.DataFrame:
        """
        단일 종목 데이터 처리 (병렬 처리용 static method)

        Args:
            stock_data: 단일 종목의 OHLCV DataFrame

        Returns:
            처리된 DataFrame
        """
        if stock_data.empty:
            return stock_data

        # 날짜 순 정렬
        stock_data = stock_data.sort_values('date').copy()

        # 1. Lag features
        stock_data['prev_open'] = stock_data['open'].shift(1)
        stock_data['prev_high'] = stock_data['high'].shift(1)
        stock_data['prev_low'] = stock_data['low'].shift(1)
        stock_data['prev_close'] = stock_data['close'].shift(1)
        stock_data['prev_volume'] = stock_data['volume'].shift(1)

        # 2. 갭 계산
        stock_data['gap_pct'] = ((stock_data['open'] - stock_data['prev_close']) / stock_data['prev_close'] * 100)
        stock_data['is_gap_up'] = (stock_data['gap_pct'] > 0).astype(int)
        stock_data['gap_size_category'] = pd.cut(
            stock_data['gap_pct'],
            bins=[-np.inf, 2, 5, np.inf],
            labels=['small', 'medium', 'large']
        )

        # 3. 수익률
        stock_data['intraday_return'] = ((stock_data['close'] - stock_data['open']) / stock_data['open'] * 100)
        stock_data['prev_return'] = ((stock_data['prev_close'] - stock_data['prev_open']) / stock_data['prev_open'] * 100)
        stock_data['prev_range_pct'] = ((stock_data['prev_high'] - stock_data['prev_low']) / stock_data['prev_close'] * 100)
        stock_data['prev_upper_shadow'] = (
            (stock_data['prev_high'] - stock_data[['prev_open', 'prev_close']].max(axis=1)) / stock_data['prev_close']
        )
        stock_data['prev_lower_shadow'] = (
            (stock_data[['prev_open', 'prev_close']].min(axis=1) - stock_data['prev_low']) / stock_data['prev_close']
        )

        # 4. 거래량
        stock_data['avg_volume_20d'] = stock_data['volume'].rolling(window=20, min_periods=1).mean()
        stock_data['volume_ratio'] = stock_data['prev_volume'] / stock_data['avg_volume_20d']

        # 5. 타겟 변수
        stock_data['target_direction'] = (stock_data['intraday_return'] > 0).astype(int)
        stock_data['target_return'] = stock_data['intraday_return']

        return stock_data

    def process_stocks_parallel(self, df: pd.DataFrame, max_workers: Optional[int] = None) -> pd.DataFrame:
        """
        종목별 병렬 처리

        Args:
            df: OHLCV DataFrame
            max_workers: 최대 워커 수 (None이면 CPU 코어 수)

        Returns:
            처리된 DataFrame
        """
        if max_workers is None:
            max_workers = self.data_config.get('parallel', {}).get('max_workers', 10)

        # InfoCode별로 그룹화
        grouped = df.groupby('InfoCode')
        stock_codes = list(grouped.groups.keys())

        print(f"   Processing {len(stock_codes)} stocks with {max_workers} workers...")

        results = []
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # 각 종목에 대해 비동기 실행
            future_to_code = {
                executor.submit(self._process_single_stock, group.copy()): code
                for code, group in grouped
            }

            # 결과 수집
            for future in tqdm(as_completed(future_to_code), total=len(stock_codes), desc="   Processing stocks"):
                try:
                    result = future.result()
                    if not result.empty:
                        results.append(result)
                except Exception as e:
                    code = future_to_code[future]
                    print(f"   Error processing InfoCode {code}: {e}")

        # 모든 결과 병합
        if results:
            return pd.concat(results, ignore_index=True)
        else:
            return pd.DataFrame()

    def add_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Lag features 추가 (전일 가격 정보)

        Args:
            df: OHLCV DataFrame (InfoCode, date, open, high, low, close, volume)

        Returns:
            Lag features가 추가된 DataFrame
        """
        # 정렬
        df = df.sort_values(['InfoCode', 'date']).copy()

        # 종목별로 전일 가격 정보 생성
        df['prev_open'] = df.groupby('InfoCode')['open'].shift(1)
        df['prev_high'] = df.groupby('InfoCode')['high'].shift(1)
        df['prev_low'] = df.groupby('InfoCode')['low'].shift(1)
        df['prev_close'] = df.groupby('InfoCode')['close'].shift(1)
        df['prev_volume'] = df.groupby('InfoCode')['volume'].shift(1)

        return df

    def calculate_gap(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        갭 계산

        Args:
            df: Lag features가 포함된 DataFrame

        Returns:
            갭 정보가 추가된 DataFrame
        """
        # 갭 상승률 계산
        df['gap_pct'] = ((df['open'] - df['prev_close']) / df['prev_close'] * 100)

        # 갭 상승 여부
        df['is_gap_up'] = (df['gap_pct'] > 0).astype(int)

        # 갭 크기 카테고리
        df['gap_size_category'] = pd.cut(
            df['gap_pct'],
            bins=[-np.inf, 2, 5, np.inf],
            labels=['small', 'medium', 'large']
        )

        return df

    def calculate_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        수익률 계산

        Args:
            df: 가격 데이터 DataFrame

        Returns:
            수익률이 추가된 DataFrame
        """
        # 당일 수익률 (시가 -> 종가)
        df['intraday_return'] = ((df['close'] - df['open']) / df['open'] * 100)

        # 전일 수익률
        df['prev_return'] = ((df['prev_close'] - df['prev_open']) / df['prev_open'] * 100)

        # 전일 변동폭 (고가-저가)
        df['prev_range_pct'] = ((df['prev_high'] - df['prev_low']) / df['prev_close'] * 100)

        # 전일 윗꼬리 (upper shadow)
        df['prev_upper_shadow'] = (
            (df['prev_high'] - df[['prev_open', 'prev_close']].max(axis=1)) / df['prev_close']
        )

        # 전일 아래꼬리 (lower shadow)
        df['prev_lower_shadow'] = (
            (df[['prev_open', 'prev_close']].min(axis=1) - df['prev_low']) / df['prev_close']
        )

        return df

    def calculate_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        거래량 Feature 계산

        Args:
            df: 가격 데이터 DataFrame

        Returns:
            거래량 Feature가 추가된 DataFrame
        """
        # 20일 평균 거래량
        df['avg_volume_20d'] = df.groupby('InfoCode')['volume'].transform(
            lambda x: x.rolling(window=20, min_periods=1).mean()
        )

        # 전일 거래량 비율
        df['volume_ratio'] = df['prev_volume'] / df['avg_volume_20d']

        return df

    def add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        시간 Features 추가

        Args:
            df: DataFrame with 'date' column

        Returns:
            시간 Features가 추가된 DataFrame
        """
        # 날짜를 datetime으로 변환
        df['date'] = pd.to_datetime(df['date'])

        # 요일 (0=Monday, 4=Friday)
        df['day_of_week'] = df['date'].dt.dayofweek

        # 월 (1-12)
        df['month'] = df['date'].dt.month

        # 분기 (1-4)
        df['quarter'] = df['date'].dt.quarter

        # 연도
        df['year'] = df['date'].dt.year

        # 월초 여부 (첫 3거래일) - 종목별로 계산
        df['day_of_month'] = df.groupby(['InfoCode', df['date'].dt.to_period('M')]).cumcount() + 1
        df['is_month_start'] = (df['day_of_month'] <= 3).astype(int)

        # 월말 여부 (마지막 3거래일) - 종목별로 계산
        df['days_in_month'] = df.groupby(['InfoCode', df['date'].dt.to_period('M')])['date'].transform('count')
        df['is_month_end'] = (df['day_of_month'] > df['days_in_month'] - 3).astype(int)

        # 분기말 여부 (마지막 3거래일)
        df['days_in_quarter'] = df.groupby(['InfoCode', df['date'].dt.to_period('Q')])['date'].transform('count')
        df['day_of_quarter'] = df.groupby(['InfoCode', df['date'].dt.to_period('Q')]).cumcount() + 1
        df['is_quarter_end'] = (df['day_of_quarter'] > df['days_in_quarter'] - 3).astype(int)

        # 임시 컬럼 제거
        df = df.drop(columns=['day_of_month', 'days_in_month', 'day_of_quarter', 'days_in_quarter'])

        return df

    def add_target_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        타겟 변수 생성

        Args:
            df: 전처리된 DataFrame

        Returns:
            타겟 변수가 추가된 DataFrame
        """
        # 분류 타겟: 상승(1) / 하락(0)
        df['target_direction'] = (df['intraday_return'] > 0).astype(int)

        # 회귀 타겟: 수익률 (%)
        df['target_return'] = df['intraday_return']

        return df

    def merge_market_cap(self, df: pd.DataFrame, market_cap_df: pd.DataFrame) -> pd.DataFrame:
        """
        시가총액 데이터 병합

        Args:
            df: 가격 데이터 DataFrame
            market_cap_df: 시가총액 DataFrame (InfoCode, date, market_cap)

        Returns:
            시가총액이 병합된 DataFrame
        """
        if market_cap_df.empty:
            df['market_cap'] = np.nan
            return df

        # 날짜를 datetime으로 통일
        df['date'] = pd.to_datetime(df['date'])
        market_cap_df['date'] = pd.to_datetime(market_cap_df['date'])

        # 병합 (left join)
        df = df.merge(
            market_cap_df[['InfoCode', 'date', 'market_cap']],
            on=['InfoCode', 'date'],
            how='left'
        )

        return df

    def merge_stock_info(self, df: pd.DataFrame, stock_list: pd.DataFrame) -> pd.DataFrame:
        """
        종목 정보 병합

        Args:
            df: 가격 데이터 DataFrame
            stock_list: 종목 리스트 DataFrame

        Returns:
            종목 정보가 병합된 DataFrame
        """
        # 필요한 컬럼만 선택 (stock_list에 있는 컬럼만)
        base_cols = ['InfoCode', 'DsCode', 'DsQtName', 'Exchange', 'ExchMnem',
                     'Ticker', 'ISIN', 'Region', 'StatusCode', 'DelistDate',
                     'sector', 'industry']

        # stock_list에 실제로 존재하는 컬럼만 선택
        cols = [c for c in base_cols if c in stock_list.columns]

        stock_info = stock_list[cols].copy()

        # 병합 (left join)
        df = df.merge(stock_info, on='InfoCode', how='left')

        return df

    def apply_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        필터링 조건 적용

        Args:
            df: 전처리된 DataFrame

        Returns:
            필터링된 DataFrame
        """
        filters_config = self.data_config.get('filters', {})

        # 최소 주가 필터
        min_price = filters_config.get('min_price', 5.0)
        if min_price > 0:
            df = df[df['close'] >= min_price].copy()

        # 최소 시가총액 필터
        min_market_cap = filters_config.get('min_market_cap', 500_000_000)
        if min_market_cap > 0 and 'market_cap' in df.columns:
            df = df[
                (df['market_cap'] >= min_market_cap) | (df['market_cap'].isna())
            ].copy()

        # 최소 평균 거래량 필터
        min_avg_volume = filters_config.get('min_avg_volume', 100_000)
        if min_avg_volume > 0 and 'avg_volume_20d' in df.columns:
            df = df[
                (df['avg_volume_20d'] >= min_avg_volume) | (df['avg_volume_20d'].isna())
            ].copy()

        # 갭 범위 필터 (학습 시 적용할 수도 있음)
        gap_config = self.data_config.get('gap', {})
        max_gap_pct = gap_config.get('max_gap_pct', 50.0)
        if 'gap_pct' in df.columns:
            df = df[
                (df['gap_pct'].isna()) | (df['gap_pct'] <= max_gap_pct)
            ].copy()

        return df

    def create_preprocessed_df(
        self,
        ohlcv_df: pd.DataFrame,
        stock_list: pd.DataFrame,
        market_cap_df: Optional[pd.DataFrame] = None,
        use_parallel: bool = True
    ) -> pd.DataFrame:
        """
        최종 preprocessed_df 생성

        Args:
            ohlcv_df: OHLCV DataFrame
            stock_list: 종목 리스트 DataFrame
            market_cap_df: 시가총액 DataFrame (optional)
            use_parallel: 병렬 처리 사용 여부

        Returns:
            전처리 완료된 DataFrame
        """
        print("\n=== 데이터 전처리 시작 ===")

        if use_parallel:
            # 병렬 처리 사용
            print("1-5. Processing features in parallel (lag, gap, returns, volume, targets)...")
            df = self.process_stocks_parallel(ohlcv_df)
            print(f"   ✓ Shape: {df.shape}")
            print(f"   ✓ Gap up events: {df['is_gap_up'].sum():,}")
        else:
            # 순차 처리 (기존 방식)
            print("1. Adding lag features...")
            df = self.add_lag_features(ohlcv_df)
            print(f"   ✓ Shape: {df.shape}")

            print("2. Calculating gap...")
            df = self.calculate_gap(df)
            print(f"   ✓ Gap up events: {df['is_gap_up'].sum():,}")

            print("3. Calculating returns...")
            df = self.calculate_returns(df)

            print("4. Calculating volume features...")
            df = self.calculate_volume_features(df)

            print("5. Adding target variables...")
            df = self.add_target_variables(df)

        # 6. 시간 Features 추가
        print("6. Adding time features...")
        df = self.add_time_features(df)
        print(f"   ✓ Added: day_of_week, month, quarter, is_month_start, is_month_end, is_quarter_end")

        # 7. 시가총액 병합
        if market_cap_df is not None and not market_cap_df.empty:
            print("7. Merging market cap data...")
            df = self.merge_market_cap(df, market_cap_df)
            print(f"   ✓ Market cap coverage: {df['market_cap'].notna().sum()}/{len(df)} "
                  f"({df['market_cap'].notna().sum()/len(df)*100:.1f}%)")
        else:
            print("7. Skipping market cap (no data)")
            df['market_cap'] = np.nan

        # 8. 종목 정보 병합
        print("8. Merging stock info...")
        df = self.merge_stock_info(df, stock_list)

        # 9. 필터링 적용
        print("9. Applying filters...")
        initial_count = len(df)
        df = self.apply_filters(df)
        print(f"   ✓ Filtered: {initial_count:,} -> {len(df):,} "
              f"({len(df)/initial_count*100:.1f}%)")

        # 10. 컬럼 정리 및 순서 조정
        print("10. Organizing columns...")
        df = self._organize_columns(df)

        print(f"\n=== 전처리 완료 ===")
        print(f"Final shape: {df.shape}")
        print(f"Date range: {df['date'].min()} ~ {df['date'].max()}")
        print(f"Unique stocks: {df['InfoCode'].nunique()}")

        return df

    def _organize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        컬럼 정리 및 순서 조정

        Args:
            df: DataFrame

        Returns:
            정리된 DataFrame
        """
        # 기본 컬럼 순서 정의
        base_cols = [
            # 기본 정보
            'InfoCode', 'DsCode', 'date', 'Ticker', 'ISIN',
            'DsQtName', 'Exchange', 'ExchMnem', 'Region',
            'sector', 'industry',

            # 당일 OHLCV
            'open', 'high', 'low', 'close', 'volume',

            # 전일 정보
            'prev_open', 'prev_high', 'prev_low', 'prev_close', 'prev_volume',

            # 갭 정보
            'gap_pct', 'is_gap_up', 'gap_size_category',

            # 수익률
            'intraday_return', 'prev_return', 'prev_range_pct',
            'prev_upper_shadow', 'prev_lower_shadow',

            # 거래량
            'avg_volume_20d', 'volume_ratio',

            # 시간 Features
            'day_of_week', 'month', 'quarter', 'year',
            'is_month_start', 'is_month_end', 'is_quarter_end',

            # 타겟 변수
            'target_direction', 'target_return',

            # 메타 정보
            'market_cap', 'StatusCode', 'DelistDate',
        ]

        # 존재하는 컬럼만 선택
        cols = [c for c in base_cols if c in df.columns]

        # 나머지 컬럼 추가
        remaining_cols = [c for c in df.columns if c not in cols]
        cols.extend(remaining_cols)

        return df[cols].copy()

    def save_to_file(self, df: pd.DataFrame, filepath: str):
        """
        DataFrame을 파일로 저장

        Args:
            df: DataFrame
            filepath: 저장 경로
        """
        from pathlib import Path

        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        if filepath.endswith('.parquet'):
            df.to_parquet(filepath, index=False)
        elif filepath.endswith('.csv'):
            df.to_csv(filepath, index=False)
        else:
            raise ValueError(f"Unsupported file format: {filepath}")

        print(f"\n✓ Saved to: {filepath}")
        print(f"  File size: {path.stat().st_size / 1024 / 1024:.2f} MB")
