"""
데이터 전처리 모듈

Raw 데이터를 전처리하여 preprocessed_df 생성
"""

from typing import Optional, Dict, List
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import pandas as pd
import numpy as np


# IQR 기반 이상치 제거 대상 컬럼 (연속형 feature + target)
CONTINUOUS_FEATURE_COLS = [
    # 갭 관련
    'gap_pct',
    
    # 전일 패턴
    'prev_return', 'prev_range_pct', 'prev_upper_shadow', 'prev_lower_shadow',
    
    # 거래량
    'volume_ratio',
    
    # 기술적 지표
    'rsi_14', 'atr_14', 'atr_ratio', 'bollinger_position',
    'return_5d', 'return_20d',
    
    # 시장 컨텍스트
    'market_gap_diff',
]

TARGET_COLS = [
    'target_return', 'target_max_return'
]


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
        stock_data['intraday_max_return'] = ((stock_data['high'] - stock_data['open']) / stock_data['open'] * 100)
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
        stock_data['target_max_return'] = stock_data['intraday_max_return']

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

        # 당일 최대 수익률 (시가 -> 고가)
        df['intraday_max_return'] = ((df['high'] - df['open']) / df['open'] * 100)

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

        # 회귀 타겟: 최대 수익률 (고가 기준, 익절 전략용)
        df['target_max_return'] = df['intraday_max_return']

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
        initial_count = len(df)
        filters_config = self.data_config.get('filters', {})

        # 최소 주가 필터
        min_price = filters_config.get('min_price', 5.0)
        if min_price > 0:
            before = len(df)
            df = df[df['close'] >= min_price].copy()
            print(f"   - 최소 주가 (${min_price}): {before:,} → {len(df):,}")

        # 최소 시가총액 필터
        min_market_cap = filters_config.get('min_market_cap', 300_000_000)
        exclude_missing = filters_config.get('exclude_missing_market_cap', True)

        if min_market_cap > 0 and 'market_cap' in df.columns:
            before = len(df)
            if exclude_missing:
                # NaN 제외
                df = df[
                    (df['market_cap'] >= min_market_cap) & (df['market_cap'].notna())
                ].copy()
            else:
                # NaN 포함
                df = df[
                    (df['market_cap'] >= min_market_cap) | (df['market_cap'].isna())
                ].copy()
            print(f"   - 시가총액 (${min_market_cap/1e6:.0f}M): {before:,} → {len(df):,}")

        # 최소 평균 거래량 필터
        min_avg_volume = filters_config.get('min_avg_volume', 100_000)
        if min_avg_volume > 0 and 'avg_volume_20d' in df.columns:
            before = len(df)
            df = df[
                (df['avg_volume_20d'] >= min_avg_volume) | (df['avg_volume_20d'].isna())
            ].copy()
            print(f"   - 평균 거래량 ({min_avg_volume:,}): {before:,} → {len(df):,}")

        # 갭 상승만 필터링 (프로젝트 핵심 조건)
        gap_config = self.data_config.get('gap', {})
        min_gap_pct = gap_config.get('min_gap_pct', 0.0)
        max_gap_pct = gap_config.get('max_gap_pct', 50.0)
        if 'gap_pct' in df.columns and 'is_gap_up' in df.columns:
            before = len(df)
            df = df[
                (df['is_gap_up'] == 1) &
                (df['gap_pct'] >= min_gap_pct) &
                (df['gap_pct'] <= max_gap_pct)
            ].copy()
            print(f"   - 갭 상승 ({min_gap_pct}%-{max_gap_pct}%): {before:,} → {len(df):,}")

        # 기술적 지표 필터링
        tech_config = self.data_config.get('technical_filters', {})
        if tech_config.get('enable', False):
            print("   기술적 지표 필터:")

            # RSI 필터
            rsi_config = tech_config.get('rsi', {})
            if 'rsi_14' in df.columns:
                before = len(df)
                rsi_min = rsi_config.get('min', 20)
                rsi_max = rsi_config.get('max', 80)
                exclude_na = rsi_config.get('exclude_na', False)

                if exclude_na:
                    df = df[
                        (df['rsi_14'] >= rsi_min) &
                        (df['rsi_14'] <= rsi_max) &
                        (df['rsi_14'].notna())
                    ].copy()
                else:
                    df = df[
                        ((df['rsi_14'] >= rsi_min) & (df['rsi_14'] <= rsi_max)) |
                        (df['rsi_14'].isna())
                    ].copy()
                print(f"     RSI ({rsi_min}-{rsi_max}): {before:,} → {len(df):,}")

            # 볼린저밴드 포지션 필터
            bb_config = tech_config.get('bollinger', {})
            if 'bollinger_position' in df.columns:
                before = len(df)
                bb_min = bb_config.get('min_position', 0.2)
                bb_max = bb_config.get('max_position', 0.8)
                exclude_na = bb_config.get('exclude_na', False)

                if exclude_na:
                    df = df[
                        (df['bollinger_position'] >= bb_min) &
                        (df['bollinger_position'] <= bb_max) &
                        (df['bollinger_position'].notna())
                    ].copy()
                else:
                    df = df[
                        ((df['bollinger_position'] >= bb_min) & (df['bollinger_position'] <= bb_max)) |
                        (df['bollinger_position'].isna())
                    ].copy()
                print(f"     볼린저밴드 ({bb_min}-{bb_max}): {before:,} → {len(df):,}")

        return df

    def create_preprocessed_df(
        self,
        ohlcv_df: pd.DataFrame,
        stock_list: pd.DataFrame,
        market_cap_df: Optional[pd.DataFrame] = None,
        use_parallel: bool = True,
        batch_size: Optional[int] = None,
        save_batches: bool = False,
        batch_output_dir: Optional[str] = None
    ) -> pd.DataFrame:
        """
        최종 preprocessed_df 생성 (배치 처리 지원)

        Args:
            ohlcv_df: OHLCV DataFrame
            stock_list: 종목 리스트 DataFrame
            market_cap_df: 시가총액 DataFrame (optional)
            use_parallel: 병렬 처리 사용 여부
            batch_size: 배치 크기 (None이면 전체를 한 번에 처리)
            save_batches: 배치 파일을 저장할지 여부
            batch_output_dir: 배치 파일 저장 디렉토리

        Returns:
            전처리 완료된 DataFrame
        """
        import gc
        from pathlib import Path

        # 배치 처리가 필요한지 확인
        if batch_size is None or len(ohlcv_df['InfoCode'].unique()) <= batch_size:
            # 배치 처리 없이 전체 처리
            return self._process_full(ohlcv_df, stock_list, market_cap_df, use_parallel)

        # 배치 처리
        print("\n=== 배치 처리 모드 ===")
        print(f"   총 종목 수: {len(stock_list):,}")
        print(f"   배치 크기: {batch_size}")

        if save_batches and batch_output_dir:
            output_dir = Path(batch_output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            print(f"   배치 저장: {output_dir}")

        # 종목별로 배치 나누기
        info_codes = stock_list['InfoCode'].unique()
        total_batches = (len(info_codes) + batch_size - 1) // batch_size

        results = []
        for i in range(0, len(info_codes), batch_size):
            batch_num = i // batch_size + 1
            batch_info_codes = info_codes[i:i+batch_size]

            print(f"\n{'='*80}")
            print(f"Batch {batch_num}/{total_batches} ({len(batch_info_codes)} stocks)")
            print(f"{'='*80}")

            try:
                # 배치 데이터 추출
                batch_ohlcv = ohlcv_df[ohlcv_df['InfoCode'].isin(batch_info_codes)].copy()
                batch_stocks = stock_list[stock_list['InfoCode'].isin(batch_info_codes)].copy()

                if batch_ohlcv.empty:
                    print("   ✗ No data for this batch")
                    continue

                # 배치 처리
                batch_df = self._process_full(batch_ohlcv, batch_stocks, market_cap_df, use_parallel)

                if not batch_df.empty:
                    print(f"   ✓ Processed: {batch_df.shape}")

                    # 배치 파일 저장 (선택)
                    if save_batches and batch_output_dir:
                        batch_file = output_dir / f"batch_{batch_num:04d}.parquet"
                        batch_df.to_parquet(batch_file, index=False)
                        print(f"   ✓ Saved: {batch_file.name}")

                    results.append(batch_df)

                # 메모리 해제
                del batch_ohlcv, batch_stocks, batch_df
                gc.collect()

            except Exception as e:
                print(f"   ✗ Error: {e}")
                import traceback
                traceback.print_exc()
                continue

        # 모든 배치 병합
        if results:
            print(f"\n{'='*80}")
            print(f"Merging {len(results)} batches...")
            print(f"{'='*80}")
            df_final = pd.concat(results, ignore_index=True)

            print(f"\n=== 배치 처리 완료 ===")
            print(f"Final shape: {df_final.shape}")
            print(f"Date range: {df_final['date'].min()} ~ {df_final['date'].max()}")
            print(f"Unique stocks: {df_final['InfoCode'].nunique()}")

            return df_final
        else:
            print("\n✗ No data processed")
            return pd.DataFrame()

    def _process_full(
        self,
        ohlcv_df: pd.DataFrame,
        stock_list: pd.DataFrame,
        market_cap_df: Optional[pd.DataFrame] = None,
        use_parallel: bool = True
    ) -> pd.DataFrame:
        """
        전체 데이터 처리 (내부 메서드)

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

        # 7. 기술적 지표 추가 (RSI, ATR, 볼린저밴드 등)
        print("7. Adding technical indicators...")
        from src.features import add_technical_features_parallel
        max_workers = self.data_config.get('parallel', {}).get('max_workers', 10)
        df = add_technical_features_parallel(df, max_workers=max_workers)
        print(f"   ✓ Added: MA, RSI, ATR, Bollinger Bands, Momentum")

        # 8. 시가총액 병합
        if market_cap_df is not None and not market_cap_df.empty:
            print("8. Merging market cap data...")
            df = self.merge_market_cap(df, market_cap_df)
            print(f"   ✓ Market cap coverage: {df['market_cap'].notna().sum()}/{len(df)} "
                  f"({df['market_cap'].notna().sum()/len(df)*100:.1f}%)")
        else:
            print("8. Skipping market cap (no data)")
            df['market_cap'] = np.nan

        # 9. 종목 정보 병합
        print("9. Merging stock info...")
        df = self.merge_stock_info(df, stock_list)

        # 10. 필터링 적용 (기술적 지표 포함)
        print("10. Applying filters...")
        initial_count = len(df)
        df = self.apply_filters(df)
        print(f"   ✓ Filtered: {initial_count:,} -> {len(df):,} "
              f"({len(df)/initial_count*100:.1f}%)")

        # 11. 컬럼 정리 및 순서 조정
        print("11. Organizing columns...")
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

    def process_stocks_incrementally(
        self,
        stock_list: pd.DataFrame,
        data_loader,
        market_indices: pd.DataFrame,
        start_date,
        batch_size: int = 10,
        save_batch_files: bool = True,
        batch_output_dir: Optional[str] = None,
        final_output: Optional[str] = None,
        apply_outlier_fix: Optional[bool] = False
    ) -> pd.DataFrame:
        """
        종목별 점진적 처리 (메모리 효율적)

        종목을 작은 배치로 나눠서:
        1. OHLCV 로드
        2. 전처리
        3. 기술적 지표 추가
        4. 시장 컨텍스트 추가
        5. 파일 저장 (선택)
        6. 메모리 해제

        Args:
            stock_list: 전체 종목 리스트
            data_loader: DataLoader 인스턴스
            market_indices: 시장 지수 데이터 (미리 로드된 것)
            start_date: 시작 날짜
            batch_size: 배치 크기 (종목 수)
            save_batch_files: 배치 파일 저장 여부
            batch_output_dir: 배치 파일 저장 디렉토리
            final_output: 최종 파일 경로

        Returns:
            최종 병합된 DataFrame
        """
        import gc
        from pathlib import Path
        from tqdm import tqdm
        from src.features import add_technical_features_parallel, add_market_context

        print(f"\n{'='*80}")
        print(f"종목별 점진적 처리 시작")
        print(f"{'='*80}")
        print(f"   총 종목 수: {len(stock_list):,}")
        print(f"   배치 크기: {batch_size}")
        print(f"   데이터 기간: {start_date.date()} ~")

        # 배치 파일 저장 설정
        batch_files = []
        if save_batch_files and batch_output_dir:
            output_dir = Path(batch_output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            print(f"   배치 저장: {output_dir}")

        # 배치 개수 계산
        total_batches = (len(stock_list) + batch_size - 1) // batch_size

        # 배치별 처리
        for i in range(0, len(stock_list), batch_size):
            batch_num = i // batch_size + 1

            print(f"\n{'='*80}")
            print(f"Batch {batch_num}/{total_batches}")
            print(f"{'='*80}")

            try:
                # 1. 배치 종목 추출
                batch_stocks = stock_list.iloc[i:i+batch_size].copy()
                print(f"   종목 수: {len(batch_stocks)}")

                # 2. 섹터/산업 정보 로드
                sector_info = data_loader.load_sector_industry(batch_stocks['InfoCode'].tolist())
                batch_stocks_with_sector = batch_stocks.merge(sector_info, on='InfoCode', how='left')

                # 3. OHLCV 데이터 로드 (이 배치만)
                ohlcv_batch = data_loader.load_ohlcv_batch(
                    batch_stocks_with_sector,
                    start_date,
                    adj_type=2,
                    batch_size=10  # DB 조회 배치
                )

                if ohlcv_batch.empty:
                    print("   ✗ OHLCV 데이터 없음")
                    continue

                print(f"   ✓ OHLCV: {len(ohlcv_batch):,} 레코드")

                # 3-1. 시가총액 데이터 로드 (이 배치만)
                market_cap_batch = data_loader.load_market_cap(
                    batch_stocks['InfoCode'].tolist(),
                    start_date
                )

                if not market_cap_batch.empty:
                    print(f"   ✓ 시가총액: {len(market_cap_batch):,} 레코드")
                else:
                    print("   ⚠ 시가총액 데이터 없음")

                # 4. 전처리
                df_preprocessed = self._process_full(
                    ohlcv_df=ohlcv_batch,
                    stock_list=batch_stocks_with_sector,
                    market_cap_df=market_cap_batch if not market_cap_batch.empty else None,
                    use_parallel=True
                )

                if df_preprocessed.empty:
                    print("   ✗ 전처리 후 데이터 없음")
                    continue

                # 5. 기술적 지표 추가
                print("   기술적 지표 추가...")
                df_with_technical = add_technical_features_parallel(df_preprocessed, max_workers=4)

                # 6. 시장 컨텍스트 추가
                print("   시장 컨텍스트 추가...")
                df_final = add_market_context(df_with_technical, market_indices)

                print(f"   ✓ 처리 완료: {df_final.shape}")

                # 7. 배치 파일 저장
                if save_batch_files and batch_output_dir:
                    batch_file = output_dir / f"batch_{batch_num:04d}.parquet"
                    df_final.to_parquet(batch_file, index=False)
                    batch_files.append(batch_file)
                    print(f"   ✓ 저장: {batch_file.name}")

                # 8. 메모리 해제
                del batch_stocks, batch_stocks_with_sector, sector_info
                del ohlcv_batch, market_cap_batch, df_preprocessed, df_with_technical, df_final
                gc.collect()

            except Exception as e:
                print(f"   ✗ 오류: {e}")
                import traceback
                traceback.print_exc()
                continue

        # 9. 모든 배치 병합
        if batch_files:
            print(f"\n{'='*80}")
            print(f"배치 병합 중... ({len(batch_files)}개)")
            print(f"{'='*80}")

            dfs = []
            for batch_file in tqdm(batch_files, desc="Loading batches"):
                df = pd.read_parquet(batch_file)
                dfs.append(df)

            df_final = pd.concat(dfs, ignore_index=True)
            
            # 10. IQR 이상치 제거 (옵션)
            if apply_outlier_fix:
                print(f"\n{'='*80}")
                print(f"IQR 이상치 제거 중...")
                print(f"{'='*80}")
                df_final = self.remove_outliers_iqr(df_final, multiplier=1.5, verbose=True)

            # 최종 파일 저장
            if final_output:
                if apply_outlier_fix:
                    file_name = "/preprocessed_df_full_outlier_fix.parquet"
                    final_output = final_output + file_name
                    print(f"\n   최종 파일 저장 중... (IQR Outlier Fix 적용)")
                    self.save_to_file(df_final, final_output)
                else:    
                    file_name = "/preprocessed_df_full.parquet"
                    final_output = final_output + file_name
                    print(f"\n   최종 파일 저장 중...")
                    self.save_to_file(df_final, final_output)

            # 배치 파일 삭제
            print(f"\n   배치 파일 삭제 중...")
            for batch_file in batch_files:
                batch_file.unlink()
            print(f"   ✓ {len(batch_files)} 파일 삭제 완료")

            print(f"\n{'='*80}")
            print(f"완료!")
            print(f"{'='*80}")
            print(f"   최종 Shape: {df_final.shape}")
            print(f"   종목 수: {df_final['InfoCode'].nunique():,}")
            print(f"   날짜 범위: {df_final['date'].min()} ~ {df_final['date'].max()}")
            print(f"   갭 상승 이벤트: {df_final['is_gap_up'].sum():,}")

            return df_final
        else:
            print("\n✗ 처리된 배치 없음")
            return pd.DataFrame()

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

    def remove_outliers_iqr(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        multiplier: float = 1.5,
        verbose: bool = True
    ) -> pd.DataFrame:
        """
        IQR (Interquartile Range) 방법으로 이상치 제거
        
        공식:
        - Q1 = 25th percentile
        - Q3 = 75th percentile
        - IQR = Q3 - Q1
        - Lower Bound = Q1 - multiplier * IQR
        - Upper Bound = Q3 + multiplier * IQR
        
        Args:
            df: DataFrame
            columns: 이상치 제거할 컬럼 리스트 (None이면 기본 연속형 컬럼 사용)
            multiplier: IQR 배수 (기본값 1.5, 더 엄격하려면 3.0 사용)
            verbose: 상세 출력 여부
            
        Returns:
            이상치가 제거된 DataFrame
        """
        if columns is None:
            # 기본 연속형 feature + target 컬럼 사용
            columns = CONTINUOUS_FEATURE_COLS + TARGET_COLS
        
        # 실제로 존재하는 컬럼만 선택
        columns = [c for c in columns if c in df.columns]
        
        if not columns:
            print("   ⚠ IQR 적용 가능한 컬럼이 없습니다.")
            return df
        
        initial_count = len(df)
        df_clean = df.copy()
        
        if verbose:
            print(f"\n   === IQR 이상치 제거 (multiplier={multiplier}) ===")
            print(f"   적용 컬럼: {len(columns)}개")
        
        # 각 컬럼별로 이상치 마스크 생성
        outlier_mask = pd.Series(False, index=df_clean.index)
        col_stats = []
        
        for col in columns:
            if df_clean[col].isna().all():
                continue
                
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - multiplier * IQR
            upper_bound = Q3 + multiplier * IQR
            
            # 이상치 판별
            col_outliers = (df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)
            outlier_count = col_outliers.sum()
            
            if outlier_count > 0:
                col_stats.append({
                    'column': col,
                    'Q1': Q1,
                    'Q3': Q3,
                    'IQR': IQR,
                    'lower': lower_bound,
                    'upper': upper_bound,
                    'outliers': outlier_count,
                    'pct': outlier_count / len(df_clean) * 100
                })
            
            # 전체 이상치 마스크에 합치기
            outlier_mask = outlier_mask | col_outliers
        
        # 이상치 제거
        df_clean = df_clean[~outlier_mask].copy()
        
        final_count = len(df_clean)
        removed_count = initial_count - final_count
        
        if verbose:
            print(f"\n   컬럼별 이상치 통계:")
            for stat in sorted(col_stats, key=lambda x: x['outliers'], reverse=True)[:10]:
                print(f"   - {stat['column']}: {stat['outliers']:,}개 ({stat['pct']:.2f}%) "
                      f"[범위: {stat['lower']:.2f} ~ {stat['upper']:.2f}]")
            
            if len(col_stats) > 10:
                print(f"   ... 외 {len(col_stats) - 10}개 컬럼")
            
            print(f"\n   ✓ 이상치 제거: {initial_count:,} → {final_count:,} "
                  f"({removed_count:,}개 제거, {removed_count/initial_count*100:.2f}%)")
        
        return df_clean

    def clip_outliers_iqr(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        multiplier: float = 1.5,
        verbose: bool = True
    ) -> pd.DataFrame:
        """
        IQR 방법으로 이상치를 경계값으로 클리핑 (제거 대신)
        
        Args:
            df: DataFrame
            columns: 클리핑할 컬럼 리스트 (None이면 기본 연속형 컬럼 사용)
            multiplier: IQR 배수 (기본값 1.5)
            verbose: 상세 출력 여부
            
        Returns:
            이상치가 클리핑된 DataFrame
        """
        if columns is None:
            columns = CONTINUOUS_FEATURE_COLS + TARGET_COLS
        
        columns = [c for c in columns if c in df.columns]
        
        if not columns:
            print("   ⚠ IQR 적용 가능한 컬럼이 없습니다.")
            return df
        
        df_clipped = df.copy()
        
        if verbose:
            print(f"\n   === IQR 이상치 클리핑 (multiplier={multiplier}) ===")
            print(f"   적용 컬럼: {len(columns)}개")
        
        clipped_stats = []
        
        for col in columns:
            if df_clipped[col].isna().all():
                continue
                
            Q1 = df_clipped[col].quantile(0.25)
            Q3 = df_clipped[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - multiplier * IQR
            upper_bound = Q3 + multiplier * IQR
            
            # 클리핑 전 이상치 수 계산
            outliers_before = ((df_clipped[col] < lower_bound) | (df_clipped[col] > upper_bound)).sum()
            
            if outliers_before > 0:
                # 클리핑 적용
                df_clipped[col] = df_clipped[col].clip(lower=lower_bound, upper=upper_bound)
                
                clipped_stats.append({
                    'column': col,
                    'lower': lower_bound,
                    'upper': upper_bound,
                    'clipped': outliers_before,
                    'pct': outliers_before / len(df_clipped) * 100
                })
        
        if verbose and clipped_stats:
            print(f"\n   컬럼별 클리핑 통계:")
            for stat in sorted(clipped_stats, key=lambda x: x['clipped'], reverse=True)[:10]:
                print(f"   - {stat['column']}: {stat['clipped']:,}개 ({stat['pct']:.2f}%) "
                      f"[범위: {stat['lower']:.2f} ~ {stat['upper']:.2f}]")
            
            if len(clipped_stats) > 10:
                print(f"   ... 외 {len(clipped_stats) - 10}개 컬럼")
            
            total_clipped = sum(s['clipped'] for s in clipped_stats)
            print(f"\n   ✓ 총 {total_clipped:,}개 값 클리핑 완료")
        
        return df_clipped

    def get_outlier_summary(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        multiplier: float = 1.5
    ) -> pd.DataFrame:
        """
        이상치 현황 요약 (제거하지 않고 확인만)
        
        Args:
            df: DataFrame
            columns: 분석할 컬럼 리스트
            multiplier: IQR 배수
            
        Returns:
            이상치 요약 DataFrame
        """
        if columns is None:
            columns = CONTINUOUS_FEATURE_COLS + TARGET_COLS
        
        columns = [c for c in columns if c in df.columns]
        
        stats = []
        for col in columns:
            if df[col].isna().all():
                continue
                
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - multiplier * IQR
            upper_bound = Q3 + multiplier * IQR
            
            outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
            
            stats.append({
                'column': col,
                'Q1': Q1,
                'median': df[col].median(),
                'Q3': Q3,
                'IQR': IQR,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'min': df[col].min(),
                'max': df[col].max(),
                'outliers': outliers,
                'outlier_pct': outliers / len(df) * 100,
                'na_count': df[col].isna().sum()
            })
        
        return pd.DataFrame(stats).sort_values('outlier_pct', ascending=False)
