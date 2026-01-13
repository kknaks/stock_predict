"""
데이터 로더 모듈

DB에서 필요한 데이터를 로드하고 병합하는 기능 제공
"""

import yaml
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

import pandas as pd
from sqlalchemy import and_, or_, func
from sqlalchemy.orm import Session

from src.data.db_connector import DatabaseConnector, SessionManager
from src.data.models import (
    DS2CtryQtInfo,
    DS2Exchange,
    DS2PrimQtPrc,
    Ds2MnemChg,
    VwDs2Pricing,
    VwDs2MktCap,
    RKDFndCmpInd,
    DS2EquityIndex,
    DS2IndexData,
)


def load_config(config_path: str = "config/config.yaml") -> dict:
    """설정 파일 로드"""
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_file, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


class DataLoader:
    """데이터 로더 클래스"""

    def __init__(self, config_path: str = "config/config.yaml"):
        """
        초기화

        Args:
            config_path: 설정 파일 경로
        """
        self.config = load_config(config_path)
        self.data_config = self.config.get('data', {})
        self.db = DatabaseConnector()

    def get_target_exchanges(self, session: Session) -> List[int]:
        """
        대상 거래소 코드 조회

        Args:
            session: DB 세션

        Returns:
            거래소 코드 리스트
        """
        exchange_names = self.data_config.get('filters', {}).get('exchanges', ['NYSE', 'NASDAQ', 'AMEX'])
        region = self.data_config.get('filters', {}).get('region', 'US')

        # 거래소 필터링
        filters = [DS2Exchange.ExchCtryCode == region]

        # 각 거래소명으로 LIKE 조건 생성
        name_filters = [DS2Exchange.ExchName.like(f'%{name}%') for name in exchange_names]
        filters.append(or_(*name_filters))

        exchanges = session.query(DS2Exchange.ExchIntCode).filter(and_(*filters)).all()

        return [exch.ExchIntCode for exch in exchanges]

    def load_sector_industry(self, info_codes: List[int]) -> pd.DataFrame:
        """
        종목별 섹터/산업 정보 로드

        Args:
            info_codes: InfoCode 리스트

        Returns:
            섹터/산업 정보 DataFrame (InfoCode, sector, industry)
        """
        with SessionManager(self.db) as session:
            # 섹터 정보 (TxoTypeCode = 7: MGSECTOR)
            sectors = session.query(
                RKDFndCmpInd.Code.label('InfoCode'),
                RKDFndCmpInd.Desc_.label('sector'),
            ).filter(
                and_(
                    RKDFndCmpInd.Code.in_(info_codes),
                    RKDFndCmpInd.TxoTypeCode == 7,  # MGSECTOR
                    RKDFndCmpInd.TxoOrder == 1,  # 첫 번째 항목만
                )
            ).all()

            # 산업 정보 (TxoTypeCode = 10: NAICS)
            industries = session.query(
                RKDFndCmpInd.Code.label('InfoCode'),
                RKDFndCmpInd.Desc_.label('industry'),
            ).filter(
                and_(
                    RKDFndCmpInd.Code.in_(info_codes),
                    RKDFndCmpInd.TxoTypeCode == 10,  # NAICS
                    RKDFndCmpInd.TxoOrder == 1,  # 첫 번째 항목만
                )
            ).all()

            # DataFrame 변환
            df_sector = pd.DataFrame([
                {'InfoCode': s.InfoCode, 'sector': s.sector}
                for s in sectors
            ])

            df_industry = pd.DataFrame([
                {'InfoCode': i.InfoCode, 'industry': i.industry}
                for i in industries
            ])

            # 병합
            if not df_sector.empty and not df_industry.empty:
                result = df_sector.merge(df_industry, on='InfoCode', how='outer')
            elif not df_sector.empty:
                result = df_sector
                result['industry'] = None
            elif not df_industry.empty:
                result = df_industry
                result['sector'] = None
            else:
                result = pd.DataFrame(columns=['InfoCode', 'sector', 'industry'])

            return result

    def get_stock_list(self) -> pd.DataFrame:
        """
        종목 리스트 조회

        Returns:
            종목 리스트 DataFrame
        """
        with SessionManager(self.db) as session:
            # 대상 거래소 코드
            target_exch_codes = self.get_target_exchanges(session)

            region = self.data_config.get('filters', {}).get('region', 'US')

            # 서브쿼리: 각 InfoCode의 최신 Ticker (MAX(EndDate))
            max_enddate_subq = (
                session.query(
                    Ds2MnemChg.InfoCode,
                    func.max(Ds2MnemChg.EndDate).label('max_enddate')
                )
                .group_by(Ds2MnemChg.InfoCode)
                .subquery()
            )

            # region에 따라 Ticker 컬럼 결정
            # US: Ds2MnemChg.Ticker 사용
            # KR, JP: DsLocalCode에서 첫 문자 제외 (2번째부터 끝까지)
            # HK: DsLocalCode에서 앞 2문자 제외 (3번째부터 끝까지)
            if region == 'US':
                ticker_column = Ds2MnemChg.Ticker.label('Ticker')
            elif region in ('KR', 'JP'):
                # SUBSTRING(DsLocalCode, 2, LEN(DsLocalCode))
                ticker_column = func.substring(
                    DS2CtryQtInfo.DsLocalCode, 
                    2, 
                    func.len(DS2CtryQtInfo.DsLocalCode)
                ).label('Ticker')
            elif region == 'HK':
                # SUBSTRING(DsLocalCode, 3, LEN(DsLocalCode))
                ticker_column = func.substring(
                    DS2CtryQtInfo.DsLocalCode, 
                    3, 
                    func.len(DS2CtryQtInfo.DsLocalCode)
                ).label('Ticker')
            else:
                # 기타 region은 Ds2MnemChg.Ticker 사용
                ticker_column = Ds2MnemChg.Ticker.label('Ticker')

            # 종목 조회 (거래소, Ticker 정보 포함)
            base_query = session.query(
                DS2CtryQtInfo.InfoCode,
                DS2CtryQtInfo.DsCode,
                DS2CtryQtInfo.DsQtName,
                DS2CtryQtInfo.StatusCode,
                DS2CtryQtInfo.DelistDate,
                DS2CtryQtInfo.Region,
                DS2PrimQtPrc.ExchIntCode,
                DS2Exchange.ExchName,
                DS2Exchange.ExchMnem,
                ticker_column,
            ).join(
                DS2PrimQtPrc,
                DS2CtryQtInfo.InfoCode == DS2PrimQtPrc.InfoCode
            ).join(
                DS2Exchange,
                DS2PrimQtPrc.ExchIntCode == DS2Exchange.ExchIntCode
            )

            # US와 기타(Ds2MnemChg.Ticker 사용하는 경우)에만 Ds2MnemChg 조인
            if region in ('US',) or region not in ('KR', 'JP', 'HK'):
                base_query = base_query.outerjoin(
                    max_enddate_subq,
                    DS2CtryQtInfo.InfoCode == max_enddate_subq.c.InfoCode
                ).outerjoin(
                    Ds2MnemChg,
                    and_(
                        DS2CtryQtInfo.InfoCode == Ds2MnemChg.InfoCode,
                        Ds2MnemChg.EndDate == max_enddate_subq.c.max_enddate
                    )
                )

            stocks = base_query.filter(
                and_(
                    DS2CtryQtInfo.Region == region,
                    DS2CtryQtInfo.IsPrimQt == 1,
                    DS2PrimQtPrc.ExchIntCode.in_(target_exch_codes)
                )
            ).distinct().all()

            # DataFrame 변환
            df = pd.DataFrame([
                {
                    'InfoCode': s.InfoCode,
                    'DsCode': s.DsCode,
                    'DsQtName': s.DsQtName,
                    'Exchange': s.ExchName,
                    'ExchMnem': s.ExchMnem,
                    'ExchIntCode': s.ExchIntCode,
                    'Ticker': s.Ticker,
                    'StatusCode': s.StatusCode,
                    'DelistDate': s.DelistDate,
                    'Region': s.Region,
                }
                for s in stocks
            ])

            # 중복 제거 (같은 InfoCode + ExchIntCode 조합에서 마지막 것만 유지)
            df = df.drop_duplicates(subset=['InfoCode', 'ExchIntCode', 'DsQtName'], keep='last')

            # ETF/SPAC 필터링 (설정에서)
            exclude_keywords = self.data_config.get('filters', {}).get('exclude_keywords', [])
            if exclude_keywords:
                # 종목명에 키워드가 포함된 경우 제외
                mask = ~df['DsQtName'].str.contains('|'.join(exclude_keywords), case=False, na=False)
                initial_count = len(df)
                df = df[mask].copy()
                print(f"✓ ETF/SPAC/기타 제외: {initial_count:,} → {len(df):,} ({len(df)/initial_count*100:.1f}%)")

            return df

    def load_ohlcv_for_stock(
        self,
        info_code: int,
        start_date: datetime,
        adj_type: int = 2,
        exch_int_code: Optional[int] = None
    ) -> pd.DataFrame:
        """
        단일 종목의 OHLCV 데이터 로드

        Args:
            info_code: InfoCode
            start_date: 시작일
            adj_type: 조정 타입 (0=미수정, 1=분할만, 2=전체)
            exch_int_code: 거래소 코드 (지정 시 해당 거래소 데이터만 조회)

        Returns:
            OHLCV DataFrame
        """
        with SessionManager(self.db) as session:
            # VwDs2Pricing 뷰에서 조회 (조정 가격 포함)
            filters = [
                VwDs2Pricing.InfoCode == info_code,
                VwDs2Pricing.MarketDate >= start_date,
                VwDs2Pricing.AdjType == adj_type,
                VwDs2Pricing.Open_.isnot(None),
                VwDs2Pricing.Close_.isnot(None),
            ]

            # 거래소 필터 추가 (multi-listed 종목 중복 방지)
            if exch_int_code is not None:
                filters.append(VwDs2Pricing.ExchIntCode == exch_int_code)

            prices = session.query(
                VwDs2Pricing.InfoCode,
                VwDs2Pricing.MarketDate,
                VwDs2Pricing.Open_,
                VwDs2Pricing.High,
                VwDs2Pricing.Low,
                VwDs2Pricing.Close_,
                VwDs2Pricing.Volume,
                VwDs2Pricing.ExchIntCode,
                VwDs2Pricing.AdjType,
            ).filter(
                and_(*filters)
            ).order_by(
                VwDs2Pricing.MarketDate
            ).all()

            if not prices:
                return pd.DataFrame()

            # DataFrame 변환
            df = pd.DataFrame([
                {
                    'InfoCode': p.InfoCode,
                    'date': p.MarketDate,
                    'open': p.Open_,
                    'high': p.High,
                    'low': p.Low,
                    'close': p.Close_,
                    'volume': p.Volume,
                    'ExchIntCode': p.ExchIntCode,
                }
                for p in prices
            ])

            return df

    def load_ohlcv_batch(
        self,
        stock_list: pd.DataFrame,
        start_date: datetime,
        adj_type: int = 2,
        batch_size: int = 100
    ) -> pd.DataFrame:
        """
        여러 종목의 OHLCV 데이터를 배치로 로드 (개선된 버전)

        Args:
            stock_list: 종목 리스트 DataFrame (InfoCode, ExchIntCode 컬럼 필수)
            start_date: 시작일
            adj_type: 조정 타입
            batch_size: 한 번에 조회할 종목 수 (기본 100개)

        Returns:
            병합된 OHLCV DataFrame
        """
        total_stocks = len(stock_list)
        all_results = []

        # InfoCode 리스트 추출
        info_codes = stock_list['InfoCode'].tolist()

        # 배치 단위로 조회
        with SessionManager(self.db) as session:
            for i in tqdm(range(0, total_stocks, batch_size), desc="Loading OHLCV batches"):
                batch_codes = info_codes[i:i + batch_size]

                try:
                    # 한 쿼리로 여러 종목 조회 (IN 절 사용)
                    prices = session.query(
                        VwDs2Pricing.InfoCode,
                        VwDs2Pricing.MarketDate,
                        VwDs2Pricing.Open_,
                        VwDs2Pricing.High,
                        VwDs2Pricing.Low,
                        VwDs2Pricing.Close_,
                        VwDs2Pricing.Volume,
                        VwDs2Pricing.ExchIntCode,
                    ).filter(
                        and_(
                            VwDs2Pricing.InfoCode.in_(batch_codes),
                            VwDs2Pricing.MarketDate >= start_date,
                            VwDs2Pricing.AdjType == adj_type,
                            VwDs2Pricing.Open_.isnot(None),
                            VwDs2Pricing.Close_.isnot(None),
                        )
                    ).order_by(
                        VwDs2Pricing.InfoCode,
                        VwDs2Pricing.MarketDate
                    ).all()

                    if prices:
                        # DataFrame 변환
                        df_batch = pd.DataFrame([
                            {
                                'InfoCode': p.InfoCode,
                                'date': p.MarketDate,
                                'open': p.Open_,
                                'high': p.High,
                                'low': p.Low,
                                'close': p.Close_,
                                'volume': p.Volume,
                                'ExchIntCode': p.ExchIntCode,
                            }
                            for p in prices
                        ])
                        all_results.append(df_batch)

                except Exception as e:
                    print(f"Error loading batch {i//batch_size + 1}: {e}")

        # 모든 결과 병합
        if all_results:
            return pd.concat(all_results, ignore_index=True)
        else:
            return pd.DataFrame()

    def load_market_cap(self, info_codes: List[int], start_date: datetime) -> pd.DataFrame:
        """
        시가총액 데이터 로드

        Args:
            info_codes: InfoCode 리스트
            start_date: 시작일

        Returns:
            시가총액 DataFrame
        """
        with SessionManager(self.db) as session:
            # 배치로 나누어 조회
            batch_size = 1000
            results = []

            for i in range(0, len(info_codes), batch_size):
                batch = info_codes[i:i + batch_size]

                mktcap = session.query(
                    VwDs2MktCap.InfoCode,
                    VwDs2MktCap.MarketDate,
                    VwDs2MktCap.MktCap,
                ).filter(
                    and_(
                        VwDs2MktCap.InfoCode.in_(batch),
                        VwDs2MktCap.MarketDate >= start_date,
                        VwDs2MktCap.MktCap.isnot(None),
                    )
                ).all()

                if mktcap:
                    results.extend(mktcap)

            # DataFrame 변환
            if results:
                df = pd.DataFrame([
                    {
                        'InfoCode': m.InfoCode,
                        'date': m.MarketDate,
                        'market_cap': m.MktCap,
                    }
                    for m in results
                ])
                return df
            else:
                return pd.DataFrame()

    def get_index_code(self, index_mnem: str) -> Optional[int]:
        """
        인덱스 니모닉으로 DSIndexCode 조회

        Args:
            index_mnem: 인덱스 니모닉 (예: 'S&PCOMP' for SPY, 'NASA100' for QQQ)

        Returns:
            DSIndexCode 또는 None
        """
        with SessionManager(self.db) as session:
            # 정확한 매칭 사용 (LIKE 대신 ==로 인덱스 활용, 성능 향상)
            index = session.query(DS2EquityIndex.DSIndexCode).filter(
                DS2EquityIndex.DSIndexMnem == index_mnem
            ).first()

            return index.DSIndexCode if index else None

    def load_market_index(
        self,
        index_mnem: str,
        start_date: datetime,
        index_name: str = None
    ) -> pd.DataFrame:
        """
        시장 지수 데이터 로드

        Args:
            index_mnem: 인덱스 니모닉 (예: 'S&PCOMP', 'NASD100', 'CBOE')
            start_date: 시작일
            index_name: 결과 컬럼에 사용할 이름 (None이면 index_mnem 사용)

        Returns:
            지수 데이터 DataFrame (date, {index_name}_close, {index_name}_pi)
        """
        if index_name is None:
            index_name = index_mnem.lower()

        # 인덱스 코드 조회
        index_code = self.get_index_code(index_mnem)
        if index_code is None:
            print(f"   Warning: Index '{index_mnem}' not found in database")
            return pd.DataFrame()

        with SessionManager(self.db) as session:
            # 지수 데이터 조회
            index_data = session.query(
                DS2IndexData.ValueDate,
                DS2IndexData.PI_,
                DS2IndexData.RI,
            ).filter(
                and_(
                    DS2IndexData.DSIndexCode == index_code,
                    DS2IndexData.ValueDate >= start_date,
                )
            ).order_by(
                DS2IndexData.ValueDate
            ).all()

            if not index_data:
                return pd.DataFrame()

            # DataFrame 변환
            df = pd.DataFrame([
                {
                    'date': d.ValueDate,
                    f'{index_name}_pi': d.PI_,  # Price Index
                    f'{index_name}_ri': d.RI,   # Return Index
                }
                for d in index_data
            ])

            return df

    def load_market_indices(
        self,
        start_date: datetime,
        indices: Optional[Dict[str, str]] = None
    ) -> pd.DataFrame:
        """
        여러 시장 지수 데이터 로드 및 병합

        Args:
            start_date: 시작일
            indices: 인덱스 니모닉 딕셔너리 {name: mnemonic}
                     None이면 region에 따라 기본값 사용

        Returns:
            병합된 지수 데이터 DataFrame
        """
        if indices is None:
            # region에 따라 기본 지수 설정
            region = self.data_config.get('filters', {}).get('region', 'US')
            
            if region == 'US':
                indices = {
                    'spy': 'S&PCOMP',   # S&P 500
                    'qqq': 'NASA100',   # NASDAQ 100
                }
            elif region == 'KR':
                indices = {
                    'kospi200': 'KOR200I',   # KOSPI 200 (S&P 500 역할)
                    'kosdaq': 'KOSCOMP',     # KOSDAQ COMPOSITE (NASDAQ 역할)
                }
            elif region == 'JP':
                indices = {
                    'nikkei225': 'JAPDOWA',  # Nikkei 225
                }
            elif region == 'HK':
                indices = {
                    'hsi': 'HNGKNGI',        # Hang Seng Index
                }
            else:
                # 기타 region은 빈 딕셔너리 (지수 로드 안 함)
                indices = {}

        print(f"\nLoading market indices...")

        # 각 지수 로드
        index_dfs = []
        for name, mnem in indices.items():
            print(f"   Loading {name.upper()} ({mnem})...")
            df = self.load_market_index(mnem, start_date, name)
            if not df.empty:
                index_dfs.append(df)
                print(f"   ✓ Loaded {len(df)} records")
            else:
                print(f"   ✗ No data found")

        # 모든 지수를 날짜 기준으로 병합
        if index_dfs:
            result = index_dfs[0]
            for df in index_dfs[1:]:
                result = result.merge(df, on='date', how='outer')

            # 날짜 순 정렬
            result = result.sort_values('date').reset_index(drop=True)
            print(f"✓ Merged {len(result)} date records")

            return result
        else:
            return pd.DataFrame()

    def load_all_data(
        self,
        load_market_indices: bool = True,
        load_sector_info: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
        """
        모든 필요한 데이터 로드

        Args:
            load_market_indices: 시장 지수 로드 여부
            load_sector_info: 섹터/산업 정보 로드 여부

        Returns:
            (stock_list, ohlcv_data, market_cap_data, market_indices) 튜플
        """
        # 1. 종목 리스트 조회
        print("Loading stock list...")
        stock_list = self.get_stock_list()
        print(f"✓ Loaded {len(stock_list)} stocks")

        # 1-1. 섹터/산업 정보 로드 및 병합
        info_codes = stock_list['InfoCode'].tolist()
        if load_sector_info:
            print("\nLoading sector/industry information...")
            sector_info = self.load_sector_industry(info_codes)
            if not sector_info.empty:
                stock_list = stock_list.merge(sector_info, on='InfoCode', how='left')
                sector_coverage = stock_list['sector'].notna().sum() / len(stock_list) * 100
                industry_coverage = stock_list['industry'].notna().sum() / len(stock_list) * 100
                print(f"✓ Sector coverage: {sector_coverage:.1f}%")
                print(f"✓ Industry coverage: {industry_coverage:.1f}%")
            else:
                print("✗ No sector/industry data found")
                stock_list['sector'] = None
                stock_list['industry'] = None
        else:
            stock_list['sector'] = None
            stock_list['industry'] = None

        # 2. 시작일 계산
        min_years = self.data_config.get('min_years', 10)
        start_date = datetime.now() - timedelta(days=365 * min_years)
        print(f"✓ Start date: {start_date.date()}")

        # 3. OHLCV 데이터 로드 (배치 단위)
        adj_type = self.data_config.get('pricing', {}).get('adj_type', 2)
        batch_size = self.data_config.get('parallel', {}).get('batch_size', 100)
        all_ohlcv = []

        print(f"\nLoading OHLCV data in batches of {batch_size}...")
        for i in range(0, len(stock_list), batch_size):
            batch_stocks = stock_list.iloc[i:i + batch_size]
            print(f"Batch {i//batch_size + 1}/{(len(stock_list)-1)//batch_size + 1}")

            df_batch = self.load_ohlcv_batch(batch_stocks, start_date, adj_type)
            if not df_batch.empty:
                all_ohlcv.append(df_batch)

        ohlcv_data = pd.concat(all_ohlcv, ignore_index=True) if all_ohlcv else pd.DataFrame()
        print(f"✓ Loaded {len(ohlcv_data)} OHLCV records")

        # 4. 시가총액 데이터 로드
        print("\nLoading market cap data...")
        market_cap_data = self.load_market_cap(info_codes, start_date)
        print(f"✓ Loaded {len(market_cap_data)} market cap records")

        # 5. 시장 지수 데이터 로드 (선택적)
        market_indices = None
        if load_market_indices:
            market_indices = self.load_market_indices(start_date)

        return stock_list, ohlcv_data, market_cap_data, market_indices
