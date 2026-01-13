"""
갭 상승 후보 메시지 처리 핸들러

Kafka에서 받은 GapCandidateMessage를 처리하여 예측 수행
Todo.md의 실전 예측 데이터 생성 흐름도 참고
"""

import logging
from datetime import datetime, timedelta, date
from typing import Optional
import pandas as pd
import numpy as np
from sqlalchemy import and_, select
from sqlalchemy.orm import Session

from app.config.db_connections import get_sync_session_factory
from app.config.settings import settings
from app.database.database.stocks import StockPrices, MarketIndices
from app.kafka.schemas import GapCandidateMessage, PredictionResultMessage
from app.kafka.producer import PredictionProducer
from app.prediction.predictor import load_predictor

logger = logging.getLogger(__name__)

# 전역 Predictor 인스턴스 (캐싱)
_predictor = None


def get_predictor():
    """Predictor 싱글톤 인스턴스 반환 (캐싱)"""
    global _predictor
    
    if _predictor is None:
        logger.info(f"모델 로드 중: {settings.model_path}")
        _predictor = load_predictor(settings.model_path, threshold=settings.prediction_threshold)
        logger.info("✓ Predictor 로드 완료")
    
    return _predictor


def handle_gap_candidate_message(message: GapCandidateMessage) -> Optional[PredictionResultMessage]:
    """
    갭 상승 후보 메시지 처리
    
    흐름:
    1. DB에서 과거 60일 StockPrices 데이터 조회 (기술지표 이미 계산됨)
    2. 오늘 날짜의 MarketIndices 조회
    3. 오늘 데이터 추가 및 gap_pct 계산
    4. Feature DataFrame 생성
    5. Predictor로 예측 수행
    6. PredictionResultMessage 생성 및 Kafka 발행
    
    Args:
        message: GapCandidateMessage 인스턴스
        
    Returns:
        PredictionResultMessage (예측 성공 시) 또는 None (실패 시)
    """
    logger.info(
        f"메시지 처리 시작: {message.stock_code} ({message.stock_name}) "
        f"gap={message.gap_rate:.2f}%"
    )
    
    try:
        # ========================================
        # 1. DB에서 과거 데이터 조회
        # ========================================
        session_factory = get_sync_session_factory()
        
        with session_factory() as session:
            # 오늘 날짜
            today = message.timestamp.date()
            start_date = today - timedelta(days=settings.feature_lookback_days)
            
            # 1-1. 종목별 과거 60일 StockPrices 조회 (기술지표 포함)
            stock_prices = session.query(StockPrices).filter(
                and_(
                    StockPrices.symbol == message.stock_code,
                    StockPrices.date >= start_date,
                    StockPrices.date < today  # 오늘 제외 (어제까지)
                )
            ).order_by(StockPrices.date.asc()).all()
            
            if len(stock_prices) < 30:  # 최소 30일 데이터 필요
                logger.warning(
                    f"과거 데이터 부족: {message.stock_code} "
                    f"(필요: 30일 이상, 실제: {len(stock_prices)}일)"
                )
                return None
            
            # 1-2. 오늘 날짜의 MarketIndices 조회
            market_indices = session.query(MarketIndices).filter(
                MarketIndices.date == today
            ).first()
            
            if not market_indices:
                logger.warning(f"시장 지수 데이터 없음: {today}")
                # market_gap_diff는 NaN으로 처리
            
            # ========================================
            # 2. DataFrame 생성 및 오늘 데이터 추가
            # ========================================
            # 과거 데이터를 DataFrame으로 변환
            historical_data = []
            for sp in stock_prices:
                historical_data.append({
                    'date': sp.date,
                    'open': float(sp.open),
                    'high': float(sp.high),
                    'low': float(sp.low),
                    'close': float(sp.close),
                    'volume': int(sp.volume),
                    'gap_pct': float(sp.gap_pct) if sp.gap_pct else None,
                    'prev_return': float(sp.prev_return) if sp.prev_return else None,
                    'prev_range_pct': float(sp.prev_range_pct) if sp.prev_range_pct else None,
                    'volume_ratio': float(sp.volume_ratio) if sp.volume_ratio else None,
                    'rsi_14': float(sp.rsi_14) if sp.rsi_14 else None,
                    'atr_14': float(sp.atr_14) if sp.atr_14 else None,
                    'atr_ratio': float(sp.atr_ratio) if sp.atr_ratio else None,
                    'bollinger_position': float(sp.bollinger_position) if sp.bollinger_position else None,
                    'ma_5': float(sp.ma_5) if sp.ma_5 else None,
                    'ma_20': float(sp.ma_20) if sp.ma_20 else None,
                    'ma_50': float(sp.ma_50) if sp.ma_50 else None,
                    'above_ma5': int(sp.above_ma5) if sp.above_ma5 is not None else None,
                    'above_ma20': int(sp.above_ma20) if sp.above_ma20 is not None else None,
                    'above_ma50': int(sp.above_ma50) if sp.above_ma50 is not None else None,
                    'return_5d': float(sp.return_5d) if sp.return_5d else None,
                    'return_20d': float(sp.return_20d) if sp.return_20d else None,
                    'consecutive_up_days': int(sp.consecutive_up_days) if sp.consecutive_up_days else None,
                    'market_gap_diff': float(sp.market_gap_diff) if sp.market_gap_diff else None,
                })
            
            df = pd.DataFrame(historical_data)
            
            # 오늘 데이터 추가 (전일 종가 필요)
            if df.empty or df['close'].iloc[-1] is None:
                logger.warning(f"전일 종가 데이터 없음: {message.stock_code}")
                return None
            
            prev_close = float(df['close'].iloc[-1])
            
            # 전일 데이터에서 윗꼬리/아랫꼬리 계산
            prev_data = df.iloc[-1]
            prev_high = float(prev_data['high']) if pd.notna(prev_data['high']) else None
            prev_low = float(prev_data['low']) if pd.notna(prev_data['low']) else None
            prev_open = float(prev_data['open']) if pd.notna(prev_data['open']) else None
            prev_close_val = float(prev_data['close']) if pd.notna(prev_data['close']) else None
            
            # 전일 윗꼬리/아랫꼬리 비율 계산
            prev_upper_shadow = None
            prev_lower_shadow = None
            if prev_high and prev_low and prev_open and prev_close_val:
                body_high = max(prev_open, prev_close_val)
                body_low = min(prev_open, prev_close_val)
                total_range = prev_high - prev_low
                if total_range > 0:
                    prev_upper_shadow = (prev_high - body_high) / total_range
                    prev_lower_shadow = (body_low - prev_low) / total_range
            
            # MA5-MA20 크로스 (MA5가 MA20 위에 있는지)
            ma5_val = float(prev_data['ma_5']) if pd.notna(prev_data['ma_5']) else None
            ma20_val = float(prev_data['ma_20']) if pd.notna(prev_data['ma_20']) else None
            ma5_ma20_cross = 1 if (ma5_val and ma20_val and ma5_val > ma20_val) else 0 if (ma5_val and ma20_val) else 0
            
            # 날짜 관련 Feature
            day_of_week = today.weekday()  # 0=월요일, 4=금요일
            month = today.month
            is_month_start = 1 if today.day <= 5 else 0
            is_month_end = 1 if today.day >= 25 else 0
            is_quarter_end = 1 if today.month in [3, 6, 9, 12] and today.day >= 25 else 0
            
            # 오늘 row 추가
            today_row = {
                'date': today,
                'open': message.stock_open,
                'high': None,  # 아직 모름
                'low': None,
                'close': None,
                'volume': None,
                'gap_pct': message.gap_rate,  # 메시지에서 받은 갭률
                'prev_return': None,  # 계산 불가 (오늘 데이터)
                'prev_range_pct': None,
                'volume_ratio': None,
                'rsi_14': float(df['rsi_14'].iloc[-1]) if not df['rsi_14'].isna().iloc[-1] else None,
                'atr_14': float(df['atr_14'].iloc[-1]) if not df['atr_14'].isna().iloc[-1] else None,
                'atr_ratio': message.gap_rate / float(df['atr_14'].iloc[-1]) if not df['atr_14'].isna().iloc[-1] else None,
                'bollinger_position': None,  # 오늘은 계산 불가
                'ma_5': float(df['ma_5'].iloc[-1]) if not df['ma_5'].isna().iloc[-1] else None,
                'ma_20': float(df['ma_20'].iloc[-1]) if not df['ma_20'].isna().iloc[-1] else None,
                'ma_50': float(df['ma_50'].iloc[-1]) if not df['ma_50'].isna().iloc[-1] else None,
                'above_ma5': 1 if message.stock_open > df['ma_5'].iloc[-1] else 0 if not df['ma_5'].isna().iloc[-1] else None,
                'above_ma20': 1 if message.stock_open > df['ma_20'].iloc[-1] else 0 if not df['ma_20'].isna().iloc[-1] else None,
                'above_ma50': 1 if message.stock_open > df['ma_50'].iloc[-1] else 0 if not df['ma_50'].isna().iloc[-1] else None,
                'return_5d': None,
                'return_20d': None,
                'consecutive_up_days': None,
                # 추가된 Feature들
                'prev_upper_shadow': prev_upper_shadow,
                'prev_lower_shadow': prev_lower_shadow,
                'ma5_ma20_cross': ma5_ma20_cross,
                'day_of_week': day_of_week,
                'month': month,
                'is_month_start': is_month_start,
                'is_month_end': is_month_end,
                'is_quarter_end': is_quarter_end,
            }
            
            # 시장 갭 차이 계산
            if market_indices:
                # ✅ 학습 시와 동일하게: KOSPI200 사용 (KOSPI 종목의 경우)
                # market_features.py의 우선순위: spy > kospi200 > 첫 번째
                if message.exchange == 'KOSPI':
                    # KOSPI 종목 → KOSPI200 지수 사용
                    market_gap = float(market_indices.kospi200_gap_pct) if market_indices.kospi200_gap_pct else None
                elif message.exchange == 'KOSDAQ':
                    market_gap = float(market_indices.kosdaq_gap_pct) if market_indices.kosdaq_gap_pct else None
                else:
                    market_gap = None

                if market_gap is not None:
                    today_row['market_gap_diff'] = message.gap_rate - market_gap
                else:
                    today_row['market_gap_diff'] = None
            else:
                today_row['market_gap_diff'] = None
            
            df = pd.concat([df, pd.DataFrame([today_row])], ignore_index=True)
            
            # ========================================
            # 3. Feature DataFrame 준비 (Predictor 입력 형식)
            # ========================================
            # 오늘 row만 추출
            today_features = df.iloc[[-1]].copy()
            
            # Predictor가 요구하는 Feature 컬럼 확인
            predictor = get_predictor()
            required_features = predictor.features if predictor.features else []
            
            # 필수 Feature가 없는 경우 NaN으로 채움
            for feat in required_features:
                if feat not in today_features.columns:
                    today_features[feat] = None
                    logger.warning(f"Feature 누락: {feat}")
            
            # Feature 선택 (Predictor가 요구하는 것만)
            if required_features:
                available_features = [f for f in required_features if f in today_features.columns]
                X = today_features[available_features]
            else:
                X = today_features

            # ✅ dtype 변환: 모든 컬럼을 float로 변환 (XGBoost 요구사항)
            for col in X.columns:
                try:
                    X[col] = pd.to_numeric(X[col], errors='coerce')
                except Exception as e:
                    logger.warning(f"컬럼 {col} 변환 실패: {e}")
            
            # NaN 체크
            nan_ratio = X.isna().sum().sum() / (len(X.columns) * len(X))
            if nan_ratio > 0.5:
                logger.warning(
                    f"Feature NaN 비율 높음: {nan_ratio:.1%} "
                    f"(종목: {message.stock_code})"
                )

            # ========================================
            # 3.5. Feature CSV 저장 (디버깅용)
            # ========================================
            try:
                from pathlib import Path
                feature_dir = Path("/app/data/features")
                feature_dir.mkdir(parents=True, exist_ok=True)

                csv_path = feature_dir / "predictions.csv"

                # 메타 정보 추가
                X_with_meta = X.copy()
                X_with_meta.insert(0, 'timestamp', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
                X_with_meta.insert(1, 'stock_code', message.stock_code)
                X_with_meta.insert(2, 'stock_name', message.stock_name)
                X_with_meta.insert(3, 'date', today.strftime('%Y-%m-%d'))
                X_with_meta.insert(4, 'stock_open', message.stock_open)
                X_with_meta.insert(5, 'gap_rate', message.gap_rate)

                # 파일이 없으면 헤더와 함께 생성, 있으면 append
                if not csv_path.exists():
                    X_with_meta.to_csv(csv_path, index=False, encoding='utf-8-sig', mode='w')
                    logger.info(f"Feature CSV 생성: {csv_path}")
                else:
                    X_with_meta.to_csv(csv_path, index=False, encoding='utf-8-sig', mode='a', header=False)
                    logger.debug(f"Feature 추가: {message.stock_code}")
            except Exception as e:
                logger.warning(f"Feature CSV 저장 실패: {e}")

            # ========================================
            # 4. 예측 수행
            # ========================================
            predictions = predictor.predict(X)
            
            if predictions.empty:
                logger.error(f"예측 실패: {message.stock_code}")
                return None
            
            pred = predictions.iloc[0]
            
            # ========================================
            # 5. PredictionResultMessage 생성
            # ========================================
            # 매매 신호 결정
            signal = 'HOLD'
            if pred['prob_up'] >= settings.min_prob_up and pred['expected_return'] >= settings.min_expected_return:
                signal = 'BUY'
            
            # 신뢰도 결정
            confidence = None
            if pred['prob_up'] >= 0.7:
                confidence = 'HIGH'
            elif pred['prob_up'] >= 0.5:
                confidence = 'MEDIUM'
            else:
                confidence = 'LOW'
            
            result_message = PredictionResultMessage(
                timestamp=datetime.now(),
                stock_code=message.stock_code,
                stock_name=message.stock_name,
                exchange=message.exchange,
                date=today.strftime('%Y-%m-%d'),
                gap_rate=message.gap_rate,
                stock_open=message.stock_open,
                prob_up=float(pred['prob_up']),
                prob_down=float(pred['prob_down']),
                predicted_direction=int(pred['predicted_direction']),
                expected_return=float(pred['expected_return']),
                return_if_up=float(pred['return_if_up']),
                return_if_down=float(pred['return_if_down']),
                max_return_if_up=float(pred.get('max_return_if_up', None)) if 'max_return_if_up' in pred and pd.notna(pred['max_return_if_up']) else None,
                take_profit_target=float(pred.get('take_profit_target', None)) if 'take_profit_target' in pred and pd.notna(pred['take_profit_target']) else None,
                signal=signal,
                model_version='v1.0',
                confidence=confidence,
            )
            
            logger.info(
                f"✓ 예측 완료: {message.stock_code} "
                f"prob_up={pred['prob_up']:.2f} "
                f"expected_return={pred['expected_return']:.2f}% "
                f"signal={signal}"
            )
            
            # ========================================
            # 6. Kafka 발행
            # ========================================
            producer = PredictionProducer()
            success = producer.send_prediction(result_message)
            
            if success:
                logger.info(f"✓ Kafka 발행 완료: {message.stock_code}")
            else:
                logger.error(f"✗ Kafka 발행 실패: {message.stock_code}")
            
            # ========================================
            # 7. (선택) DB 저장 - TODO: gap_predictions 테이블 생성 후 구현
            # ========================================
            # TODO: 예측 결과를 gap_predictions 테이블에 저장
            # - id, stock_code, date, gap_pct, prob_up, expected_return 등
            
            return result_message
            
    except Exception as e:
        logger.error(f"메시지 처리 실패: {message.stock_code} - {e}", exc_info=True)
        return None
