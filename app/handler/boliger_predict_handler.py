"""
Boliger 앙상블 모델 예측 핸들러

Kafka에서 BoligerTriggerMessage를 수신하면:
1. pickle 파일 로드 (Input1, Input2 완성됨)
2. SeparatedPredictor로 배치 추론
3. PredictionResultBatchMessage 생성 + Kafka 발행 + DB 저장
"""

import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from sqlalchemy import and_

from app.config.db_connections import get_sync_session_factory
from app.config.settings import settings
from app.database.database.strategy import GapPredictions, SignalType, ConfidenceLevel
from app.kafka.schemas import (
    BoligerTriggerMessage,
    PredictionResultMessage,
    PredictionResultBatchMessage,
)
from app.kafka.producer import PredictionProducer

logger = logging.getLogger(__name__)

# SeparatedPredictor 싱글톤
_predictor = None


def get_boliger_predictor():
    """Boliger SeparatedPredictor 싱글톤 반환."""
    global _predictor

    if _predictor is None:
        try:
            from boliger.ensemble_sep import SeparatedPredictor

            model_dir = settings.boliger_model_dir
            config_dir = settings.boliger_config_dir
            ensemble_mode = settings.boliger_ensemble_mode

            logger.info(
                f"Loading Boliger predictor: model_dir={model_dir}, "
                f"config_dir={config_dir}, mode={ensemble_mode}"
            )

            predictor = SeparatedPredictor(
                ensemble_mode=ensemble_mode,
                model_dir=model_dir,
                config_dir=config_dir,
                device="cpu",
            )
            predictor.load_models()

            # Meta learner 로드 (stacking 모드인 경우)
            meta_dir = Path(model_dir) / "meta"
            if meta_dir.exists() and ensemble_mode.startswith("stacking"):
                predictor.load_meta_learners(str(meta_dir))

            _predictor = predictor
            logger.info("Boliger predictor loaded successfully")

        except ImportError as e:
            logger.error(f"Failed to import boliger package: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to load Boliger predictor: {e}", exc_info=True)
            raise

    return _predictor


def handle_boliger_trigger(message: BoligerTriggerMessage) -> Optional[PredictionResultBatchMessage]:
    """Boliger 트리거 메시지 처리.

    Args:
        message: BoligerTriggerMessage (pickle 경로 포함).

    Returns:
        PredictionResultBatchMessage (성공 시) or None.
    """
    logger.info(
        f"Boliger trigger received: {message.exchange_type}, "
        f"{message.stock_count} stocks, date={message.date}"
    )

    try:
        # 1. Pickle 로드
        pickle_path = Path(message.pickle_path)
        if not pickle_path.exists():
            logger.error(f"Pickle file not found: {pickle_path}")
            return None

        with open(pickle_path, "rb") as f:
            data = pickle.load(f)

        stocks = data.get("stocks", [])
        if not stocks:
            logger.warning("No stocks in pickle")
            return None

        logger.info(f"Loaded {len(stocks)} stocks from {pickle_path}")

        # 2. 배치 텐서 구성
        predictor = get_boliger_predictor()

        max_seq_len = max(s["length"] for s in stocks)
        n_stocks = len(stocks)
        n_features = stocks[0]["input1"].shape[1]  # 33
        n_input2 = stocks[0]["input2"].shape[0]     # 12

        # 패딩된 텐서 생성
        input1_batch = torch.zeros(n_stocks, max_seq_len, n_features, dtype=torch.float32)
        lengths_batch = torch.zeros(n_stocks, dtype=torch.long)
        input2_batch = torch.zeros(n_stocks, n_input2, dtype=torch.float32)

        # Flat features (Tree 모델용): input1의 마지막 행 + input2
        flat_features = np.zeros((n_stocks, n_features), dtype=np.float32)

        for i, stock in enumerate(stocks):
            seq_len = stock["length"]
            input1 = stock["input1"]  # (seq_len, 33)
            input2 = stock["input2"]  # (12,)

            # 패딩
            actual_len = min(seq_len, max_seq_len)
            input1_batch[i, :actual_len, :] = torch.from_numpy(input1[:actual_len])
            lengths_batch[i] = actual_len
            input2_batch[i] = torch.from_numpy(input2)

            # Flat features: 시퀀스 마지막 행 (변곡점 직전)
            flat_features[i] = input1[actual_len - 1]

        # 3. 배치 추론
        logger.info(f"Running inference on {n_stocks} stocks...")
        results = predictor.predict_tensors(
            input1_batch, lengths_batch, input2_batch, flat_features
        )

        probs = results["prob"].numpy().squeeze(-1)       # (n_stocks,)
        high_returns = results["high_return"].numpy().squeeze(-1)  # (n_stocks,)
        low_returns = results["low_return"].numpy().squeeze(-1)    # (n_stocks,)

        expected_returns = probs * high_returns * 100 + (1 - probs) * low_returns * 100
        logger.info(f"Inference complete ({n_stocks} stocks):")
        logger.info(f"  상승수익: max={high_returns.max()*100:.2f}% med={np.median(high_returns)*100:.2f}% avg={high_returns.mean()*100:.2f}% min={high_returns.min()*100:.2f}%")
        logger.info(f"  하락손실: max={low_returns.max()*100:.2f}% med={np.median(low_returns)*100:.2f}% avg={low_returns.mean()*100:.2f}% min={low_returns.min()*100:.2f}%")
        logger.info(f"  기대수익: max={expected_returns.max():.2f}% med={np.median(expected_returns):.2f}% avg={expected_returns.mean():.2f}% min={expected_returns.min():.2f}%")
        logger.info(f"  상승확률: max={probs.max():.3f} med={np.median(probs):.3f} avg={probs.mean():.3f} min={probs.min():.3f}")

        # 4. PredictionResultMessage 생성
        prediction_results = []
        session_factory = get_sync_session_factory()

        with session_factory() as session:
            for i, stock in enumerate(stocks):
                prob_up = float(np.clip(probs[i], 0, 1))
                prob_down = 1.0 - prob_up
                high_ret = float(high_returns[i])
                low_ret = float(low_returns[i])
                gap_ratio = float(stock.get("gap_ratio", 0))

                # 기대 수익률 계산
                expected_return = prob_up * high_ret * 100 + prob_down * low_ret * 100

                # 매매 신호 결정
                signal = "HOLD"
                if prob_up >= settings.min_prob_up:
                    signal = "BUY"

                # 신뢰도 결정
                if prob_up >= 0.4:
                    confidence = "HIGH"
                elif prob_up >= 0.3:
                    confidence = "MEDIUM"
                else:
                    confidence = "LOW"

                # 목표 수익률 (고점 예측 기반)
                take_profit_target = high_ret * 100 * settings.take_profit_ratio if high_ret > 0 else None

                result_msg = PredictionResultMessage(
                    timestamp=datetime.now(),
                    stock_code=stock["stock_code"],
                    stock_name=stock.get("stock_name", ""),
                    market_cap=stock.get("market_cap", 0),
                    exchange=stock.get("exchange", ""),
                    strategy_id=1,
                    date=message.date,
                    gap_rate=gap_ratio * 100,
                    stock_open=float(stock.get("stock_open", 0)),
                    avg_volume_20d=int(stock["avg_volume_20d"]) if stock.get("avg_volume_20d") else None,
                    prob_up=prob_up,
                    prob_down=prob_down,
                    predicted_direction=1 if prob_up >= 0.5 else 0,
                    expected_return=expected_return,
                    return_if_up=high_ret * 100,
                    return_if_down=low_ret * 100,
                    max_return_if_up=high_ret * 100 if high_ret > 0 else None,
                    take_profit_target=take_profit_target,
                    signal=signal,
                    model_version="boliger-v2.0",
                    confidence=confidence,
                )
                # 상승 시 예상 수익률 3% 이상만 저장 및 발행
                if result_msg.return_if_up >= 2:
                    prediction_results.append(result_msg)
                    _save_prediction_to_db(session, result_msg, message)

            session.commit()

        logger.info(
            f"Predictions filtered (return_if_up>=3%): {len(prediction_results)} stocks"
        )

        # 5. 배치 결과 메시지 생성
        batch_result = PredictionResultBatchMessage(
            timestamp=datetime.now(),
            total_count=len(prediction_results),
            exchange_type=message.exchange_type,
            predictions=prediction_results,
        )

        return batch_result

    except Exception as e:
        logger.error(f"Boliger prediction failed: {e}", exc_info=True)
        return None


def _save_prediction_to_db(session, result: PredictionResultMessage, trigger: BoligerTriggerMessage):
    """예측 결과를 DB에 저장."""
    try:
        from datetime import date as date_type

        prediction_date = datetime.strptime(trigger.date, "%Y-%m-%d").date()

        existing = session.query(GapPredictions).filter(
            and_(
                GapPredictions.stock_code == result.stock_code,
                GapPredictions.prediction_date == prediction_date,
            )
        ).first()

        signal_enum = SignalType[result.signal]
        confidence_enum = ConfidenceLevel[result.confidence] if result.confidence else None

        if existing:
            existing.timestamp = result.timestamp.date()
            existing.stock_name = result.stock_name
            existing.gap_rate = result.gap_rate
            existing.stock_open = result.stock_open
            existing.prob_up = result.prob_up
            existing.prob_down = result.prob_down
            existing.predicted_direction = result.predicted_direction
            existing.expected_return = result.expected_return
            existing.return_if_up = result.return_if_up
            existing.return_if_down = result.return_if_down
            existing.max_return_if_up = result.max_return_if_up
            existing.take_profit_target = result.take_profit_target
            existing.signal = signal_enum
            existing.model_version = result.model_version
            existing.confidence = confidence_enum
            existing.is_nxt = trigger.exchange_type == "NXT"
        else:
            new_prediction = GapPredictions(
                timestamp=result.timestamp.date(),
                stock_code=result.stock_code,
                stock_name=result.stock_name,
                exchange=result.exchange,
                strategy_id=result.strategy_id,
                prediction_date=prediction_date,
                gap_rate=result.gap_rate,
                stock_open=result.stock_open,
                prob_up=result.prob_up,
                prob_down=result.prob_down,
                predicted_direction=result.predicted_direction,
                expected_return=result.expected_return,
                return_if_up=result.return_if_up,
                return_if_down=result.return_if_down,
                max_return_if_up=result.max_return_if_up,
                take_profit_target=result.take_profit_target,
                signal=signal_enum,
                model_version=result.model_version,
                confidence=confidence_enum,
                is_nxt=trigger.exchange_type == "NXT",
            )
            session.add(new_prediction)

    except Exception as e:
        logger.error(f"DB save failed for {result.stock_code}: {e}", exc_info=True)
