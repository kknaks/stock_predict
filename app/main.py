"""
AI 서버 메인 엔트리 포인트

Kafka Consumer를 실행하여 갭 상승 후보 메시지를 수신하고 예측 수행
"""

import json
import logging
import signal
import sys
import threading
from typing import Optional

from kafka import KafkaConsumer
from kafka.errors import KafkaError

from app.kafka.consumer import GapCandidateConsumer
from app.kafka.producer import PredictionProducer
from app.handler.gap_predict_handler import handle_gap_candidate_message, reload_predictor
from app.handler.boliger_predict_handler import handle_boliger_trigger
from app.config.settings import settings
from app.config.kafka_connections import get_kafka_config
from app.prediction.predictor import HybridPredictor
from app.kafka.schemas import (
    GapCandidateBatchMessage,
    PredictionResultBatchMessage,
    BoligerTriggerMessage,
)
from datetime import datetime

# 모델 파일에서 참조하는 클래스 (노트북에서 저장 시 사용된 이름)
# joblib unpickle을 위해 이 모듈에 정의되어야 함
StackingHybridPredictor = HybridPredictor

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger(__name__)

# 전역 변수 (시그널 핸들러용)
consumer: Optional[GapCandidateConsumer] = None
producer: Optional[PredictionProducer] = None


def signal_handler(sig, frame):
    """시그널 핸들러 (Ctrl+C 등)"""
    logger.info("종료 신호 수신...")
    if consumer:
        consumer.close()
    sys.exit(0)


def start_retrain_result_consumer():
    """model_retrain_result 토픽 consume (별도 스레드)"""
    kafka_config = get_kafka_config()
    consumer_config = kafka_config.get_consumer_config()
    consumer_config['group_id'] = f"{settings.kafka_group_id}-retrain"

    try:
        retrain_consumer = KafkaConsumer(
            settings.topic_retrain_result,
            **consumer_config
        )
        logger.info(f"✓ Retrain result consumer 시작: {settings.topic_retrain_result}")

        for message in retrain_consumer:
            try:
                data = json.loads(message.value)
                status = data.get("status")
                is_deployed = data.get("is_deployed", False)
                version = data.get("new_version", "unknown")

                logger.info(
                    f"Retrain 결과 수신: version={version}, "
                    f"status={status}, is_deployed={is_deployed}"
                )

                if status == "success" and is_deployed:
                    logger.info(f"새 모델 배포 감지 (version={version}), 모델 리로드...")
                    reload_predictor()
                else:
                    logger.info(f"모델 리로드 건너뜀 (status={status}, is_deployed={is_deployed})")

            except Exception as e:
                logger.error(f"Retrain 결과 메시지 처리 오류: {e}", exc_info=True)

    except KafkaError as e:
        logger.error(f"Retrain result consumer 초기화 실패: {e}")
    except Exception as e:
        logger.error(f"Retrain result consumer 오류: {e}", exc_info=True)


def start_boliger_consumer():
    """boliger_prediction_trigger 토픽 consume (별도 스레드)"""
    kafka_config = get_kafka_config()
    consumer_config = kafka_config.get_consumer_config()
    consumer_config['group_id'] = f"{settings.kafka_group_id}-boliger"

    boliger_producer = None

    try:
        boliger_consumer = KafkaConsumer(
            settings.topic_boliger_trigger,
            **consumer_config
        )
        logger.info(f"Boliger consumer started: {settings.topic_boliger_trigger}")

        boliger_producer = PredictionProducer()

        for message in boliger_consumer:
            try:
                trigger = BoligerTriggerMessage.from_json(message.value)
                logger.info(
                    f"Boliger trigger: {trigger.exchange_type}, "
                    f"{trigger.stock_count} stocks"
                )

                batch_result = handle_boliger_trigger(trigger)

                if batch_result and batch_result.predictions:
                    success = boliger_producer.send_batch_message(batch_result)
                    if success:
                        logger.info(
                            f"Boliger predictions published: "
                            f"{batch_result.total_count} stocks "
                            f"({trigger.exchange_type})"
                        )
                    else:
                        logger.error("Boliger prediction publish failed")
                else:
                    logger.warning("Boliger prediction returned no results")

            except Exception as e:
                logger.error(f"Boliger message processing error: {e}", exc_info=True)

    except KafkaError as e:
        logger.error(f"Boliger consumer init failed: {e}")
    except Exception as e:
        logger.error(f"Boliger consumer error: {e}", exc_info=True)
    finally:
        if boliger_producer:
            boliger_producer.close()


def main():
    """메인 함수"""
    global consumer, producer

    logger.info("=" * 80)
    logger.info("Stock Predict AI Server 시작")
    logger.info("=" * 80)
    logger.info(f"  - Kafka Topic: {settings.topic_gap_candidate}")
    logger.info(f"  - Boliger Topic: {settings.topic_boliger_trigger}")
    logger.info(f"  - Group ID: {settings.kafka_group_id}")
    logger.info(f"  - Model Path: {settings.model_path}")
    logger.info(f"  - Boliger Model: {settings.boliger_model_dir}")
    logger.info(f"  - DB Host: {settings.db_host}")
    logger.info("=" * 80)
    
    # 시그널 핸들러 등록
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Retrain result consumer 스레드 시작
    retrain_thread = threading.Thread(
        target=start_retrain_result_consumer,
        daemon=True,
        name="retrain-result-consumer"
    )
    retrain_thread.start()

    # Boliger prediction trigger consumer 스레드 시작
    boliger_thread = threading.Thread(
        target=start_boliger_consumer,
        daemon=True,
        name="boliger-consumer"
    )
    boliger_thread.start()

    try:
        # Consumer 생성
        consumer = GapCandidateConsumer()
        
        logger.info("메시지 수신 대기 중...")
        logger.info("(종료하려면 Ctrl+C)")
        
        # Producer 생성
        producer = PredictionProducer()
        
        # 스트리밍 모드로 메시지 수신 및 처리
        def batch_message_handler(batch_message: GapCandidateBatchMessage):
            """배치 메시지 처리 핸들러"""
            try:
                logger.info(
                    f"✓ 배치 메시지 수신: {batch_message.total_count}개 종목 "
                    f"(timestamp: {batch_message.timestamp})"
                )
                
                # 개별 메시지로 변환
                individual_messages = batch_message.to_individual_messages()
                
                # 모든 종목 예측 수행
                prediction_results = []
                success_count = 0
                failed_count = 0
                
                for message in individual_messages:
                    try:
                        result = handle_gap_candidate_message(message)
                        if result:
                            prediction_results.append(result)
                            success_count += 1
                            logger.debug(
                                f"✓ 처리 완료: {result.stock_code} "
                                f"signal={result.signal} "
                                f"expected_return={result.expected_return:.2f}%"
                            )
                        else:
                            failed_count += 1
                            logger.warning(f"✗ 처리 실패: {message.stock_code}")
                    except Exception as e:
                        failed_count += 1
                        logger.error(f"종목 처리 중 오류 ({message.stock_code}): {e}", exc_info=True)
                
                # 배치 예측 결과 메시지 생성 및 발행
                batch_result = PredictionResultBatchMessage(
                    timestamp=datetime.now(),
                    total_count=len(prediction_results),
                    exchange_type=batch_message.exchange_type,
                    predictions=prediction_results
                )

                success = producer.send_batch_message(batch_result)
                if success:
                    logger.info(
                        f"✓ 배치 예측 결과 발행 완료: {len(prediction_results)}개 종목 "
                        f"(성공: {success_count}, 실패: {failed_count})"
                    )
                else:
                    logger.error("✗ 배치 예측 결과 발행 실패")
                    
            except Exception as e:
                logger.error(f"배치 메시지 처리 중 오류: {e}", exc_info=True)
        
        # 스트리밍 모드로 실행 (blocking)
        # 배치 메시지를 직접 처리하도록 수정
        if not consumer.consumer:
            raise RuntimeError("Consumer not initialized")

        logger.info(f"Starting streaming consumer for {consumer.topic}...")

        try:
            for message in consumer.consumer:
                try:
                    # 배치 메시지 파싱
                    batch_message = GapCandidateBatchMessage.from_json(message.value)
                    
                    # 배치 메시지 처리
                    batch_message_handler(batch_message)

                except Exception as e:
                    logger.error(f"Error processing batch message: {e}")
                    logger.error(f"Raw message: {message.value}")

        except KeyboardInterrupt:
            logger.info("Consumer interrupted by user")
        
    except KeyboardInterrupt:
        logger.info("사용자에 의해 종료됨")
    except Exception as e:
        logger.error(f"서버 실행 중 오류: {e}", exc_info=True)
        sys.exit(1)
    finally:
        if consumer:
            consumer.close()
        if producer:
            producer.close()
        logger.info("서버 종료")


if __name__ == "__main__":
    main()
