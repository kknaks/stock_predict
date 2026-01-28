"""
Kafka Producer - AI 예측 결과 발행
"""

import logging
from typing import Optional, List
from kafka import KafkaProducer
from kafka.errors import KafkaError

from app.config.kafka_connections import get_kafka_config
from .schemas import PredictionResultMessage, PredictionResultBatchMessage

logger = logging.getLogger(__name__)


class PredictionProducer:
    """
    AI 예측 결과 발행 Producer

    토픽: ai_prediction_result
    """

    def __init__(self, custom_config: Optional[dict] = None):
        """
        Args:
            custom_config: 커스텀 Producer 설정 (선택)
        """
        self.config = get_kafka_config()
        self.topic = self.config.topic_prediction_result

        # Producer 설정
        producer_config = self.config.get_producer_config()
        if custom_config:
            producer_config.update(custom_config)

        # Producer 생성
        self.producer: Optional[KafkaProducer] = None
        self._init_producer(producer_config)

    def _init_producer(self, config: dict):
        """Producer 초기화"""
        try:
            self.producer = KafkaProducer(**config)
            logger.info(f"✓ Kafka Producer initialized: {self.topic}")
            logger.info(f"  - Brokers: {config['bootstrap_servers']}")
        except KafkaError as e:
            logger.error(f"Failed to initialize Kafka Producer: {e}")
            raise

    def send_prediction(
        self,
        prediction: PredictionResultMessage,
        key: Optional[str] = None
    ) -> bool:
        """
        예측 결과 메시지 발행

        Args:
            prediction: 예측 결과 메시지
            key: 메시지 키 (파티셔닝용, None이면 stock_code 사용)

        Returns:
            성공 여부
        """
        if not self.producer:
            raise RuntimeError("Producer not initialized")

        try:
            # 메시지 키 (동일 종목은 같은 파티션으로)
            if key is None:
                key = prediction.stock_code

            # JSON으로 직렬화
            message_value = prediction.to_json()

            # 발행
            future = self.producer.send(
                self.topic,
                key=key.encode('utf-8'),
                value=message_value
            )

            # 전송 완료 대기 (동기)
            record_metadata = future.get(timeout=10)

            logger.info(
                f"✓ Sent prediction: {prediction.stock_code} ({prediction.stock_name}) "
                f"expected_return={prediction.expected_return:.2f}% "
                f"signal={prediction.signal} "
                f"partition={record_metadata.partition} offset={record_metadata.offset}"
            )

            return True

        except KafkaError as e:
            logger.error(f"Failed to send prediction: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error sending prediction: {e}")
            return False

    def send_batch(
        self,
        predictions: List[PredictionResultMessage]
    ) -> dict:
        """
        배치로 예측 결과 발행 (개별 메시지로)

        Args:
            predictions: 예측 결과 리스트

        Returns:
            통계 정보 {'success': int, 'failed': int}
        """
        if not self.producer:
            raise RuntimeError("Producer not initialized")

        stats = {'success': 0, 'failed': 0}

        for prediction in predictions:
            if self.send_prediction(prediction):
                stats['success'] += 1
            else:
                stats['failed'] += 1

        logger.info(
            f"Batch send completed: {stats['success']} success, {stats['failed']} failed"
        )

        return stats

    def send_batch_message(
        self,
        batch_message: PredictionResultBatchMessage,
        key: Optional[str] = None
    ) -> bool:
        """
        배치 예측 결과 메시지 발행 (하나의 메시지로)

        Args:
            batch_message: 배치 예측 결과 메시지
            key: 메시지 키 (파티셔닝용, None이면 timestamp 사용)

        Returns:
            성공 여부
        """
        if not self.producer:
            raise RuntimeError("Producer not initialized")

        try:
            # 메시지 키 (배치 식별용)
            if key is None:
                key = f"batch_{batch_message.timestamp.isoformat()}"

            # JSON으로 직렬화
            message_value = batch_message.to_json()

            # 발행 (send_prediction과 동일하게 문자열로 전달)
            future = self.producer.send(
                self.topic,
                key=key.encode('utf-8'),
                value=message_value
            )

            # 전송 완료 대기 (동기)
            record_metadata = future.get(timeout=10)

            logger.info(
                f"✓ Sent batch prediction: {batch_message.total_count} predictions "
                f"partition={record_metadata.partition} offset={record_metadata.offset}"
            )

            return True

        except KafkaError as e:
            logger.error(f"Failed to send batch prediction: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error sending batch prediction: {e}")
            return False

    def flush(self):
        """버퍼에 남은 메시지 모두 전송"""
        if self.producer:
            self.producer.flush()
            logger.info("Producer flushed")

    def close(self):
        """Producer 종료"""
        if self.producer:
            self.producer.flush()
            self.producer.close()
            logger.info("Producer closed")

    def __enter__(self):
        """Context manager 진입"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager 종료"""
        self.close()


def create_producer() -> PredictionProducer:
    """
    Producer 생성 헬퍼 함수

    Returns:
        PredictionProducer 인스턴스
    """
    return PredictionProducer()
