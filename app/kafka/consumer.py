"""
Kafka Consumer - 갭 상승 후보 메시지 수신
"""

import json
import logging
from typing import Optional, Callable, List
from kafka import KafkaConsumer
from kafka.errors import KafkaError

from app.config.kafka_connections import get_kafka_config
from .schemas import GapCandidateMessage, GapCandidateBatchMessage

logger = logging.getLogger(__name__)


class GapCandidateConsumer:
    """
    갭 상승 후보 메시지 수신 Consumer

    토픽: extract_daily_candidate
    """

    def __init__(self, custom_config: Optional[dict] = None):
        """
        Args:
            custom_config: 커스텀 Consumer 설정 (선택)
        """
        self.config = get_kafka_config()
        self.topic = self.config.topic_gap_candidate

        # Consumer 설정
        consumer_config = self.config.get_consumer_config()
        if custom_config:
            consumer_config.update(custom_config)

        # Consumer 생성
        self.consumer: Optional[KafkaConsumer] = None
        self._init_consumer(consumer_config)

    def _init_consumer(self, config: dict):
        """Consumer 초기화"""
        try:
            self.consumer = KafkaConsumer(
                self.topic,
                **config
            )
            logger.info(f"✓ Kafka Consumer initialized: {self.topic}")
            logger.info(f"  - Brokers: {config['bootstrap_servers']}")
            logger.info(f"  - Group ID: {config['group_id']}")
        except KafkaError as e:
            logger.error(f"Failed to initialize Kafka Consumer: {e}")
            raise

    def consume_batch(
        self,
        max_messages: int = 100,
        timeout_ms: int = 10000
    ) -> List[GapCandidateMessage]:
        """
        메시지 수신 (non-blocking)
        
        Note: 배치 메시지 형식으로 발행되므로, 하나의 배치 메시지를 받아서
        개별 종목 메시지 리스트로 변환하여 반환합니다.

        Args:
            max_messages: 최대 수신 메시지 수 (기본값: 100, 실제로는 배치 1개)
            timeout_ms: 타임아웃 (밀리초)

        Returns:
            개별 종목 메시지 리스트
        """
        if not self.consumer:
            raise RuntimeError("Consumer not initialized")

        messages = []

        try:
            # poll로 메시지 가져오기
            records = self.consumer.poll(
                timeout_ms=timeout_ms,
                max_records=max_messages
            )

            # 파티션별 메시지 처리
            for topic_partition, records_list in records.items():
                for record in records_list:
                    try:
                        # 배치 메시지 파싱
                        batch_message = GapCandidateBatchMessage.from_json(record.value)
                        
                        # 개별 메시지로 변환
                        individual_messages = batch_message.to_individual_messages()
                        messages.extend(individual_messages)

                        logger.info(
                            f"✓ Received batch message: {batch_message.total_count} stocks "
                            f"(timestamp: {batch_message.timestamp})"
                        )
                        
                        for msg in individual_messages:
                            logger.debug(
                                f"  - {msg.stock_code} ({msg.stock_name}) "
                                f"gap={msg.gap_rate:.2f}%"
                            )
                    except Exception as e:
                        logger.error(f"Failed to parse batch message: {e}")
                        logger.error(f"Raw message: {record.value}")

            if messages:
                logger.info(f"✓ Consumed {len(messages)} individual messages from {self.topic}")

        except KafkaError as e:
            logger.error(f"Error consuming messages: {e}")
            raise

        return messages

    def consume_stream(
        self,
        handler: Callable[[GapCandidateMessage], None]
    ):
        """
        스트리밍 모드로 메시지 수신 (blocking)
        
        Note: 배치 메시지 형식으로 발행되므로, 하나의 배치 메시지를 받아서
        개별 종목 메시지로 변환한 후 각각을 처리합니다.

        Args:
            handler: 개별 종목 메시지 처리 함수
        """
        if not self.consumer:
            raise RuntimeError("Consumer not initialized")

        logger.info(f"Starting streaming consumer for {self.topic}...")

        try:
            for message in self.consumer:
                try:
                    # 배치 메시지 파싱
                    batch_message = GapCandidateBatchMessage.from_json(message.value)
                    
                    # 개별 메시지로 변환하여 각각 처리
                    individual_messages = batch_message.to_individual_messages()
                    
                    logger.info(
                        f"✓ Received batch: {batch_message.total_count} stocks, "
                        f"processing individually..."
                    )
                    
                    for gap_message in individual_messages:
                        handler(gap_message)

                except Exception as e:
                    logger.error(f"Error processing batch message: {e}")
                    logger.error(f"Raw message: {message.value}")

        except KeyboardInterrupt:
            logger.info("Consumer interrupted by user")

    def close(self):
        """Consumer 종료"""
        if self.consumer:
            self.consumer.close()
            logger.info("Consumer closed")

    def __enter__(self):
        """Context manager 진입"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager 종료"""
        self.close()


def create_consumer(
    group_id: Optional[str] = None,
    auto_offset_reset: str = 'latest'
) -> GapCandidateConsumer:
    """
    Consumer 생성 헬퍼 함수

    Args:
        group_id: Consumer Group ID (None이면 설정 파일 사용)
        auto_offset_reset: 오프셋 리셋 정책

    Returns:
        GapCandidateConsumer 인스턴스
    """
    custom_config = {}

    if group_id:
        custom_config['group_id'] = group_id

    if auto_offset_reset:
        custom_config['auto_offset_reset'] = auto_offset_reset

    return GapCandidateConsumer(custom_config=custom_config)
