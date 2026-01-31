"""
재학습 서버 Kafka consumer/producer (app 미의존)

Consumer: model_retrain_command 토픽
Producer: model_retrain_result 토픽
"""

import logging
from typing import Optional, Callable

from kafka import KafkaConsumer, KafkaProducer
from kafka.errors import KafkaError

from .config import settings
from .schemas import ModelRetrainCommandMessage, ModelRetrainResultMessage

logger = logging.getLogger(__name__)


class RetrainConsumer:
    """model_retrain_command 토픽 Consumer"""

    def __init__(self):
        self.topic = settings.topic_retrain_command
        config = {
            "bootstrap_servers": settings.kafka_servers_list,
            "group_id": settings.kafka_group_id,
            "auto_offset_reset": settings.kafka_auto_offset_reset,
            "enable_auto_commit": False,
            "value_deserializer": lambda m: m.decode("utf-8"),
        }

        try:
            self.consumer = KafkaConsumer(self.topic, **config)
            logger.info(f"Retrain Consumer initialized: {self.topic}")
        except KafkaError as e:
            logger.error(f"Failed to initialize Retrain Consumer: {e}")
            raise

    def consume_stream(self, handler: Callable[[ModelRetrainCommandMessage], None]):
        """스트리밍 모드로 재학습 커맨드 수신

        메시지 수신 즉시 커밋 후, 컨슈머를 닫고 학습 실행.
        학습 중 Kafka 연결을 유지하지 않으므로 heartbeat/리밸런싱 문제 없음.
        """
        logger.info(f"Starting retrain consumer for {self.topic}...")

        try:
            for message in self.consumer:
                try:
                    command = ModelRetrainCommandMessage.from_json(message.value)
                    logger.info(
                        f"Received retrain command: trigger={command.trigger_type} "
                        f"timestamp={command.timestamp}"
                    )
                    # 메시지 수신 즉시 커밋 → 학습 중 중복 소비 방지
                    self.consumer.commit()
                    self.consumer.close()
                    self.consumer = None
                    logger.info("Consumer closed before training start")

                    handler(command)

                    # 학습 완료 후 다음 메시지 대기를 위해 재연결
                    self._reconnect()
                except Exception as e:
                    logger.error(f"Error processing retrain command: {e}", exc_info=True)
                    if self.consumer is None:
                        self._reconnect()
        except KeyboardInterrupt:
            logger.info("Retrain consumer interrupted by user")
        finally:
            self.close()

    def _reconnect(self):
        """컨슈머 재연결"""
        config = {
            "bootstrap_servers": settings.kafka_servers_list,
            "group_id": settings.kafka_group_id,
            "auto_offset_reset": settings.kafka_auto_offset_reset,
            "enable_auto_commit": False,
            "value_deserializer": lambda m: m.decode("utf-8"),
        }
        self.consumer = KafkaConsumer(self.topic, **config)
        logger.info(f"Consumer reconnected: {self.topic}")

    def close(self):
        if self.consumer:
            self.consumer.close()
            logger.info("Retrain consumer closed")


class RetrainProducer:
    """model_retrain_result 토픽 Producer"""

    def __init__(self):
        self.topic = settings.topic_retrain_result
        config = {
            "bootstrap_servers": settings.kafka_servers_list,
            "value_serializer": lambda m: m.encode("utf-8"),
            "acks": "all",
            "retries": 3,
        }

        try:
            self.producer = KafkaProducer(**config)
            logger.info(f"Retrain Producer initialized: {self.topic}")
        except KafkaError as e:
            logger.error(f"Failed to initialize Retrain Producer: {e}")
            raise

    def send_result(self, result: ModelRetrainResultMessage) -> bool:
        """재학습 결과 발행"""
        try:
            future = self.producer.send(
                self.topic,
                key=f"retrain_{result.new_version or 'unknown'}".encode("utf-8"),
                value=result.to_json(),
            )
            record = future.get(timeout=10)
            logger.info(
                f"Sent retrain result: status={result.status} "
                f"version={result.new_version} deployed={result.is_deployed} "
                f"partition={record.partition} offset={record.offset}"
            )
            return True
        except Exception as e:
            logger.error(f"Failed to send retrain result: {e}")
            return False

    def close(self):
        if self.producer:
            self.producer.flush()
            self.producer.close()
            logger.info("Retrain producer closed")
