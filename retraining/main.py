"""
재학습 서버 엔트리포인트

Kafka consumer 루프 실행
model_retrain_command 수신 → 재학습 → model_retrain_result 발행
"""

import logging
import sys

from .kafka_client import RetrainConsumer, RetrainProducer
from .handler import handle_retrain_command
from .schemas import ModelRetrainCommandMessage

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger(__name__)


def main():
    """메인 루프"""
    logger.info("Starting retraining server...")

    consumer = RetrainConsumer()
    producer = RetrainProducer()

    def on_command(command: ModelRetrainCommandMessage):
        logger.info(f"Processing retrain command: trigger={command.trigger_type}")
        result = handle_retrain_command(command)
        producer.send_result(result)
        logger.info(
            f"Retrain completed: status={result.status}, "
            f"version={result.new_version}, deployed={result.is_deployed}"
        )

    try:
        consumer.consume_stream(handler=on_command)
    except KeyboardInterrupt:
        logger.info("Shutting down retraining server...")
    finally:
        consumer.close()
        producer.close()
        logger.info("Retraining server stopped")


if __name__ == "__main__":
    main()
