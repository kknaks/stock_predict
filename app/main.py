"""
AI 서버 메인 엔트리 포인트

Kafka Consumer를 실행하여 갭 상승 후보 메시지를 수신하고 예측 수행
"""

import logging
import signal
import sys
from typing import Optional

from app.kafka.consumer import GapCandidateConsumer
from app.handler.gap_predict_handler import handle_gap_candidate_message
from app.config.settings import settings
from app.prediction.predictor import HybridPredictor

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


def signal_handler(sig, frame):
    """시그널 핸들러 (Ctrl+C 등)"""
    logger.info("종료 신호 수신...")
    if consumer:
        consumer.close()
    sys.exit(0)


def main():
    """메인 함수"""
    global consumer
    
    logger.info("=" * 80)
    logger.info("Stock Predict AI Server 시작")
    logger.info("=" * 80)
    logger.info(f"  - Kafka Topic: {settings.topic_gap_candidate}")
    logger.info(f"  - Group ID: {settings.kafka_group_id}")
    logger.info(f"  - Model Path: {settings.model_path}")
    logger.info(f"  - DB Host: {settings.db_host}")
    logger.info("=" * 80)
    
    # 시그널 핸들러 등록
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Consumer 생성
        consumer = GapCandidateConsumer()
        
        logger.info("메시지 수신 대기 중...")
        logger.info("(종료하려면 Ctrl+C)")
        
        # 스트리밍 모드로 메시지 수신 및 처리
        def message_handler(message):
            """메시지 처리 핸들러"""
            try:
                result = handle_gap_candidate_message(message)
                if result:
                    logger.info(
                        f"✓ 처리 완료: {result.stock_code} "
                        f"signal={result.signal} "
                        f"expected_return={result.expected_return:.2f}%"
                    )
                else:
                    logger.warning(f"✗ 처리 실패: {message.stock_code}")
            except Exception as e:
                logger.error(f"메시지 처리 중 오류: {e}", exc_info=True)
        
        # 스트리밍 모드로 실행 (blocking)
        consumer.consume_stream(handler=message_handler)
        
    except KeyboardInterrupt:
        logger.info("사용자에 의해 종료됨")
    except Exception as e:
        logger.error(f"서버 실행 중 오류: {e}", exc_info=True)
        sys.exit(1)
    finally:
        if consumer:
            consumer.close()
        logger.info("서버 종료")


if __name__ == "__main__":
    main()
