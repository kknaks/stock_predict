"""
Kafka 연결 설정

settings에서 설정을 가져와 Kafka용으로 변환
"""

from typing import List
from .settings import settings


class KafkaConfig:
    """
    Kafka 연결 설정 (Settings 기반)
    """

    def __init__(self):
        """Settings에서 Kafka 설정 로드"""
        self._settings = settings

    @property
    def bootstrap_servers(self) -> str:
        """환경에 따른 브로커 주소 반환"""
        return self._settings.kafka_servers

    @property
    def bootstrap_servers_list(self) -> List[str]:
        """브로커 주소를 리스트로 반환"""
        return self._settings.kafka_servers_list

    @property
    def kafka_group_id(self) -> str:
        """Consumer Group ID"""
        return self._settings.kafka_group_id

    @property
    def kafka_auto_offset_reset(self) -> str:
        """오프셋 리셋 정책"""
        return self._settings.kafka_auto_offset_reset

    @property
    def kafka_enable_auto_commit(self) -> bool:
        """자동 오프셋 커밋 활성화"""
        return self._settings.kafka_enable_auto_commit

    @property
    def kafka_max_poll_records(self) -> int:
        """한 번에 가져올 최대 레코드 수"""
        return self._settings.kafka_max_poll_records

    @property
    def topic_gap_candidate(self) -> str:
        """갭 상승 후보 토픽"""
        return self._settings.topic_gap_candidate

    @property
    def topic_prediction_result(self) -> str:
        """AI 예측 결과 토픽"""
        return self._settings.topic_prediction_result

    def get_consumer_config(self) -> dict:
        """Consumer 설정 딕셔너리 반환"""
        return {
            'bootstrap_servers': self.bootstrap_servers_list,
            'group_id': self.kafka_group_id,
            'auto_offset_reset': self.kafka_auto_offset_reset,
            'enable_auto_commit': self.kafka_enable_auto_commit,
            'max_poll_records': self.kafka_max_poll_records,
            'value_deserializer': lambda m: m.decode('utf-8'),
        }

    def get_producer_config(self) -> dict:
        """Producer 설정 딕셔너리 반환"""
        return {
            'bootstrap_servers': self.bootstrap_servers_list,
            'value_serializer': lambda m: m.encode('utf-8'),
            'acks': 'all',  # 모든 레플리카 확인
            'retries': 3,
            'max_in_flight_requests_per_connection': 1,  # 순서 보장
        }


# 싱글톤 인스턴스
_config_instance: KafkaConfig | None = None


def get_kafka_config() -> KafkaConfig:
    """
    Kafka 설정 싱글톤 인스턴스 반환
    """
    global _config_instance
    if _config_instance is None:
        _config_instance = KafkaConfig()
    return _config_instance
