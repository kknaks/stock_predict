"""
AI 서버 설정

Pydantic Settings를 사용하여 .env 파일에서 설정 로드
"""

from functools import lru_cache
from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """AI 서버 설정"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # -------------------------------------------
    # AI Server
    # -------------------------------------------
    app_name: str = "Stock Predict AI"
    app_version: str = "0.1.0"
    debug: bool = True
    ai_host: str = "0.0.0.0"
    ai_port: int = 8001

    # -------------------------------------------
    # Database (PostgreSQL) - stock-predict-network
    # -------------------------------------------
    db_host: str = "localhost"
    db_port: int = 5432
    db_name: str = "stock_predict"
    db_user: str = "postgres"
    db_password: str = "postgres"
    db_echo: bool = False

    # Docker 네트워크 사용 시 (stock-predict-network)
    db_use_docker_network: bool = False
    db_docker_host: str = "stock-predict-db"  # docker-compose의 service 이름

    # -------------------------------------------
    # Kafka / Redpanda - stock-predict-network
    # -------------------------------------------
    # 외부 접속용 (로컬 개발)
    kafka_bootstrap_servers: str = "localhost:19092"

    # 내부 접속용 (Docker 네트워크)
    kafka_bootstrap_servers_internal: str = "redpanda-0:9092,redpanda-1:9092,redpanda-2:9092"

    # Docker 네트워크 사용 여부
    kafka_use_internal: bool = False

    # Consumer 설정
    kafka_group_id: str = "ai-server-group"
    kafka_auto_offset_reset: str = "latest"
    kafka_enable_auto_commit: bool = True
    kafka_max_poll_records: int = 100

    # 토픽 이름
    topic_gap_candidate: str = "extract_daily_candidate"
    topic_prediction_result: str = "ai_prediction_result"

    # -------------------------------------------
    # Model Storage
    # -------------------------------------------
    model_path: str = "./models/stacking/stacking_hybrid_model.pkl"

    # -------------------------------------------
    # Feature 계산 설정
    # -------------------------------------------
    feature_lookback_days: int = 60

    # -------------------------------------------
    # 예측 설정
    # -------------------------------------------
    prediction_threshold: float = 0.4
    min_expected_return: float = 1.0
    min_prob_up: float = 0.2
    take_profit_ratio: float = 0.8
    stop_loss_ratio: float = 0.5
    max_positions: int = 20

    # -------------------------------------------
    # CORS
    # -------------------------------------------
    cors_origins: List[str] = ["http://localhost:3000", "http://localhost:5173"]

    @property
    def database_url(self) -> str:
        """SQLAlchemy 동기 연결 URL"""
        host = self.db_docker_host if self.db_use_docker_network else self.db_host
        return f"postgresql://{self.db_user}:{self.db_password}@{host}:{self.db_port}/{self.db_name}"

    @property
    def async_database_url(self) -> str:
        """SQLAlchemy 비동기 연결 URL (asyncpg)"""
        host = self.db_docker_host if self.db_use_docker_network else self.db_host
        return f"postgresql+asyncpg://{self.db_user}:{self.db_password}@{host}:{self.db_port}/{self.db_name}"

    @property
    def kafka_servers(self) -> str:
        """Kafka 브로커 주소 (환경에 따라)"""
        return self.kafka_bootstrap_servers_internal if self.kafka_use_internal else self.kafka_bootstrap_servers

    @property
    def kafka_servers_list(self) -> List[str]:
        """Kafka 브로커 주소 리스트"""
        return self.kafka_servers.split(',')


@lru_cache()
def get_settings() -> Settings:
    """
    설정 싱글톤 인스턴스 반환 (캐싱)
    
    Returns:
        Settings 인스턴스
    """
    return Settings()


# 편의를 위한 전역 인스턴스
settings = get_settings()
