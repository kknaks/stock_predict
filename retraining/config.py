"""
재학습 서버 설정

app과 독립적으로 환경변수에서 설정 로드
"""

from functools import lru_cache
from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict


class RetrainingSettings(BaseSettings):
    """재학습 서버 설정"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # -------------------------------------------
    # Database (PostgreSQL)
    # -------------------------------------------
    db_host: str = "localhost"
    db_port: int = 5432
    db_name: str = "stock_predict"
    db_user: str = "postgres"
    db_password: str = "postgres"
    db_echo: bool = False

    db_use_docker_network: bool = False
    db_docker_host: str = "stock-predict-db"

    # -------------------------------------------
    # Kafka / Redpanda
    # -------------------------------------------
    kafka_bootstrap_servers: str = "localhost:19092"
    kafka_bootstrap_servers_internal: str = "redpanda-0:9092,redpanda-1:9092,redpanda-2:9092"
    kafka_use_internal: bool = False
    kafka_group_id: str = "retraining-server-group"
    kafka_auto_offset_reset: str = "latest"

    # 토픽
    topic_retrain_command: str = "model_retrain_command"
    topic_retrain_result: str = "model_retrain_result"

    # -------------------------------------------
    # 데이터 경로
    # -------------------------------------------
    parquet_path: str = "./data/processed/preprocessed_df_full_outlier_fix.parquet"

    # -------------------------------------------
    # 모델 경로
    # -------------------------------------------
    model_base_dir: str = "./models/stacking"
    active_symlink_name: str = "active"

    # -------------------------------------------
    # 학습 기본값
    # -------------------------------------------
    default_threshold: float = 0.4
    default_n_estimators: int = 300
    default_test_size: float = 0.1
    default_valid_size: float = 0.1

    @property
    def database_url(self) -> str:
        host = self.db_docker_host if self.db_use_docker_network else self.db_host
        return f"postgresql://{self.db_user}:{self.db_password}@{host}:{self.db_port}/{self.db_name}"

    @property
    def kafka_servers(self) -> str:
        return self.kafka_bootstrap_servers_internal if self.kafka_use_internal else self.kafka_bootstrap_servers

    @property
    def kafka_servers_list(self) -> List[str]:
        return self.kafka_servers.split(",")


@lru_cache()
def get_retraining_settings() -> RetrainingSettings:
    return RetrainingSettings()


settings = get_retraining_settings()
