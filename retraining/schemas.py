"""
Kafka 메시지 스키마 - 모델 재학습용

토픽:
- model_retrain_command: 재학습 트리거 메시지
- model_retrain_result: 재학습 결과 메시지
"""

from datetime import datetime
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field


class RetrainConfig(BaseModel):
    """재학습 설정"""
    data_start_date: Optional[str] = Field(
        None, description="학습 데이터 시작일 (YYYY-MM-DD), None이면 전체"
    )
    data_end_date: Optional[str] = Field(
        None, description="학습 데이터 종료일 (YYYY-MM-DD), None이면 최신"
    )
    threshold: float = Field(default=0.4, description="분류 임계값")
    n_estimators: int = Field(default=300, description="앙상블 모델 수")
    test_size: float = Field(default=0.1, description="테스트 비율")
    valid_size: float = Field(default=0.2, description="검증 비율")
    test_recent: bool = Field(default=True, description="True: 최근 데이터로 test, False: 랜덤 분할")
    min_roc_auc: float = Field(default=0.55, description="ROC AUC 최소 기준")
    max_roc_auc_drop: float = Field(default=0.02, description="ROC AUC 최대 허용 하락폭")
    max_f1_drop: float = Field(default=0.05, description="F1 최대 허용 하락폭")
    min_backtest_return: float = Field(default=0.0, description="백테스트 최소 수익률 (%)")
    min_sharpe_ratio: float = Field(default=1.0, description="백테스트 최소 Sharpe Ratio")
    min_profit_factor: float = Field(default=1.0, description="백테스트 최소 Profit Factor")


class ModelRetrainCommandMessage(BaseModel):
    """
    모델 재학습 트리거 메시지

    토픽: model_retrain_command
    발행자: Airflow DAG 또는 수동 트리거
    """
    timestamp: datetime = Field(default_factory=datetime.now, description="메시지 발행 시각")
    trigger_type: str = Field(
        default="scheduled",
        description="트리거 유형 (scheduled/manual/drift_detected)"
    )
    config: RetrainConfig = Field(default_factory=RetrainConfig, description="재학습 설정")

    def to_json(self) -> str:
        return self.model_dump_json()

    @classmethod
    def from_json(cls, json_str: str | bytes) -> "ModelRetrainCommandMessage":
        if isinstance(json_str, bytes):
            json_str = json_str.decode("utf-8")
        return cls.model_validate_json(json_str)


class ComparisonResult(BaseModel):
    """Champion-Challenger 비교 결과"""
    champion_version: Optional[str] = Field(None, description="기존 챔피언 버전")
    challenger_version: str = Field(..., description="새 모델 버전")
    champion_roc_auc: Optional[float] = Field(None, description="챔피언 ROC AUC")
    challenger_roc_auc: float = Field(..., description="챌린저 ROC AUC")
    champion_f1: Optional[float] = Field(None, description="챔피언 F1")
    challenger_f1: float = Field(..., description="챌린저 F1")
    decision: str = Field(..., description="결정 (deploy/reject)")
    reason: str = Field(..., description="결정 사유")


class ModelRetrainResultMessage(BaseModel):
    """
    모델 재학습 결과 메시지

    토픽: model_retrain_result
    발행자: retraining 컨테이너
    """
    timestamp: datetime = Field(default_factory=datetime.now, description="결과 시각")
    status: str = Field(..., description="상태 (success/failed/rejected)")
    new_version: Optional[str] = Field(None, description="새 모델 버전")
    is_deployed: bool = Field(default=False, description="배포 여부")
    test_metrics: Optional[Dict[str, Any]] = Field(None, description="테스트 메트릭")
    comparison_result: Optional[ComparisonResult] = Field(None, description="비교 결과")
    error_message: Optional[str] = Field(None, description="에러 메시지 (실패 시)")
    trigger_type: str = Field(default="scheduled", description="원본 트리거 유형")
    training_duration_seconds: Optional[float] = Field(None, description="학습 소요 시간(초)")

    def to_json(self) -> str:
        return self.model_dump_json()

    @classmethod
    def from_json(cls, json_str: str | bytes) -> "ModelRetrainResultMessage":
        if isinstance(json_str, bytes):
            json_str = json_str.decode("utf-8")
        return cls.model_validate_json(json_str)
