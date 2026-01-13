"""
Kafka 메시지 스키마 정의

Pydantic을 사용하여 메시지 검증 및 직렬화/역직렬화
"""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field, field_validator


class GapCandidateMessage(BaseModel):
    """
    갭 상승 후보 메시지 (수신용)

    토픽: extract_daily_candidate
    발행자: 데이터 수집 시스템
    """
    timestamp: datetime = Field(..., description="메시지 발행 시각")
    stock_code: str = Field(..., description="종목 코드 (InfoCode)")
    stock_name: str = Field(..., description="종목명")
    exchange: str = Field(..., description="거래소 (KOSPI/KOSDAQ)")

    # 가격 정보
    stock_open: float = Field(..., gt=0, description="당일 시가")
    gap_rate: float = Field(..., description="갭 상승률 (%)")
    expected_change_rate: float = Field(default=0.0, description="예상 변동률")

    # 시장 지수
    kospi_open: Optional[float] = Field(None, description="KOSPI 시가")
    kosdaq_open: Optional[float] = Field(None, description="KOSDAQ 시가")
    kospi200_open: Optional[float] = Field(None, description="KOSPI200 시가")

    @field_validator('exchange')
    @classmethod
    def validate_exchange(cls, v: str) -> str:
        """거래소 검증"""
        allowed = {'KOSPI', 'KOSDAQ'}
        if v.upper() not in allowed:
            raise ValueError(f"exchange must be one of {allowed}")
        return v.upper()

    @field_validator('gap_rate')
    @classmethod
    def validate_gap_rate(cls, v: float) -> float:
        """갭 비율 검증 (-50% ~ 50%)"""
        if not -50.0 <= v <= 50.0:
            raise ValueError("gap_rate must be between -50.0 and 50.0")
        return v

    def to_json(self) -> str:
        """JSON 문자열로 변환 (Kafka 발행용)"""
        return self.model_dump_json()

    @classmethod
    def from_json(cls, json_str: str) -> 'GapCandidateMessage':
        """JSON 문자열에서 생성 (Kafka 수신용)"""
        return cls.model_validate_json(json_str)


class PredictionResultMessage(BaseModel):
    """
    AI 예측 결과 메시지 (발행용)

    토픽: ai_prediction_result
    발행자: AI 서버
    """
    # 원본 정보
    timestamp: datetime = Field(default_factory=datetime.now, description="예측 시각")
    stock_code: str = Field(..., description="종목 코드")
    stock_name: str = Field(..., description="종목명")
    exchange: str = Field(..., description="거래소")
    date: str = Field(..., description="예측 대상 날짜 (YYYY-MM-DD)")

    # 입력 데이터
    gap_rate: float = Field(..., description="갭 상승률 (%)")
    stock_open: float = Field(..., description="당일 시가")

    # 예측 결과
    prob_up: float = Field(..., ge=0, le=1, description="상승 확률")
    prob_down: float = Field(..., ge=0, le=1, description="하락 확률")
    predicted_direction: int = Field(..., ge=0, le=1, description="예측 방향 (0:하락, 1:상승)")

    expected_return: float = Field(..., description="기대 수익률 (%)")
    return_if_up: float = Field(..., description="상승 시 예상 수익률 (%)")
    return_if_down: float = Field(..., description="하락 시 예상 손실률 (%)")

    # 고가 예측 (선택)
    max_return_if_up: Optional[float] = Field(None, description="상승 시 최대 수익률 (%)")
    take_profit_target: Optional[float] = Field(None, description="익절 목표 수익률 (%)")

    # 매매 신호
    signal: str = Field(default='HOLD', description="매매 신호 (BUY/HOLD)")

    # 메타 정보
    model_version: str = Field(default='v1.0', description="모델 버전")
    confidence: Optional[str] = Field(None, description="신뢰도 (HIGH/MEDIUM/LOW)")

    @field_validator('signal')
    @classmethod
    def validate_signal(cls, v: str) -> str:
        """신호 검증"""
        allowed = {'BUY', 'HOLD', 'SELL'}
        if v.upper() not in allowed:
            raise ValueError(f"signal must be one of {allowed}")
        return v.upper()

    def to_json(self) -> str:
        """JSON 문자열로 변환"""
        return self.model_dump_json()

    @classmethod
    def from_json(cls, json_str: str) -> 'PredictionResultMessage':
        """JSON 문자열에서 생성"""
        return cls.model_validate_json(json_str)


class MarketIndexData(BaseModel):
    """
    시장 지수 데이터 (내부 사용)
    """
    date: str = Field(..., description="날짜 (YYYY-MM-DD)")
    kospi_open: Optional[float] = None
    kospi_close: Optional[float] = None
    kospi_gap_pct: Optional[float] = None

    kosdaq_open: Optional[float] = None
    kosdaq_close: Optional[float] = None
    kosdaq_gap_pct: Optional[float] = None

    kospi200_open: Optional[float] = None
    kospi200_close: Optional[float] = None
    kospi200_gap_pct: Optional[float] = None
