"""
모델 레지스트리 DB CRUD

모델 등록, 활성화, 조회, 비교 결과 저장
"""

import logging
from datetime import datetime
from typing import Optional, Dict, Any

from sqlalchemy.orm import Session

from app.database.database.model_registry import ModelRegistry, ModelStatus

logger = logging.getLogger(__name__)


def create_entry(
    session: Session,
    version: str,
    model_path: str,
    training_data_start: Optional[str] = None,
    training_data_end: Optional[str] = None,
    training_samples: Optional[int] = None,
    training_duration_seconds: Optional[float] = None,
    trigger_type: Optional[str] = None,
    hyperparameters: Optional[Dict[str, Any]] = None,
    test_metrics: Optional[Dict[str, Any]] = None,
    status: ModelStatus = ModelStatus.CANDIDATE,
) -> ModelRegistry:
    """새 모델 레지스트리 엔트리 생성 (같은 버전이면 덮어쓰기)"""
    existing = (
        session.query(ModelRegistry)
        .filter(ModelRegistry.version == version)
        .first()
    )
    if existing:
        session.delete(existing)
        session.flush()
        logger.info(f"Deleted existing registry entry: version={version}")

    entry = ModelRegistry(
        version=version,
        model_path=model_path,
        training_data_start=training_data_start,
        training_data_end=training_data_end,
        training_samples=training_samples,
        training_duration_seconds=training_duration_seconds,
        trigger_type=trigger_type,
        hyperparameters=hyperparameters,
        test_metrics=test_metrics,
        status=status,
    )
    session.add(entry)
    session.flush()
    logger.info(f"Created registry entry: version={version}, status={status}")
    return entry


def get_active_model(session: Session) -> Optional[ModelRegistry]:
    """현재 활성 모델 조회"""
    return (
        session.query(ModelRegistry)
        .filter(ModelRegistry.status == ModelStatus.ACTIVE)
        .order_by(ModelRegistry.activated_at.desc())
        .first()
    )


def activate_model(session: Session, version: str) -> ModelRegistry:
    """
    모델 활성화

    기존 ACTIVE 모델은 RETIRED로 변경
    """
    # 기존 활성 모델 retire
    active_models = (
        session.query(ModelRegistry)
        .filter(ModelRegistry.status == ModelStatus.ACTIVE)
        .all()
    )
    for model in active_models:
        model.status = ModelStatus.RETIRED
        logger.info(f"Retired model: {model.version}")

    # 새 모델 활성화
    new_model = (
        session.query(ModelRegistry)
        .filter(ModelRegistry.version == version)
        .first()
    )
    if not new_model:
        raise ValueError(f"Model version not found: {version}")

    new_model.status = ModelStatus.ACTIVE
    new_model.activated_at = datetime.now()
    session.flush()

    logger.info(f"Activated model: {version}")
    return new_model


def update_comparison_result(
    session: Session,
    version: str,
    comparison_result: Dict[str, Any],
) -> ModelRegistry:
    """비교 결과 업데이트"""
    model = (
        session.query(ModelRegistry)
        .filter(ModelRegistry.version == version)
        .first()
    )
    if not model:
        raise ValueError(f"Model version not found: {version}")

    model.comparison_result = comparison_result
    session.flush()

    logger.info(f"Updated comparison result for: {version}")
    return model


def reject_model(session: Session, version: str) -> ModelRegistry:
    """모델 거부 처리"""
    model = (
        session.query(ModelRegistry)
        .filter(ModelRegistry.version == version)
        .first()
    )
    if not model:
        raise ValueError(f"Model version not found: {version}")

    model.status = ModelStatus.REJECTED
    session.flush()

    logger.info(f"Rejected model: {version}")
    return model
