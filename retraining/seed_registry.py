"""
기존 v1.0.0 모델을 model_registry DB에 시드 등록하는 스크립트

pkl 파일에서 test_metrics를 추출하여 DB에 active 상태로 등록
Docker 환경 또는 xgboost/libomp가 설치된 환경에서 실행 필요

Usage:
    python -m retraining.seed_registry
"""

import sys
import logging
from datetime import datetime
from pathlib import Path

import joblib

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def seed_v1_model(
    pkl_path: str = "./models/stacking/v1.0.0/stacking_hybrid_model.pkl",
    version: str = "v1.0.0",
):
    """
    기존 모델을 model_registry에 active로 등록

    Args:
        pkl_path: pkl 파일 경로
        version: 등록할 버전
    """
    from retraining.database import get_session_factory
    from retraining import model_registry_service as registry
    from app.database.database.model_registry import ModelRegistry, ModelStatus

    # 1. pkl 로드 → test_metrics 추출
    path = Path(pkl_path)
    if not path.exists():
        logger.error(f"pkl not found: {path}")
        return

    logger.info(f"Loading pkl: {path}")

    # __main__.StackingHybridPredictor 참조 우회
    from app.prediction.predictor import HybridPredictor
    sys.modules.setdefault("__main__", type(sys)("__main__"))
    if not hasattr(sys.modules["__main__"], "StackingHybridPredictor"):
        sys.modules["__main__"].StackingHybridPredictor = HybridPredictor

    data = joblib.load(pkl_path)

    test_metrics = data.get("test_metrics")
    created_at = data.get("created_at")
    threshold = data.get("optimal_threshold", 0.4)
    features = data.get("features", [])

    logger.info(f"Extracted test_metrics: {test_metrics is not None}")
    logger.info(f"Created at: {created_at}")
    logger.info(f"Threshold: {threshold}")
    logger.info(f"Features: {len(features)}")

    if test_metrics:
        # test_metrics의 값들이 BaseStackingModel 객체일 수 있으므로 dict로 변환
        serializable_metrics = {}
        for model_name, metrics in test_metrics.items():
            if isinstance(metrics, dict):
                # float로 변환 (numpy float → Python float)
                serializable_metrics[model_name] = {
                    k: float(v) if hasattr(v, "__float__") else v
                    for k, v in metrics.items()
                }
            else:
                serializable_metrics[model_name] = str(metrics)
        test_metrics = serializable_metrics

    logger.info(f"Test metrics (serialized): {test_metrics}")

    # 2. DB에 등록
    session_factory = get_session_factory()
    with session_factory() as session:
        # 이미 등록되어 있는지 확인
        existing = (
            session.query(ModelRegistry)
            .filter(ModelRegistry.version == version)
            .first()
        )
        if existing:
            logger.info(f"Version {version} already registered (status={existing.status})")
            return

        entry = registry.create_entry(
            session=session,
            version=version,
            model_path=str(path),
            training_data_start=None,
            training_data_end=None,
            training_samples=None,
            training_duration_seconds=None,
            trigger_type="initial",
            hyperparameters={
                "threshold": threshold,
                "n_estimators": 300,
                "features_count": len(features),
            },
            test_metrics=test_metrics,
            status=ModelStatus.CANDIDATE,
        )

        # active로 변경
        registry.activate_model(session, version)
        session.commit()

        logger.info(f"Seeded v1.0.0 as ACTIVE in model_registry (id={entry.id})")


if __name__ == "__main__":
    seed_v1_model()
