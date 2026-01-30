"""
모델 학습 래퍼 - 버전 관리 + StackingModelTrainer 래핑

버전별 디렉토리에 모델 저장
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional

import pandas as pd

from src.models.trainer import StackingModelTrainer
from src.models.storage import ModelStorage

logger = logging.getLogger(__name__)


def train_versioned_model(
    df_model: pd.DataFrame,
    version: str,
    model_base_dir: str = "./models/stacking",
    threshold: float = 0.4,
    n_estimators: int = 300,
    test_size: float = 0.1,
    valid_size: float = 0.2,
    test_recent: bool = True,
) -> Dict[str, Any]:
    """
    버전별 모델 학습 및 저장

    Args:
        df_model: 전처리된 학습 데이터
        version: 모델 버전 (e.g. "v1.2501.15")
        model_base_dir: 모델 베이스 디렉토리
        threshold: 분류 임계값
        n_estimators: 앙상블 모델 수
        test_size: 테스트 비율
        valid_size: 검증 비율

    Returns:
        {'test_metrics': ..., 'model_path': ..., 'features': ...}
    """
    version_dir = Path(model_base_dir) / version
    version_dir.mkdir(parents=True, exist_ok=True)

    # 임시 parquet 저장 (StackingModelTrainer가 파일 경로를 요구)
    temp_parquet = version_dir / "training_data.parquet"
    df_model.to_parquet(temp_parquet, index=False)

    logger.info(f"Training model version={version}, "
                f"data_shape={df_model.shape}, dir={version_dir}")

    # StackingModelTrainer 실행
    trainer = StackingModelTrainer(
        data_path=str(temp_parquet),
        model_dir=str(Path(model_base_dir).parent),  # models/
        random_state=42,
        test_size=test_size,
        valid_size=valid_size,
        threshold=threshold,
        n_estimators=n_estimators,
        verbose=True,
        test_recent=test_recent,
    )

    result = trainer.train_all()

    # 버전별 디렉토리에 저장
    storage = ModelStorage(base_dir=str(Path(model_base_dir).parent))
    model_path = storage.save_all_models(
        classifier=result["classifier"],
        regressor_up=result["regressor_up"],
        regressor_high=result["regressor_high"],
        regressor_down=result["regressor_down"],
        test_metrics=result["test_metrics"],
        subdir=f"stacking/{version}",
    )

    # 임시 parquet 삭제
    if temp_parquet.exists():
        temp_parquet.unlink()

    logger.info(f"Model saved: version={version}, path={model_path}")

    return {
        "test_metrics": result["test_metrics"],
        "model_path": str(model_path),
        "features": result["features"],
    }
