"""
모델 모듈

Stacking 앙상블 모델 학습 및 관리
- Model 1: Stacking Classifier (상승 확률 예측)
- Model 2: Stacking Regressor (상승 시 종가 수익률)
- Model 2-1: Stacking Regressor (상승 시 고가 수익률)
- Model 3: Stacking Regressor (하락 시 손실률)
"""

from .base import (
    FEATURE_COLS,
    TARGET_DIRECTION,
    TARGET_RETURN,
    TARGET_MAX_RETURN,
    CORE_FEATURES,
    DEFAULT_RANDOM_STATE,
    DEFAULT_N_ESTIMATORS,
    prepare_data_for_training,
    split_data,
)
from .classifier import StackingClassifierModel
from .regressor_up import StackingRegressorUp
from .regressor_high import StackingRegressorHigh
from .regressor_down import StackingRegressorDown
from .storage import ModelStorage
from .trainer import StackingModelTrainer, train_models

__all__ = [
    # 상수
    'FEATURE_COLS',
    'TARGET_DIRECTION',
    'TARGET_RETURN',
    'TARGET_MAX_RETURN',
    'CORE_FEATURES',
    'DEFAULT_RANDOM_STATE',
    'DEFAULT_N_ESTIMATORS',
    
    # 유틸리티
    'prepare_data_for_training',
    'split_data',
    
    # 모델
    'StackingClassifierModel',
    'StackingRegressorUp',
    'StackingRegressorHigh',
    'StackingRegressorDown',
    'ModelStorage',
    
    # 학습 파이프라인
    'StackingModelTrainer',
    'train_models',
]
