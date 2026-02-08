"""분리 학습용 모델 (v3).

mode 파라미터로 출력 head를 선택:
- 'cls': prob만 출력 (분류 모델)
- 'high': high_return만 출력 (High 회귀 모델)
- 'low': low_return만 출력 (Low 회귀 모델)
- 'all': 전부 출력 (호환성)

Neural Network: LSTMPredictor, CNNPredictor, MLPPredictor
Tree Models: XGBoostWrapper, RandomForestWrapper
"""

from boliger.models_sep.base import BaseModel, OPEN_INFO_DIM
from boliger.models_sep.lstm import LSTMPredictor
from boliger.models_sep.cnn import CNNPredictor
from boliger.models_sep.mlp import MLPPredictor
from boliger.models_sep.xgboost import XGBoostWrapper, create_xgboost_model
from boliger.models_sep.rf import RandomForestWrapper, create_rf_model

__all__ = [
    # Base
    "BaseModel",
    "OPEN_INFO_DIM",
    # Neural Networks
    "LSTMPredictor",
    "CNNPredictor",
    "MLPPredictor",
    # Tree Models
    "XGBoostWrapper",
    "RandomForestWrapper",
    "create_xgboost_model",
    "create_rf_model",
]
