"""분리 학습 앙상블 모듈 (v3).

5개 Base 모델 (LSTM, CNN, MLP, XGBoost, RF) 출력을 결합하는 앙상블 시스템.

앙상블 방식:
- simple_avg: 단순 평균
- weighted_avg: 가중 평균 (Task별 모델 가중치 조정 가능)
- learned_avg: 검증셋 성능 기반 가중치 자동 계산
- stacking_ridge: Ridge 스태킹 메타러너
- stacking_xgb: XGBoost 스태킹 메타러너
"""

from boliger.ensemble_sep.base_ensemble import BaseEnsemble
from boliger.ensemble_sep.meta_learner import (
    MetaLearner,
    SimpleAverageMetaLearner,
    WeightedAverageMetaLearner,
    StackingMetaLearner,
)
from boliger.ensemble_sep.separated_predictor import SeparatedPredictor

__all__ = [
    # Core classes
    "BaseEnsemble",
    "SeparatedPredictor",
    # Meta learners
    "MetaLearner",
    "SimpleAverageMetaLearner",
    "WeightedAverageMetaLearner",
    "StackingMetaLearner",
]
