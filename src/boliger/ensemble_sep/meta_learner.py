"""Meta Learner - Base 모델 출력을 결합하는 앙상블 전략.

지원 방식:
1. SimpleAverageMetaLearner: 단순 평균
2. WeightedAverageMetaLearner: 가중 평균 (가중치 조정 가능)
3. StackingMetaLearner: Ridge/XGBoost 스태킹
"""

import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Literal, Optional

import numpy as np
from omegaconf import DictConfig, OmegaConf

# Optional imports for stacking
try:
    from sklearn.linear_model import Ridge
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


ModelName = Literal["lstm", "cnn", "mlp", "xgboost", "rf"]
ALL_MODEL_NAMES = ["lstm", "cnn", "mlp", "xgboost", "rf"]


class MetaLearner(ABC):
    """Meta Learner 베이스 클래스."""

    @abstractmethod
    def predict(self, base_predictions: dict[str, np.ndarray]) -> np.ndarray:
        """Base 모델 예측들을 결합하여 최종 예측 생성.

        Args:
            base_predictions: {model_name: (batch,) predictions} 딕셔너리.

        Returns:
            (batch,) 최종 예측.
        """
        ...

    @abstractmethod
    def train(
        self,
        base_predictions: dict[str, np.ndarray],
        targets: np.ndarray,
    ) -> "MetaLearner":
        """검증셋으로 메타러너 학습 (stacking용).

        Args:
            base_predictions: {model_name: (n_samples,) predictions}.
            targets: (n_samples,) 실제값.

        Returns:
            self.
        """
        ...


class SimpleAverageMetaLearner(MetaLearner):
    """단순 평균 메타러너.

    모든 base 모델의 예측을 동일 가중치로 평균.
    """

    def __init__(self):
        pass

    def predict(self, base_predictions: dict[str, np.ndarray]) -> np.ndarray:
        """단순 평균 예측."""
        if not base_predictions:
            raise ValueError("No base predictions provided.")

        predictions = list(base_predictions.values())
        return np.mean(predictions, axis=0)

    def train(
        self,
        base_predictions: dict[str, np.ndarray],
        targets: np.ndarray,
    ) -> "SimpleAverageMetaLearner":
        """학습 불필요 (pass-through)."""
        return self


class WeightedAverageMetaLearner(MetaLearner):
    """가중 평균 메타러너.

    각 base 모델에 다른 가중치를 부여하여 평균.
    가중치는 수동 설정 또는 검증셋 기반 자동 계산.

    Args:
        weights: {model_name: weight} 딕셔너리. None이면 동일 가중치.
        normalize: True면 가중치 합이 1이 되도록 정규화.
    """

    def __init__(
        self,
        weights: Optional[dict[str, float]] = None,
        normalize: bool = True,
    ):
        self.weights = weights or {}
        self.normalize = normalize

    def set_weights(self, weights: dict[str, float]) -> None:
        """가중치 수동 설정."""
        self.weights = weights.copy()

    def get_weights(self) -> dict[str, float]:
        """현재 가중치 반환."""
        return self.weights.copy()

    def predict(self, base_predictions: dict[str, np.ndarray]) -> np.ndarray:
        """가중 평균 예측."""
        if not base_predictions:
            raise ValueError("No base predictions provided.")

        # 가중치 결정
        model_names = list(base_predictions.keys())
        weights = []
        for name in model_names:
            w = self.weights.get(name, 1.0)  # 기본 가중치 1.0
            weights.append(w)

        weights = np.array(weights)
        if self.normalize:
            weights = weights / weights.sum()

        # 가중 평균 계산
        predictions = np.stack([base_predictions[name] for name in model_names], axis=0)
        weighted_avg = np.sum(predictions * weights[:, np.newaxis], axis=0)

        return weighted_avg

    def train(
        self,
        base_predictions: dict[str, np.ndarray],
        targets: np.ndarray,
        method: str = "softmax_inv_loss",
        temperature: float = 1.0,
    ) -> "WeightedAverageMetaLearner":
        """검증셋 성능 기반 가중치 자동 계산.

        Args:
            base_predictions: {model_name: (n_samples,) predictions}.
            targets: (n_samples,) 실제값.
            method: 'softmax_inv_loss' 또는 'softmax_metric'.
            temperature: Softmax temperature (높을수록 균일).

        Returns:
            self.
        """
        model_names = list(base_predictions.keys())
        losses = []

        for name in model_names:
            pred = base_predictions[name]
            # MSE loss 계산
            loss = np.mean((pred - targets) ** 2)
            losses.append(loss)

        losses = np.array(losses)

        if method == "softmax_inv_loss":
            # Softmax(1/loss): 손실이 낮을수록 높은 가중치
            inv_losses = 1.0 / (losses + 1e-8)
            scores = inv_losses / temperature
        else:
            # softmax_metric: 상관계수 기반
            scores = []
            for name in model_names:
                pred = base_predictions[name]
                corr = np.corrcoef(pred, targets)[0, 1]
                scores.append(max(0, corr))  # 음수 상관계수는 0으로
            scores = np.array(scores) / temperature

        # Softmax 정규화
        exp_scores = np.exp(scores - np.max(scores))
        weights = exp_scores / exp_scores.sum()

        self.weights = {name: float(w) for name, w in zip(model_names, weights)}
        return self

    def save(self, path: Path) -> None:
        """가중치 저장."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(OmegaConf.create(self.weights), path)

    def load(self, path: Path) -> "WeightedAverageMetaLearner":
        """가중치 로드."""
        path = Path(path)
        self.weights = dict(OmegaConf.load(path))
        return self


class StackingMetaLearner(MetaLearner):
    """스태킹 메타러너.

    Base 모델 출력을 피처로 사용하여 2차 모델 학습.
    Ridge 또는 XGBoost 지원.

    Args:
        method: 'ridge' 또는 'xgb'.
        task_type: 'regression' 또는 'classification'.
        cfg: 스태킹 모델 config.
    """

    def __init__(
        self,
        method: Literal["ridge", "xgb"] = "ridge",
        task_type: Literal["regression", "classification"] = "regression",
        cfg: Optional[DictConfig] = None,
    ):
        self.method = method
        self.task_type = task_type
        self.cfg = cfg or OmegaConf.create({})
        self.model = None
        self._is_fitted = False
        self._model_order: list[str] = []

        self._init_model()

    def _init_model(self) -> None:
        """2차 모델 초기화."""
        if self.method == "ridge":
            if not SKLEARN_AVAILABLE:
                raise ImportError("sklearn is required for ridge stacking.")
            self.model = Ridge(
                alpha=self.cfg.get("alpha", 1.0),
                fit_intercept=self.cfg.get("fit_intercept", True),
            )
        elif self.method == "xgb":
            if not XGBOOST_AVAILABLE:
                raise ImportError("xgboost is required for xgb stacking.")
            if self.task_type == "classification":
                self.model = xgb.XGBClassifier(
                    n_estimators=self.cfg.get("n_estimators", 100),
                    max_depth=self.cfg.get("max_depth", 3),
                    learning_rate=self.cfg.get("learning_rate", 0.1),
                    reg_alpha=self.cfg.get("reg_alpha", 0.5),
                    reg_lambda=self.cfg.get("reg_lambda", 1.0),
                    eval_metric="logloss",
                    n_jobs=-1,
                    verbosity=0,
                )
            else:
                self.model = xgb.XGBRegressor(
                    n_estimators=self.cfg.get("n_estimators", 100),
                    max_depth=self.cfg.get("max_depth", 3),
                    learning_rate=self.cfg.get("learning_rate", 0.1),
                    reg_alpha=self.cfg.get("reg_alpha", 0.5),
                    reg_lambda=self.cfg.get("reg_lambda", 1.0),
                    n_jobs=-1,
                    verbosity=0,
                )
        else:
            raise ValueError(f"Unknown stacking method: {self.method}")

    def _build_meta_features(
        self,
        base_predictions: dict[str, np.ndarray],
    ) -> np.ndarray:
        """Base 예측을 meta 피처로 변환."""
        if not self._model_order:
            self._model_order = sorted(base_predictions.keys())

        features = [base_predictions[name] for name in self._model_order]
        return np.stack(features, axis=1)  # (n_samples, n_models)

    def train(
        self,
        base_predictions: dict[str, np.ndarray],
        targets: np.ndarray,
    ) -> "StackingMetaLearner":
        """검증셋으로 스태킹 모델 학습.

        Args:
            base_predictions: {model_name: (n_samples,) predictions}.
            targets: (n_samples,) 실제값.

        Returns:
            self.
        """
        self._model_order = sorted(base_predictions.keys())
        X_meta = self._build_meta_features(base_predictions)

        y = targets.astype(int) if self.task_type == "classification" else targets
        self.model.fit(X_meta, y)
        self._is_fitted = True

        # 학습 결과 출력
        if self.method == "ridge" and hasattr(self.model, "coef_"):
            print(f"Ridge coefficients: {dict(zip(self._model_order, self.model.coef_))}")
        elif self.method == "xgb" and hasattr(self.model, "feature_importances_"):
            print(f"XGB importances: {dict(zip(self._model_order, self.model.feature_importances_))}")

        return self

    def predict(self, base_predictions: dict[str, np.ndarray]) -> np.ndarray:
        """스태킹 예측."""
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call train() first.")

        X_meta = self._build_meta_features(base_predictions)
        if self.task_type == "classification" and self.method == "xgb":
            return self.model.predict_proba(X_meta)[:, 1]
        return self.model.predict(X_meta)

    def get_coefficients(self) -> dict[str, float]:
        """모델 계수/중요도 반환."""
        if not self._is_fitted:
            raise RuntimeError("Model not fitted.")

        if self.method == "ridge":
            return dict(zip(self._model_order, self.model.coef_))
        elif self.method == "xgb":
            return dict(zip(self._model_order, self.model.feature_importances_))
        return {}

    def save(self, path: Path) -> None:
        """모델 저장."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "method": self.method,
                "task_type": self.task_type,
                "model": self.model,
                "model_order": self._model_order,
                "is_fitted": self._is_fitted,
            }, f)

    def load(self, path: Path) -> "StackingMetaLearner":
        """모델 로드."""
        path = Path(path)
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.method = data["method"]
        self.task_type = data.get("task_type", "regression")
        self.model = data["model"]
        self._model_order = data["model_order"]
        self._is_fitted = data["is_fitted"]
        return self


def create_meta_learner(
    mode: str = "simple_avg",
    cfg: Optional[DictConfig] = None,
    task_type: str = "regression",
) -> MetaLearner:
    """Meta Learner 팩토리 함수.

    Args:
        mode: 'simple_avg', 'weighted_avg', 'learned_avg',
              'stacking_ridge', 'stacking_xgb'.
        cfg: 설정 (weights, stacking params 등).

    Returns:
        MetaLearner 인스턴스.
    """
    cfg = cfg or OmegaConf.create({})

    if mode == "simple_avg":
        return SimpleAverageMetaLearner()
    elif mode == "weighted_avg":
        weights = dict(cfg.get("weights", {}))
        return WeightedAverageMetaLearner(weights=weights)
    elif mode == "learned_avg":
        return WeightedAverageMetaLearner()  # train()에서 가중치 계산
    elif mode == "stacking_ridge":
        stacking_cfg = cfg.get("stacking", {}).get("ridge", {})
        return StackingMetaLearner(
            method="ridge", task_type=task_type, cfg=OmegaConf.create(stacking_cfg)
        )
    elif mode == "stacking_xgb":
        stacking_cfg = cfg.get("stacking", {}).get("xgb", {})
        return StackingMetaLearner(
            method="xgb", task_type=task_type, cfg=OmegaConf.create(stacking_cfg)
        )
    else:
        raise ValueError(f"Unknown ensemble mode: {mode}")
