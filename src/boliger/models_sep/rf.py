"""RandomForest Wrapper - 신경망과 유사한 인터페이스 제공.

RandomForest를 분리 학습 파이프라인에 통합하기 위한 래퍼 클래스.
sklearn 호환 API (fit, predict, predict_proba) 제공.
"""

import pickle
from pathlib import Path
from typing import Literal, Optional

import numpy as np
from omegaconf import DictConfig
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

ModeType = Literal["cls", "high", "low"]


class RandomForestWrapper(BaseEstimator):
    """RandomForest 모델 래퍼.

    분리 학습 파이프라인과 호환되는 인터페이스 제공:
    - cls: RandomForestClassifier (prob 출력)
    - high/low: RandomForestRegressor (return 예측)

    Args:
        cfg: RF config (n_estimators, max_depth, etc.).
        mode: 'cls', 'high', 'low' 중 하나.
    """

    def __init__(self, cfg: DictConfig, mode: ModeType = "cls"):
        self.cfg = cfg
        self.mode = mode
        self.model = None
        self._is_fitted = False

        # 공통 파라미터
        common_params = {
            "n_estimators": cfg.get("n_estimators", 300),
            "max_depth": cfg.get("max_depth", 12),
            "min_samples_split": cfg.get("min_samples_split", 10),
            "min_samples_leaf": cfg.get("min_samples_leaf", 5),
            "max_features": cfg.get("max_features", "sqrt"),
            "n_jobs": cfg.get("n_jobs", -1),
            "random_state": cfg.get("random_state", 42),
            "verbose": cfg.get("verbose", 0),
        }

        if mode == "cls":
            self.model = RandomForestClassifier(**common_params)
        else:
            self.model = RandomForestRegressor(**common_params)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        verbose: bool = True,
    ) -> "RandomForestWrapper":
        """모델 학습.

        Note: RF는 early stopping이 없어 X_val, y_val은 무시됨.

        Args:
            X: (n_samples, n_features) 피처.
            y: (n_samples,) 타겟.
            X_val: (무시됨) 검증 피처.
            y_val: (무시됨) 검증 타겟.
            verbose: 학습 로그 출력.

        Returns:
            self.
        """
        if verbose:
            print(f"[RF/{self.mode}] Training with {X.shape[0]:,} samples, {X.shape[1]} features...")

        self.model.fit(X, y)
        self._is_fitted = True

        if verbose:
            print(f"[RF/{self.mode}] Training complete.")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """예측.

        Args:
            X: (n_samples, n_features) 피처.

        Returns:
            cls: (n_samples,) 클래스 예측 (0 or 1).
            high/low: (n_samples,) 회귀 예측.
        """
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """확률 예측 (cls 모드 전용).

        Args:
            X: (n_samples, n_features) 피처.

        Returns:
            (n_samples,) positive class 확률.
        """
        if self.mode != "cls":
            raise ValueError("predict_proba() only available in 'cls' mode.")
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        proba = self.model.predict_proba(X)
        return proba[:, 1]  # positive class 확률

    def save(self, path: Path) -> None:
        """모델 저장.

        Args:
            path: 저장 경로.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "model": self.model,
                "mode": self.mode,
                "is_fitted": self._is_fitted,
            }, f)

    def load(self, path: Path) -> "RandomForestWrapper":
        """모델 로드.

        Args:
            path: 로드 경로.

        Returns:
            self.
        """
        path = Path(path)
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.model = data["model"]
        self.mode = data["mode"]
        self._is_fitted = data["is_fitted"]
        return self

    @property
    def feature_importances_(self) -> np.ndarray:
        """피처 중요도 반환."""
        if not self._is_fitted:
            raise ValueError("Model not fitted.")
        return self.model.feature_importances_


def create_rf_model(cfg: DictConfig, mode: ModeType = "cls") -> RandomForestWrapper:
    """RandomForest 모델 생성 팩토리 함수.

    Args:
        cfg: RF config.
        mode: 'cls', 'high', 'low'.

    Returns:
        RandomForestWrapper 인스턴스.
    """
    return RandomForestWrapper(cfg, mode)
