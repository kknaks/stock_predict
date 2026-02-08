"""XGBoost Wrapper - 신경망과 유사한 인터페이스 제공.

XGBoost를 분리 학습 파이프라인에 통합하기 위한 래퍼 클래스.
sklearn 호환 API (fit, predict, predict_proba) 제공.
"""

import pickle
from pathlib import Path
from typing import Literal, Optional

import numpy as np
from omegaconf import DictConfig
from sklearn.base import BaseEstimator

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

ModeType = Literal["cls", "high", "low"]


class XGBoostWrapper(BaseEstimator):
    """XGBoost 모델 래퍼.

    분리 학습 파이프라인과 호환되는 인터페이스 제공:
    - cls: XGBClassifier (prob 출력)
    - high/low: XGBRegressor (return 예측)

    Args:
        cfg: XGBoost config (n_estimators, max_depth, etc.).
        mode: 'cls', 'high', 'low' 중 하나.
    """

    def __init__(self, cfg: DictConfig, mode: ModeType = "cls"):
        if not XGBOOST_AVAILABLE:
            raise ImportError("xgboost is not installed. Run: pip install xgboost")

        self.cfg = cfg
        self.mode = mode
        self.model = None
        self._is_fitted = False

        # 공통 파라미터
        common_params = {
            "n_estimators": cfg.get("n_estimators", 500),
            "max_depth": cfg.get("max_depth", 6),
            "learning_rate": cfg.get("learning_rate", 0.05),
            "subsample": cfg.get("subsample", 0.8),
            "colsample_bytree": cfg.get("colsample_bytree", 0.8),
            "min_child_weight": cfg.get("min_child_weight", 3),
            "reg_alpha": cfg.get("reg_alpha", 0.1),
            "reg_lambda": cfg.get("reg_lambda", 1.0),
            "n_jobs": cfg.get("n_jobs", -1),
            "random_state": cfg.get("random_state", 42),
            "verbosity": cfg.get("verbosity", 0),
        }

        if mode == "cls":
            self.model = xgb.XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                use_label_encoder=False,
                early_stopping_rounds=cfg.get("early_stopping_rounds", 50),
                **common_params,
            )
        else:
            self.model = xgb.XGBRegressor(
                objective="reg:squarederror",
                eval_metric="rmse",
                early_stopping_rounds=cfg.get("early_stopping_rounds", 50),
                **common_params,
            )

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        verbose: bool = True,
    ) -> "XGBoostWrapper":
        """모델 학습.

        Args:
            X: (n_samples, n_features) 피처.
            y: (n_samples,) 타겟.
            X_val: 검증 피처 (early stopping용).
            y_val: 검증 타겟.
            verbose: 학습 로그 출력.

        Returns:
            self.
        """
        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val)]
            self.model.fit(
                X, y,
                eval_set=eval_set,
                verbose=verbose,
            )
        else:
            # early stopping 없이 학습
            self.model.set_params(early_stopping_rounds=None)
            self.model.fit(X, y, verbose=verbose)

        self._is_fitted = True
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

    def load(self, path: Path) -> "XGBoostWrapper":
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

    def get_best_iteration(self) -> int:
        """Early stopping된 best iteration 반환."""
        if hasattr(self.model, "best_iteration"):
            return self.model.best_iteration
        return self.cfg.get("n_estimators", 500)


def create_xgboost_model(cfg: DictConfig, mode: ModeType = "cls") -> XGBoostWrapper:
    """XGBoost 모델 생성 팩토리 함수.

    Args:
        cfg: XGBoost config.
        mode: 'cls', 'high', 'low'.

    Returns:
        XGBoostWrapper 인스턴스.
    """
    return XGBoostWrapper(cfg, mode)
