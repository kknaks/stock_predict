"""
Model 2: Stacking Regressor (상승 시 종가 수익률 예측)

Base Learners: RandomForest, XGBoost, LightGBM
Meta Learner: Ridge
"""

import time
from typing import Optional, Dict, Any
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .base import BaseStackingModel


class StackingRegressorUp(BaseStackingModel):
    """
    Stacking Regressor for 상승 케이스 종가 수익률 예측
    
    Model 2: 갭 상승 후 실제 상승한 케이스의 종가 기준 수익률 예측
    """
    
    def __init__(
        self,
        ridge_alpha: float = 1.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.ridge_alpha = ridge_alpha
    
    def _create_base_learners(self) -> list:
        """Base learners 생성"""
        base_learners = [
            ('rf', RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth_rf,
                min_samples_split=10,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                verbose=0
            )),
        ]
        
        # XGBoost
        if self.has_xgb:
            import xgboost as xgb
            
            xgb_params = dict(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth_xgb,
                learning_rate=self.learning_rate,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                verbosity=0,
            )
            if self.use_gpu:
                xgb_params["device"] = "cuda"
                xgb_params["tree_method"] = "hist"
            base_learners.append(('xgb', xgb.XGBRegressor(**xgb_params)))

        # LightGBM
        if self.has_lgb:
            import lightgbm as lgb

            lgb_params = dict(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth_lgb,
                learning_rate=self.learning_rate,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                verbosity=-1,
            )
            if self.use_gpu:
                lgb_params["device"] = "gpu"
            base_learners.append(('lgb', lgb.LGBMRegressor(**lgb_params)))
        
        return base_learners
    
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        verbose: bool = True
    ) -> 'StackingRegressorUp':
        """
        모델 학습
        
        Args:
            X: Feature DataFrame (상승 케이스만)
            y: Target Series (target_return)
            verbose: 상세 출력 여부
            
        Returns:
            self
        """
        if verbose:
            print("=" * 80)
            print("Model 2: Stacking Regressor (상승 케이스 - 종가) 학습")
            print("=" * 80)
        
        # Feature 저장
        self.features = list(X.columns)
        
        # Base learners 생성
        base_learners = self._create_base_learners()
        
        if verbose:
            print(f"\nBase Learners ({len(base_learners)}개):")
            for name, _ in base_learners:
                print(f"  - {name.upper()}")
        
        # Stacking Regressor 생성
        self.model = StackingRegressor(
            estimators=base_learners,
            final_estimator=Ridge(alpha=self.ridge_alpha, random_state=self.random_state),
            cv=self.cv_folds,
            n_jobs=self.n_jobs,
            verbose=self.verbose
        )
        
        # 학습
        if verbose:
            print("\n학습 시작...")
        
        start = time.time()
        self.model.fit(X, y)
        elapsed = time.time() - start
        
        self.is_fitted = True
        
        if verbose:
            print(f"\n✓ 학습 완료 (소요 시간: {elapsed:.1f}초)")
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        수익률 예측
        
        Args:
            X: Feature DataFrame
            
        Returns:
            예측 수익률 (%)
        """
        if not self.is_fitted:
            raise ValueError("모델이 학습되지 않았습니다. fit()을 먼저 실행하세요.")
        
        return self.model.predict(X)
    
    def evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        dataset_name: str = "Test"
    ) -> Dict[str, float]:
        """
        모델 평가
        
        Args:
            X: Feature DataFrame
            y: 실제 값
            dataset_name: 데이터셋 이름
            
        Returns:
            평가 지표 딕셔너리
        """
        y_pred = self.predict(X)
        
        metrics = {
            'mae': mean_absolute_error(y, y_pred),
            'rmse': np.sqrt(mean_squared_error(y, y_pred)),
            'r2': r2_score(y, y_pred),
        }
        
        print(f"\n{dataset_name} Set 성능:")
        print(f"  MAE:  {metrics['mae']:.4f}%")
        print(f"  RMSE: {metrics['rmse']:.4f}%")
        print(f"  R²:   {metrics['r2']:.4f}")
        
        return metrics
    
    def get_params(self) -> Dict[str, Any]:
        """모델 파라미터 반환"""
        params = super().get_params()
        params.update({
            'ridge_alpha': self.ridge_alpha,
            'model_type': 'regressor_up',
        })
        return params
