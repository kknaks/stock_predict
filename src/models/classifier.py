"""
Model 1: Stacking Classifier (상승 확률 예측)

Base Learners: RandomForest, XGBoost, LightGBM
Meta Learner: LogisticRegression
"""

import time
from typing import Optional, Dict, Any, Tuple
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)

from .base import (
    BaseStackingModel,
    get_available_features,
    DEFAULT_THRESHOLD,
)


class StackingClassifierModel(BaseStackingModel):
    """
    Stacking Classifier for 방향 예측 (상승/하락)
    
    Model 1: 갭 상승 후 당일 종가가 시가보다 높을 확률 예측
    """
    
    def __init__(
        self,
        threshold: float = DEFAULT_THRESHOLD,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.threshold = threshold
        self.scale_pos_weight = None  # XGBoost용 클래스 불균형 가중치
    
    def _create_base_learners(self, y_train: Optional[pd.Series] = None) -> list:
        """Base learners 생성"""
        base_learners = [
            ('rf', RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth_rf,
                min_samples_split=10,
                class_weight='balanced',
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                verbose=0
            )),
        ]
        
        # XGBoost
        if self.has_xgb:
            import xgboost as xgb
            
            # 클래스 불균형 가중치 계산
            if y_train is not None:
                n_neg = (y_train == 0).sum()
                n_pos = (y_train == 1).sum()
                self.scale_pos_weight = n_neg / n_pos
            else:
                self.scale_pos_weight = 1.0
            
            xgb_params = dict(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth_xgb,
                learning_rate=self.learning_rate,
                scale_pos_weight=self.scale_pos_weight,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                verbosity=0,
            )
            if self.use_gpu:
                xgb_params["device"] = "cuda"
                xgb_params["tree_method"] = "hist"
            base_learners.append(('xgb', xgb.XGBClassifier(**xgb_params)))
        
        # LightGBM
        if self.has_lgb:
            import lightgbm as lgb
            
            lgb_params = dict(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth_lgb,
                learning_rate=self.learning_rate,
                class_weight='balanced',
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                verbosity=-1,
            )
            if self.use_gpu:
                lgb_params["device"] = "cuda"
            base_learners.append(('lgb', lgb.LGBMClassifier(**lgb_params)))
        
        return base_learners
    
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        verbose: bool = True
    ) -> 'StackingClassifierModel':
        """
        모델 학습
        
        Args:
            X: Feature DataFrame
            y: Target Series (0 or 1)
            verbose: 상세 출력 여부
            
        Returns:
            self
        """
        if verbose:
            print("=" * 80)
            print("Model 1: Stacking Classifier 학습")
            print("=" * 80)
        
        # Feature 저장
        self.features = list(X.columns)
        
        # Base learners 생성
        base_learners = self._create_base_learners(y)
        
        if verbose:
            print(f"\nBase Learners ({len(base_learners)}개):")
            for name, _ in base_learners:
                print(f"  - {name.upper()}")
        
        # Stacking Classifier 생성
        self.model = StackingClassifier(
            estimators=base_learners,
            final_estimator=LogisticRegression(max_iter=1000, random_state=self.random_state),
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
        예측 (threshold 적용)
        
        Args:
            X: Feature DataFrame
            
        Returns:
            예측 결과 (0 or 1)
        """
        if not self.is_fitted:
            raise ValueError("모델이 학습되지 않았습니다. fit()을 먼저 실행하세요.")
        
        proba = self.predict_proba(X)[:, 1]
        return (proba >= self.threshold).astype(int)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        확률 예측
        
        Args:
            X: Feature DataFrame
            
        Returns:
            확률 배열 [P(하락), P(상승)]
        """
        if not self.is_fitted:
            raise ValueError("모델이 학습되지 않았습니다. fit()을 먼저 실행하세요.")
        
        return self.model.predict_proba(X)
    
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
        y_proba = self.predict_proba(X)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, pos_label=1, zero_division=0),
            'recall': recall_score(y, y_pred, pos_label=1, zero_division=0),
            'f1_score': f1_score(y, y_pred, pos_label=1, zero_division=0),
            'roc_auc': roc_auc_score(y, y_proba),
        }
        
        print(f"\n{dataset_name} Set 성능:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1-Score:  {metrics['f1_score']:.4f}")
        print(f"  ROC AUC:   {metrics['roc_auc']:.4f}")
        
        return metrics
    
    def get_params(self) -> Dict[str, Any]:
        """모델 파라미터 반환"""
        params = super().get_params()
        params.update({
            'threshold': self.threshold,
            'scale_pos_weight': self.scale_pos_weight,
            'model_type': 'classifier',
        })
        return params
