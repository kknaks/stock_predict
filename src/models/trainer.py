"""
모델 학습 파이프라인

노트북 06_modeling_stacking.ipynb의 자동화 버전
"""

import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, Tuple

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, mean_absolute_error, mean_squared_error, r2_score
)

from .base import (
    FEATURE_COLS,
    TARGET_DIRECTION,
    TARGET_RETURN,
    TARGET_MAX_RETURN,
    CORE_FEATURES,
    DEFAULT_RANDOM_STATE,
    get_available_features,
    prepare_data_for_training,
)
from .classifier import StackingClassifierModel
from .regressor_up import StackingRegressorUp
from .regressor_high import StackingRegressorHigh
from .regressor_down import StackingRegressorDown
from .storage import ModelStorage


class StackingModelTrainer:
    """
    Stacking 앙상블 모델 전체 학습 파이프라인
    
    노트북 06_modeling_stacking.ipynb를 자동화
    """
    
    def __init__(
        self,
        data_path: str,
        model_dir: str = "models",
        random_state: int = DEFAULT_RANDOM_STATE,
        test_size: float = 0.1,
        valid_size: float = 0.1,
        threshold: float = 0.4,
        n_estimators: int = 300,
        verbose: bool = True
    ):
        """
        Args:
            data_path: 전처리된 데이터 경로 (.parquet)
            model_dir: 모델 저장 디렉토리
            random_state: 랜덤 시드
            test_size: 테스트 비율
            valid_size: 검증 비율
            threshold: 분류 임계값
            n_estimators: 앙상블 모델 수
            verbose: 상세 출력 여부
        """
        self.data_path = Path(data_path)
        self.model_dir = Path(model_dir)
        self.random_state = random_state
        self.test_size = test_size
        self.valid_size = valid_size
        self.threshold = threshold
        self.n_estimators = n_estimators
        self.verbose = verbose
        
        # 모델 인스턴스
        self.classifier = None
        self.regressor_up = None
        self.regressor_high = None
        self.regressor_down = None
        
        # 데이터
        self.df = None
        self.df_model = None
        self.features = None
        
        # 메트릭
        self.test_metrics = {}
    
    def load_data(self) -> pd.DataFrame:
        """데이터 로드"""
        if self.verbose:
            print("=" * 80)
            print("데이터 로드")
            print("=" * 80)
        
        if not self.data_path.exists():
            raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다: {self.data_path}")
        
        self.df = pd.read_parquet(self.data_path)
        
        if self.verbose:
            print(f"\n✓ 데이터 로드 완료")
            print(f"  - Shape: {self.df.shape}")
            print(f"  - 날짜 범위: {self.df['date'].min()} ~ {self.df['date'].max()}")
            print(f"  - 종목 수: {self.df['InfoCode'].nunique():,}")
        
        return self.df
    
    def prepare_data(self) -> pd.DataFrame:
        """학습용 데이터 전처리"""
        if self.df is None:
            self.load_data()
        
        if self.verbose:
            print("\n" + "=" * 80)
            print("데이터 전처리")
            print("=" * 80)
        
        self.df_model = prepare_data_for_training(self.df, include_max_return=True)
        self.features = get_available_features(self.df_model)
        
        if self.verbose:
            print(f"\n✓ 전처리 완료")
            print(f"  - Shape: {self.df_model.shape}")
            print(f"  - Features: {len(self.features)}개")
        
        return self.df_model
    
    def split_data(self) -> Tuple[Dict, Dict, Dict]:
        """데이터 분할"""
        if self.df_model is None:
            self.prepare_data()
        
        if self.verbose:
            print("\n" + "=" * 80)
            print("데이터 분할")
            print("=" * 80)
        
        # 전체 데이터
        X_all = self.df_model[self.features]
        y_direction = self.df_model[TARGET_DIRECTION]
        y_return = self.df_model[TARGET_RETURN]
        
        # Train+Valid / Test 분할
        X_temp, X_test, y_dir_temp, y_dir_test, y_ret_temp, y_ret_test = train_test_split(
            X_all, y_direction, y_return,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y_direction
        )
        
        # Train / Valid 분할
        valid_ratio = self.valid_size / (1 - self.test_size)
        X_train, X_valid, y_dir_train, y_dir_valid, y_ret_train, y_ret_valid = train_test_split(
            X_temp, y_dir_temp, y_ret_temp,
            test_size=valid_ratio,
            random_state=self.random_state,
            stratify=y_dir_temp
        )
        
        if self.verbose:
            print(f"\n전체 데이터 분할:")
            print(f"  Train: {len(X_train):,} ({len(X_train)/len(X_all)*100:.1f}%)")
            print(f"  Valid: {len(X_valid):,} ({len(X_valid)/len(X_all)*100:.1f}%)")
            print(f"  Test:  {len(X_test):,} ({len(X_test)/len(X_all)*100:.1f}%)")
        
        # 상승 케이스
        df_up = self.df_model[self.df_model[TARGET_DIRECTION] == 1].copy()
        X_up = df_up[self.features]
        y_up = df_up[TARGET_RETURN]
        y_up_max = df_up[TARGET_MAX_RETURN] if TARGET_MAX_RETURN in df_up.columns else None
        
        X_up_temp, X_up_test, y_up_temp, y_up_test = train_test_split(
            X_up, y_up, test_size=self.test_size, random_state=self.random_state
        )
        X_up_train, X_up_valid, y_up_train, y_up_valid = train_test_split(
            X_up_temp, y_up_temp, test_size=valid_ratio, random_state=self.random_state
        )
        
        # 고가 데이터 분할
        y_up_max_train, y_up_max_valid, y_up_max_test = None, None, None
        if y_up_max is not None:
            y_up_max_train = df_up.loc[X_up_train.index, TARGET_MAX_RETURN]
            y_up_max_valid = df_up.loc[X_up_valid.index, TARGET_MAX_RETURN]
            y_up_max_test = df_up.loc[X_up_test.index, TARGET_MAX_RETURN]
        
        if self.verbose:
            print(f"\n상승 케이스 분할:")
            print(f"  Train: {len(X_up_train):,}")
            print(f"  Valid: {len(X_up_valid):,}")
            print(f"  Test:  {len(X_up_test):,}")
        
        # 하락 케이스
        df_down = self.df_model[self.df_model[TARGET_DIRECTION] == 0].copy()
        X_down = df_down[self.features]
        y_down = df_down[TARGET_RETURN]
        
        X_down_temp, X_down_test, y_down_temp, y_down_test = train_test_split(
            X_down, y_down, test_size=self.test_size, random_state=self.random_state
        )
        X_down_train, X_down_valid, y_down_train, y_down_valid = train_test_split(
            X_down_temp, y_down_temp, test_size=valid_ratio, random_state=self.random_state
        )
        
        if self.verbose:
            print(f"\n하락 케이스 분할:")
            print(f"  Train: {len(X_down_train):,}")
            print(f"  Valid: {len(X_down_valid):,}")
            print(f"  Test:  {len(X_down_test):,}")
        
        # 데이터 반환
        all_data = {
            'X_train': X_train, 'X_valid': X_valid, 'X_test': X_test,
            'y_dir_train': y_dir_train, 'y_dir_valid': y_dir_valid, 'y_dir_test': y_dir_test,
            'y_ret_train': y_ret_train, 'y_ret_valid': y_ret_valid, 'y_ret_test': y_ret_test,
        }
        
        up_data = {
            'X_train': X_up_train, 'X_valid': X_up_valid, 'X_test': X_up_test,
            'y_train': y_up_train, 'y_valid': y_up_valid, 'y_test': y_up_test,
            'y_max_train': y_up_max_train, 'y_max_valid': y_up_max_valid, 'y_max_test': y_up_max_test,
        }
        
        down_data = {
            'X_train': X_down_train, 'X_valid': X_down_valid, 'X_test': X_down_test,
            'y_train': y_down_train, 'y_valid': y_down_valid, 'y_test': y_down_test,
        }
        
        return all_data, up_data, down_data
    
    def train_classifier(self, all_data: Dict) -> StackingClassifierModel:
        """Model 1: Stacking Classifier 학습"""
        self.classifier = StackingClassifierModel(
            threshold=self.threshold,
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            verbose=1 if self.verbose else 0
        )
        
        self.classifier.fit(
            all_data['X_train'],
            all_data['y_dir_train'],
            verbose=self.verbose
        )
        
        # 평가
        metrics = self.classifier.evaluate(
            all_data['X_test'],
            all_data['y_dir_test'],
            dataset_name="Test"
        )
        self.test_metrics['classifier'] = metrics
        
        return self.classifier
    
    def train_regressor_up(self, up_data: Dict) -> StackingRegressorUp:
        """Model 2: 상승 케이스 종가 회귀 학습"""
        self.regressor_up = StackingRegressorUp(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            verbose=1 if self.verbose else 0
        )
        
        self.regressor_up.fit(
            up_data['X_train'],
            up_data['y_train'],
            verbose=self.verbose
        )
        
        # 평가
        metrics = self.regressor_up.evaluate(
            up_data['X_test'],
            up_data['y_test'],
            dataset_name="Test"
        )
        self.test_metrics['regressor_up'] = metrics
        
        return self.regressor_up
    
    def train_regressor_high(self, up_data: Dict) -> Optional[StackingRegressorHigh]:
        """Model 2-1: 상승 케이스 고가 회귀 학습"""
        if up_data.get('y_max_train') is None:
            if self.verbose:
                print("\n⚠ target_max_return 없음, Model 2-1 건너뜀")
            return None
        
        self.regressor_high = StackingRegressorHigh(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            verbose=1 if self.verbose else 0
        )
        
        self.regressor_high.fit(
            up_data['X_train'],
            up_data['y_max_train'],
            verbose=self.verbose
        )
        
        # 평가
        metrics = self.regressor_high.evaluate(
            up_data['X_test'],
            up_data['y_max_test'],
            dataset_name="Test"
        )
        self.test_metrics['regressor_high'] = metrics
        
        return self.regressor_high
    
    def train_regressor_down(self, down_data: Dict) -> StackingRegressorDown:
        """Model 3: 하락 케이스 회귀 학습"""
        self.regressor_down = StackingRegressorDown(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            verbose=1 if self.verbose else 0
        )
        
        self.regressor_down.fit(
            down_data['X_train'],
            down_data['y_train'],
            verbose=self.verbose
        )
        
        # 평가
        metrics = self.regressor_down.evaluate(
            down_data['X_test'],
            down_data['y_test'],
            dataset_name="Test"
        )
        self.test_metrics['regressor_down'] = metrics
        
        return self.regressor_down
    
    def train_all(self) -> Dict[str, Any]:
        """모든 모델 학습"""
        if self.verbose:
            print("\n" + "=" * 80)
            print("Stacking 앙상블 모델 전체 학습 시작")
            print("=" * 80)
        
        # 데이터 준비
        all_data, up_data, down_data = self.split_data()
        
        # 모델 학습
        self.train_classifier(all_data)
        self.train_regressor_up(up_data)
        self.train_regressor_high(up_data)
        self.train_regressor_down(down_data)
        
        if self.verbose:
            self.print_summary()
        
        return {
            'classifier': self.classifier,
            'regressor_up': self.regressor_up,
            'regressor_high': self.regressor_high,
            'regressor_down': self.regressor_down,
            'features': self.features,
            'test_metrics': self.test_metrics,
        }
    
    def save_models(self, subdir: str = "stacking") -> Path:
        """모든 모델 저장"""
        storage = ModelStorage(base_dir=str(self.model_dir))
        
        return storage.save_all_models(
            classifier=self.classifier,
            regressor_up=self.regressor_up,
            regressor_high=self.regressor_high,
            regressor_down=self.regressor_down,
            test_metrics=self.test_metrics,
            subdir=subdir
        )
    
    def print_summary(self):
        """학습 결과 요약 출력"""
        print("\n" + "=" * 80)
        print("학습 완료 요약")
        print("=" * 80)
        
        print("\n1. Model 1: Stacking Classifier")
        if 'classifier' in self.test_metrics:
            m = self.test_metrics['classifier']
            print(f"   - Accuracy: {m['accuracy']:.4f}")
            print(f"   - ROC AUC:  {m['roc_auc']:.4f}")
            print(f"   - F1-Score: {m['f1_score']:.4f}")
        
        print("\n2. Model 2: Stacking Regressor (상승 - 종가)")
        if 'regressor_up' in self.test_metrics:
            m = self.test_metrics['regressor_up']
            print(f"   - MAE:  {m['mae']:.4f}%")
            print(f"   - RMSE: {m['rmse']:.4f}%")
            print(f"   - R²:   {m['r2']:.4f}")
        
        if 'regressor_high' in self.test_metrics:
            print("\n3. Model 2-1: Stacking Regressor (상승 - 고가)")
            m = self.test_metrics['regressor_high']
            print(f"   - MAE:  {m['mae']:.4f}%")
            print(f"   - RMSE: {m['rmse']:.4f}%")
            print(f"   - R²:   {m['r2']:.4f}")
        
        print("\n4. Model 3: Stacking Regressor (하락)")
        if 'regressor_down' in self.test_metrics:
            m = self.test_metrics['regressor_down']
            print(f"   - MAE:  {m['mae']:.4f}%")
            print(f"   - RMSE: {m['rmse']:.4f}%")
            print(f"   - R²:   {m['r2']:.4f}")


def train_models(
    data_path: str,
    model_dir: str = "models",
    save: bool = True,
    **kwargs
) -> Dict[str, Any]:
    """
    전체 모델 학습 실행 함수
    
    Args:
        data_path: 전처리된 데이터 경로
        model_dir: 모델 저장 디렉토리
        save: 모델 저장 여부
        **kwargs: 추가 파라미터
        
    Returns:
        학습된 모델 및 메트릭
    """
    trainer = StackingModelTrainer(
        data_path=data_path,
        model_dir=model_dir,
        **kwargs
    )
    
    result = trainer.train_all()
    
    if save:
        trainer.save_models()
    
    return result


if __name__ == "__main__":
    # 기본 실행
    result = train_models(
        data_path="data/processed/preprocessed_df_full.parquet",
        model_dir="models",
        save=True
    )
