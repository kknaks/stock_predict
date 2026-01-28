"""
통합 예측기

저장된 pkl 모델을 로드하여 예측 수행
기대 수익률 계산: P(상승) × E[상승시] + P(하락) × E[하락시]
"""

from pathlib import Path
from typing import Optional, Dict, Any, Union, List
import joblib
import pandas as pd
import numpy as np


class HybridPredictor:
    """
    하이브리드 통합 예측기
    
    3개(또는 4개) 모델을 결합하여 기대 수익률 계산:
    - Model 1: 방향 예측 (상승 확률)
    - Model 2: 상승 시 종가 수익률 예측
    - Model 2-1: 상승 시 고가 수익률 예측 (익절용, 선택)
    - Model 3: 하락 시 손실률 예측
    """
    
    def __init__(
        self,
        classifier,
        regressor_up,
        regressor_down,
        regressor_high=None,
        threshold: float = 0.4,
        features: Optional[List[str]] = None
    ):
        """
        Args:
            classifier: Model 1 (분류 모델)
            regressor_up: Model 2 (상승 종가 회귀)
            regressor_down: Model 3 (하락 회귀)
            regressor_high: Model 2-1 (상승 고가 회귀, 선택)
            threshold: 상승 예측 임계값
            features: 사용할 Feature 컬럼 리스트
        """
        self.classifier = classifier
        self.regressor_up = regressor_up
        self.regressor_down = regressor_down
        self.regressor_high = regressor_high
        self.threshold = threshold
        self.features = features
        
        # 모델에서 feature 추출 (없으면)
        if self.features is None:
            if hasattr(classifier, 'features') and classifier.features:
                self.features = classifier.features
            elif hasattr(regressor_up, 'features') and regressor_up.features:
                self.features = regressor_up.features
    
    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        통합 예측 수행
        
        Args:
            X: Feature DataFrame
            
        Returns:
            예측 결과 DataFrame:
            - prob_up: 상승 확률
            - prob_down: 하락 확률
            - return_if_up: 상승 시 예상 수익률 (종가)
            - return_if_down: 하락 시 예상 손실률
            - expected_return: 기대 수익률
            - predicted_direction: 예측 방향 (0/1)
            - max_return_if_up: 상승 시 예상 최대 수익률 (고가, 선택)
            - take_profit_target: 익절 목표가 (선택)
        """
        # Feature 선택
        if self.features:
            available_features = [f for f in self.features if f in X.columns]
            X_input = X[available_features]
        else:
            X_input = X
        
        # 1. 상승 확률 예측
        if hasattr(self.classifier, 'predict_proba'):
            prob_up = self.classifier.predict_proba(X_input)[:, 1]
        else:
            prob_up = self.classifier.model.predict_proba(X_input)[:, 1]
        prob_down = 1 - prob_up
        
        # 2. 상승 시 예상 수익률 (종가)
        if hasattr(self.regressor_up, 'predict'):
            return_if_up = self.regressor_up.predict(X_input)
        else:
            return_if_up = self.regressor_up.model.predict(X_input)
        
        # 3. 하락 시 예상 손실률
        if hasattr(self.regressor_down, 'predict'):
            return_if_down = self.regressor_down.predict(X_input)
        else:
            return_if_down = self.regressor_down.model.predict(X_input)
        
        # 4. 기대 수익률 계산
        expected_return = (prob_up * return_if_up) + (prob_down * return_if_down)
        
        # 5. 예측 방향
        predicted_direction = (prob_up >= self.threshold).astype(int)
        
        # 결과 DataFrame 생성
        result = pd.DataFrame({
            'prob_up': prob_up,
            'prob_down': prob_down,
            'return_if_up': return_if_up,
            'return_if_down': return_if_down,
            'expected_return': expected_return,
            'predicted_direction': predicted_direction,
        }, index=X.index)
        
        # 6. 고가 예측 (Model 2-1이 있는 경우)
        if self.regressor_high is not None:
            if hasattr(self.regressor_high, 'predict'):
                max_return_if_up = self.regressor_high.predict(X_input)
            else:
                max_return_if_up = self.regressor_high.model.predict(X_input)
            
            result['max_return_if_up'] = max_return_if_up
            
            # 익절 목표가 (고가의 80%)
            take_profit_ratio = getattr(self.regressor_high, 'take_profit_ratio', 0.8)
            result['take_profit_target'] = max_return_if_up * take_profit_ratio
        
        return result
    
    def predict_direction(self, X: pd.DataFrame) -> np.ndarray:
        """
        방향만 예측 (0: 하락, 1: 상승)
        
        Args:
            X: Feature DataFrame
            
        Returns:
            예측 방향 배열
        """
        predictions = self.predict(X)
        return predictions['predicted_direction'].values
    
    def predict_expected_return(self, X: pd.DataFrame) -> np.ndarray:
        """
        기대 수익률만 예측
        
        Args:
            X: Feature DataFrame
            
        Returns:
            기대 수익률 배열 (%)
        """
        predictions = self.predict(X)
        return predictions['expected_return'].values
    
    def predict_with_ranking(
        self,
        X: pd.DataFrame,
        top_n: Optional[int] = None
    ) -> pd.DataFrame:
        """
        예측 + 기대 수익률 기준 랭킹
        
        Args:
            X: Feature DataFrame
            top_n: 상위 N개만 반환 (None이면 전체)
            
        Returns:
            랭킹이 포함된 예측 결과
        """
        predictions = self.predict(X)
        
        # 기대 수익률 기준 랭킹
        predictions['rank'] = predictions['expected_return'].rank(ascending=False, method='min')
        predictions = predictions.sort_values('expected_return', ascending=False)
        
        if top_n:
            predictions = predictions.head(top_n)
        
        return predictions
    
    def get_trading_signals(
        self,
        X: pd.DataFrame,
        min_prob_up: float = 0.4,
        min_expected_return: float = 0.0
    ) -> pd.DataFrame:
        """
        매매 신호 생성
        
        Args:
            X: Feature DataFrame
            min_prob_up: 최소 상승 확률
            min_expected_return: 최소 기대 수익률
            
        Returns:
            매매 신호 DataFrame
        """
        predictions = self.predict(X)
        
        # 매매 신호 조건
        predictions['signal'] = 'HOLD'
        
        buy_mask = (
            (predictions['prob_up'] >= min_prob_up) &
            (predictions['expected_return'] >= min_expected_return)
        )
        predictions.loc[buy_mask, 'signal'] = 'BUY'
        
        # 익절/손절 목표가
        if 'take_profit_target' in predictions.columns:
            predictions['take_profit_price'] = predictions['take_profit_target']
        
        return predictions


def load_predictor(
    model_path: str,
    threshold: Optional[float] = None
) -> HybridPredictor:
    """
    저장된 모델 파일에서 HybridPredictor 생성
    
    Args:
        model_path: 모델 파일 경로 (.pkl)
        threshold: 상승 예측 임계값 (None이면 저장된 값 사용)
        
    Returns:
        HybridPredictor 인스턴스
    """
    # joblib unpickle을 위해 필요한 클래스를 sys.modules에 등록
    # 노트북에서 저장된 모델이 app.main.StackingHybridPredictor를 참조할 수 있도록
    import sys
    import types
    
    # app.main 모듈이 없으면 생성, 있으면 기존 모듈 사용
    if 'app.main' not in sys.modules:
        main_module = types.ModuleType('app.main')
        sys.modules['app.main'] = main_module
    else:
        main_module = sys.modules['app.main']
    
    # StackingHybridPredictor를 HybridPredictor로 매핑
    # (노트북에서 저장된 모델이 이 클래스를 참조하지만, 실제로는 HybridPredictor를 사용)
    if not hasattr(main_module, 'StackingHybridPredictor'):
        main_module.StackingHybridPredictor = HybridPredictor
    
    path = Path(model_path)
    
    if not path.exists():
        raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {path}")
    
    data = joblib.load(path)
    
    # 데이터에서 모델 추출
    classifier = data.get('stacking_clf')
    regressor_up = data.get('stacking_reg_up')
    regressor_down = data.get('stacking_reg_down')
    regressor_high = data.get('stacking_reg_up_max')
    features = data.get('features')
    saved_threshold = data.get('optimal_threshold', 0.4)
    
    # threshold 결정
    if threshold is None:
        threshold = saved_threshold
    
    print(f"✓ 모델 로드 완료: {path}")
    print(f"  - Features: {len(features) if features else 'N/A'}개")
    print(f"  - Threshold: {threshold}")
    print(f"  - 생성일시: {data.get('created_at', 'N/A')}")
    
    # HybridPredictor 생성
    predictor = HybridPredictor(
        classifier=classifier,
        regressor_up=regressor_up,
        regressor_down=regressor_down,
        regressor_high=regressor_high,
        threshold=threshold,
        features=features
    )
    
    return predictor


def predict_from_file(
    model_path: str,
    data: pd.DataFrame,
    threshold: Optional[float] = None
) -> pd.DataFrame:
    """
    파일에서 모델 로드 후 바로 예측
    
    Args:
        model_path: 모델 파일 경로
        data: 예측할 데이터
        threshold: 상승 예측 임계값
        
    Returns:
        예측 결과 DataFrame
    """
    predictor = load_predictor(model_path, threshold)
    return predictor.predict(data)
