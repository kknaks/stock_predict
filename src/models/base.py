"""
공통 설정 및 베이스 클래스

모든 모델에서 공유하는 설정값과 유틸리티 함수
"""

from typing import Optional, List, Tuple, Dict, Any
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# ============================================================
# Feature 및 Target 정의
# ============================================================

FEATURE_COLS = [
    # 갭 관련
    'gap_pct',
    
    # 전일 패턴
    'prev_return', 'prev_range_pct', 'prev_upper_shadow', 'prev_lower_shadow',
    
    # 거래량
    'volume_ratio',
    
    # 기술적 지표
    'rsi_14', 'atr_14', 'atr_ratio', 'bollinger_position',
    'return_5d', 'return_20d', 'consecutive_up_days',
    
    # 이동평균
    'above_ma5', 'above_ma20', 'above_ma50', 'ma5_ma20_cross',
    
    # 시장 컨텍스트
    'market_gap_diff',
    
    # 시간 Features
    'day_of_week', 'month', 'is_month_start', 'is_month_end', 'is_quarter_end',
]

# 타겟 변수
TARGET_DIRECTION = 'target_direction'
TARGET_RETURN = 'target_return'
TARGET_MAX_RETURN = 'target_max_return'

# 핵심 컬럼 (NaN이 있으면 안 되는 필수 컬럼)
CORE_FEATURES = ['gap_pct', 'prev_return', 'market_gap_diff', TARGET_DIRECTION, TARGET_RETURN]

# ============================================================
# 기본 하이퍼파라미터
# ============================================================

DEFAULT_RANDOM_STATE = 42
DEFAULT_N_ESTIMATORS = 300
DEFAULT_MAX_DEPTH_RF = 15
DEFAULT_MAX_DEPTH_XGB = 6
DEFAULT_MAX_DEPTH_LGB = 15
DEFAULT_LEARNING_RATE = 0.1
DEFAULT_MIN_SAMPLES_SPLIT = 10
DEFAULT_CV_FOLDS = 5
DEFAULT_THRESHOLD = 0.4

# ============================================================
# 데이터 준비 함수
# ============================================================

def get_available_features(df: pd.DataFrame) -> List[str]:
    """
    DataFrame에서 사용 가능한 Feature 컬럼 반환
    
    Args:
        df: DataFrame
        
    Returns:
        사용 가능한 Feature 컬럼 리스트
    """
    return [f for f in FEATURE_COLS if f in df.columns]


def prepare_data_for_training(
    df: pd.DataFrame,
    include_max_return: bool = True
) -> pd.DataFrame:
    """
    학습용 데이터 전처리
    
    Args:
        df: 원본 DataFrame
        include_max_return: target_max_return 포함 여부
        
    Returns:
        전처리된 DataFrame
    """
    available_features = get_available_features(df)
    
    # 필요한 컬럼 정의
    target_cols = [TARGET_DIRECTION, TARGET_RETURN]
    if include_max_return and TARGET_MAX_RETURN in df.columns:
        target_cols.append(TARGET_MAX_RETURN)
    
    required_cols = list(dict.fromkeys(
        available_features + target_cols + ['date', 'InfoCode']
    ))
    
    # 필요한 컬럼만 선택하고 중복 제거
    df_model = df[required_cols].drop_duplicates().copy()
    
    # 핵심 Feature NaN 제거
    core_cols = [c for c in CORE_FEATURES if c in df_model.columns]
    df_model = df_model.dropna(subset=core_cols)
    
    # 나머지 Feature fillna
    for col in available_features:
        if col not in CORE_FEATURES and col in df_model.columns:
            if df_model[col].isna().sum() > 0:
                if df_model[col].dtype in ['float64', 'int64']:
                    df_model[col] = df_model[col].fillna(df_model[col].median())
                else:
                    df_model[col] = df_model[col].fillna(0)
    
    return df_model


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.1,
    valid_size: float = 0.1,
    random_state: int = DEFAULT_RANDOM_STATE,
    stratify_col: Optional[str] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    데이터를 Train/Valid/Test로 분할
    
    Args:
        df: DataFrame
        test_size: 테스트 비율 (전체 대비)
        valid_size: 검증 비율 (Train+Valid 대비)
        random_state: 랜덤 시드
        stratify_col: 층화 추출 컬럼
        
    Returns:
        (df_train, df_valid, df_test)
    """
    stratify = df[stratify_col] if stratify_col and stratify_col in df.columns else None
    
    # Train+Valid와 Test 분할
    df_temp, df_test = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=stratify
    )
    
    # Train과 Valid 분할
    stratify_temp = df_temp[stratify_col] if stratify_col and stratify_col in df_temp.columns else None
    valid_ratio = valid_size / (1 - test_size)
    
    df_train, df_valid = train_test_split(
        df_temp, test_size=valid_ratio, random_state=random_state, stratify=stratify_temp
    )
    
    return df_train, df_valid, df_test


def split_xy(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    DataFrame을 X, y로 분리
    
    Args:
        df: DataFrame
        target_col: 타겟 컬럼명
        feature_cols: Feature 컬럼 리스트 (None이면 자동 탐지)
        
    Returns:
        (X, y)
    """
    if feature_cols is None:
        feature_cols = get_available_features(df)
    
    X = df[feature_cols]
    y = df[target_col]
    
    return X, y


# ============================================================
# 베이스 모델 클래스
# ============================================================

class BaseStackingModel:
    """모든 Stacking 모델의 베이스 클래스"""
    
    def __init__(
        self,
        n_estimators: int = DEFAULT_N_ESTIMATORS,
        max_depth_rf: int = DEFAULT_MAX_DEPTH_RF,
        max_depth_xgb: int = DEFAULT_MAX_DEPTH_XGB,
        max_depth_lgb: int = DEFAULT_MAX_DEPTH_LGB,
        learning_rate: float = DEFAULT_LEARNING_RATE,
        cv_folds: int = DEFAULT_CV_FOLDS,
        random_state: int = DEFAULT_RANDOM_STATE,
        n_jobs: int = -1,
        verbose: int = 1
    ):
        self.n_estimators = n_estimators
        self.max_depth_rf = max_depth_rf
        self.max_depth_xgb = max_depth_xgb
        self.max_depth_lgb = max_depth_lgb
        self.learning_rate = learning_rate
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose
        
        self.model = None
        self.features = None
        self.is_fitted = False
        
        # XGBoost/LightGBM 사용 가능 여부 확인
        self._check_libraries()
    
    def _check_libraries(self):
        """외부 라이브러리 사용 가능 여부 확인"""
        try:
            import xgboost
            self.has_xgb = True
        except ImportError:
            self.has_xgb = False
            
        try:
            import lightgbm
            self.has_lgb = True
        except ImportError:
            self.has_lgb = False
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'BaseStackingModel':
        """모델 학습 (서브클래스에서 구현)"""
        raise NotImplementedError
    
    def predict(self, X: pd.DataFrame):
        """예측 (서브클래스에서 구현)"""
        raise NotImplementedError
    
    def get_params(self) -> Dict[str, Any]:
        """모델 파라미터 반환"""
        return {
            'n_estimators': self.n_estimators,
            'max_depth_rf': self.max_depth_rf,
            'max_depth_xgb': self.max_depth_xgb,
            'max_depth_lgb': self.max_depth_lgb,
            'learning_rate': self.learning_rate,
            'cv_folds': self.cv_folds,
            'random_state': self.random_state,
            'features': self.features,
            'is_fitted': self.is_fitted,
        }
