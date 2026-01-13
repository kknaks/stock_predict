"""
모델 저장/로드 유틸리티

학습된 모델을 pkl 파일로 저장하고 로드하는 기능
"""

from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, Union
import joblib

from .classifier import StackingClassifierModel
from .regressor_up import StackingRegressorUp
from .regressor_high import StackingRegressorHigh
from .regressor_down import StackingRegressorDown


class ModelStorage:
    """모델 저장/로드 관리 클래스"""
    
    def __init__(self, base_dir: str = "models"):
        """
        Args:
            base_dir: 모델 저장 기본 디렉토리
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
    
    def save_model(
        self,
        model: Any,
        filename: str,
        subdir: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Path:
        """
        단일 모델 저장
        
        Args:
            model: 저장할 모델
            filename: 파일명 (확장자 없이)
            subdir: 하위 디렉토리
            metadata: 추가 메타데이터
            
        Returns:
            저장된 파일 경로
        """
        if subdir:
            save_dir = self.base_dir / subdir
        else:
            save_dir = self.base_dir
        
        save_dir.mkdir(parents=True, exist_ok=True)
        
        filepath = save_dir / f"{filename}.pkl"
        
        # 메타데이터와 함께 저장
        save_data = {
            'model': model,
            'params': model.get_params() if hasattr(model, 'get_params') else {},
            'saved_at': datetime.now().isoformat(),
        }
        
        if metadata:
            save_data['metadata'] = metadata
        
        joblib.dump(save_data, filepath)
        
        print(f"✓ 모델 저장: {filepath}")
        print(f"  파일 크기: {filepath.stat().st_size / 1024 / 1024:.2f} MB")
        
        return filepath
    
    def load_model(
        self,
        filename: str,
        subdir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        단일 모델 로드
        
        Args:
            filename: 파일명 (확장자 포함 또는 미포함)
            subdir: 하위 디렉토리
            
        Returns:
            로드된 데이터 딕셔너리 {'model': ..., 'params': ..., ...}
        """
        if subdir:
            load_dir = self.base_dir / subdir
        else:
            load_dir = self.base_dir
        
        if not filename.endswith('.pkl'):
            filename = f"{filename}.pkl"
        
        filepath = load_dir / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {filepath}")
        
        data = joblib.load(filepath)
        
        print(f"✓ 모델 로드: {filepath}")
        
        return data
    
    def save_all_models(
        self,
        classifier: StackingClassifierModel,
        regressor_up: StackingRegressorUp,
        regressor_high: Optional[StackingRegressorHigh],
        regressor_down: StackingRegressorDown,
        test_metrics: Optional[Dict[str, Any]] = None,
        subdir: str = "stacking"
    ) -> Path:
        """
        모든 모델을 하나의 파일로 저장
        
        Args:
            classifier: Model 1 (분류)
            regressor_up: Model 2 (상승 종가)
            regressor_high: Model 2-1 (상승 고가)
            regressor_down: Model 3 (하락)
            test_metrics: 테스트 성능 지표
            subdir: 하위 디렉토리
            
        Returns:
            저장된 파일 경로
        """
        save_dir = self.base_dir / subdir
        save_dir.mkdir(parents=True, exist_ok=True)
        
        filepath = save_dir / "stacking_hybrid_model.pkl"
        
        # 통합 저장 데이터
        save_data = {
            'stacking_clf': classifier,
            'stacking_reg_up': regressor_up,
            'stacking_reg_up_max': regressor_high,
            'stacking_reg_down': regressor_down,
            'features': classifier.features if classifier.features else [],
            'optimal_threshold': classifier.threshold,
            'test_metrics': test_metrics,
            'created_at': datetime.now().isoformat(),
        }
        
        joblib.dump(save_data, filepath)
        
        print("=" * 80)
        print("모델 저장 완료")
        print("=" * 80)
        print(f"\n저장 경로: {filepath}")
        print(f"파일 크기: {filepath.stat().st_size / 1024 / 1024:.2f} MB")
        print("\n저장된 모델:")
        print(f"  - Stacking Classifier (Model 1)")
        print(f"  - Stacking Regressor Up (Model 2 - 종가)")
        if regressor_high is not None:
            print(f"  - Stacking Regressor Up Max (Model 2-1 - 고가)")
        print(f"  - Stacking Regressor Down (Model 3)")
        
        return filepath
    
    def load_all_models(
        self,
        subdir: str = "stacking",
        filename: str = "stacking_hybrid_model.pkl"
    ) -> Dict[str, Any]:
        """
        통합 저장된 모든 모델 로드
        
        Args:
            subdir: 하위 디렉토리
            filename: 파일명
            
        Returns:
            모든 모델과 메타데이터
        """
        filepath = self.base_dir / subdir / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {filepath}")
        
        data = joblib.load(filepath)
        
        print("=" * 80)
        print("모델 로드 완료")
        print("=" * 80)
        print(f"\n로드 경로: {filepath}")
        print(f"생성일시: {data.get('created_at', 'N/A')}")
        print(f"Features: {len(data.get('features', []))}개")
        print(f"Threshold: {data.get('optimal_threshold', 'N/A')}")
        
        return data
    
    def list_models(self, subdir: Optional[str] = None) -> list:
        """
        저장된 모델 파일 목록 조회
        
        Args:
            subdir: 하위 디렉토리 (None이면 전체)
            
        Returns:
            모델 파일 경로 리스트
        """
        if subdir:
            search_dir = self.base_dir / subdir
        else:
            search_dir = self.base_dir
        
        if not search_dir.exists():
            return []
        
        return list(search_dir.glob("**/*.pkl"))
