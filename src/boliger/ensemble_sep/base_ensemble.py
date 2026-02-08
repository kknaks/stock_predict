"""Base Ensemble - 5개 Base 모델 로드 및 예측 수집.

Neural Networks: LSTM, CNN, MLP
Tree Models: XGBoost, RF

각 모델은 3가지 task (cls, high, low) 별로 별도 학습됨.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

from boliger.models_sep import (
    LSTMPredictor,
    CNNPredictor,
    MLPPredictor,
    XGBoostWrapper,
    RandomForestWrapper,
)
from boliger.data.dataset import FEATURE_COLS

# 상수 정의
NN_MODEL_NAMES = ["lstm", "cnn", "mlp"]
TREE_MODEL_NAMES = ["xgboost", "rf"]
ALL_MODEL_NAMES = NN_MODEL_NAMES + TREE_MODEL_NAMES
TASKS = ["cls", "high", "low"]

# 피처 수
INPUT_SIZE = len(FEATURE_COLS)  # 33


class BaseEnsemble:
    """5개 Base 모델 관리 및 예측 수집.

    Neural Networks와 Tree Models를 로드하고,
    각 task별로 모든 모델의 예측을 수집하는 역할.

    Args:
        model_dir: 모델 파일 디렉토리 (기본: models_sep).
        config_dir: config 파일 디렉토리 (기본: configs/sep).
        device: torch device (기본: cpu).
    """

    def __init__(
        self,
        model_dir: str = "models_sep",
        config_dir: str = "configs/sep",
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.model_dir = Path(model_dir)
        self.config_dir = Path(config_dir)

        # Config 로드
        self.model_cfgs = self._load_model_configs()

        # 모델 저장소: {task: {model_name: model}}
        self.models: dict[str, dict[str, nn.Module]] = {
            "cls": {},
            "high": {},
            "low": {},
        }

        # 로드 상태 추적
        self._loaded = False

    def _load_model_configs(self) -> dict[str, DictConfig]:
        """모델별 config 로드."""
        cfgs = {}
        for name in NN_MODEL_NAMES:
            cfg_path = self.config_dir / "model" / f"{name}.yaml"
            if cfg_path.exists():
                cfgs[name] = OmegaConf.load(cfg_path)
            else:
                print(f"Warning: Config not found: {cfg_path}")
                cfgs[name] = OmegaConf.create({})

        for name in TREE_MODEL_NAMES:
            cfg_path = self.config_dir / "model" / f"{name}.yaml"
            if cfg_path.exists():
                cfgs[name] = OmegaConf.load(cfg_path)
            else:
                print(f"Warning: Config not found: {cfg_path}")
                cfgs[name] = OmegaConf.create({})

        return cfgs

    def load_all_models(self) -> "BaseEnsemble":
        """모든 모델 로드 (NN + Tree, 모든 task)."""
        print("Loading all base models...")

        for task in TASKS:
            self._load_nn_models(task)
            self._load_tree_models(task)

        self._loaded = True
        self._print_load_summary()
        return self

    def _load_nn_models(self, task: str) -> None:
        """Neural Network 모델 로드 (LSTM, CNN, MLP).

        Args:
            task: 'cls', 'high', 'low'.
        """
        model_classes = {
            "lstm": LSTMPredictor,
            "cnn": CNNPredictor,
            "mlp": MLPPredictor,
        }

        for name, model_cls in model_classes.items():
            path = self.model_dir / task / f"{name}_best.pt"
            if not path.exists():
                print(f"  [Skip] {task}/{name}: not found at {path}")
                continue

            try:
                model = model_cls(
                    input_size=INPUT_SIZE,
                    cfg=self.model_cfgs[name],
                    mode=task,
                )
                state_dict = torch.load(path, map_location=self.device, weights_only=True)
                model.load_state_dict(state_dict)
                model.to(self.device).eval()
                self.models[task][name] = model
                print(f"  [Loaded] {task}/{name}")
            except Exception as e:
                print(f"  [Error] {task}/{name}: {e}")

    def _load_tree_models(self, task: str) -> None:
        """Tree 모델 로드 (XGBoost, RF).

        Args:
            task: 'cls', 'high', 'low'.
        """
        model_classes = {
            "xgboost": XGBoostWrapper,
            "rf": RandomForestWrapper,
        }

        for name, model_cls in model_classes.items():
            path = self.model_dir / task / f"{name}_best.pkl"
            if not path.exists():
                print(f"  [Skip] {task}/{name}: not found at {path}")
                continue

            try:
                model = model_cls(cfg=self.model_cfgs[name], mode=task)
                model.load(path)
                self.models[task][name] = model
                print(f"  [Loaded] {task}/{name}")
            except Exception as e:
                print(f"  [Error] {task}/{name}: {e}")

    def _print_load_summary(self) -> None:
        """로드 현황 요약 출력."""
        print("\n=== Model Load Summary ===")
        print(f"{'Task':<8} {'Models Loaded':<30}")
        print("-" * 40)
        for task in TASKS:
            loaded_names = list(self.models[task].keys())
            print(f"{task:<8} {', '.join(loaded_names) if loaded_names else '(none)'}")
        print()

    @torch.no_grad()
    def predict_nn(
        self,
        task: str,
        input1: torch.Tensor,
        lengths: torch.Tensor,
        input2: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Neural Network 모델들의 예측 수집.

        Args:
            task: 'cls', 'high', 'low'.
            input1: (batch, seq_len, features) 시계열 입력.
            lengths: (batch,) 시퀀스 길이.
            input2: (batch, 12) open_info.

        Returns:
            {model_name: prediction_tensor} 딕셔너리.
            - cls: (batch, 1) prob
            - high: (batch, 1) high_return
            - low: (batch, 1) low_return
        """
        if not self._loaded:
            raise RuntimeError("Models not loaded. Call load_all_models() first.")

        input1 = input1.to(self.device)
        lengths = lengths.to(self.device)
        input2 = input2.to(self.device)

        predictions = {}
        for name in NN_MODEL_NAMES:
            if name not in self.models[task]:
                continue

            model = self.models[task][name]
            output = model(input1, lengths, input2)

            # task에 따라 적절한 출력 선택
            if task == "cls":
                predictions[name] = output["prob"]
            elif task == "high":
                predictions[name] = output["high_return"]
            elif task == "low":
                predictions[name] = output["low_return"]

        return predictions

    def predict_tree(
        self,
        task: str,
        flat_features: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Tree 모델들의 예측 수집.

        Args:
            task: 'cls', 'high', 'low'.
            flat_features: (batch, n_features) 플랫 피처.

        Returns:
            {model_name: prediction_array} 딕셔너리.
            - cls: (batch,) prob
            - high/low: (batch,) return
        """
        if not self._loaded:
            raise RuntimeError("Models not loaded. Call load_all_models() first.")

        predictions = {}
        for name in TREE_MODEL_NAMES:
            if name not in self.models[task]:
                continue

            model = self.models[task][name]

            if task == "cls":
                # 분류: predict_proba 사용
                predictions[name] = model.predict_proba(flat_features)
            else:
                # 회귀: predict 사용
                predictions[name] = model.predict(flat_features)

        return predictions

    def predict_all_base(
        self,
        task: str,
        input1: torch.Tensor,
        lengths: torch.Tensor,
        input2: torch.Tensor,
        flat_features: Optional[np.ndarray] = None,
    ) -> dict[str, np.ndarray]:
        """모든 Base 모델의 예측 수집.

        Args:
            task: 'cls', 'high', 'low'.
            input1: (batch, seq_len, features) 시계열 입력 (NN용).
            lengths: (batch,) 시퀀스 길이 (NN용).
            input2: (batch, 12) open_info (NN용).
            flat_features: (batch, n_features) 플랫 피처 (Tree용).
                          None이면 input1의 마지막 행 + input2 사용.

        Returns:
            {model_name: (batch,) prediction_array} 딕셔너리.
            모든 예측은 numpy array로 통일.
        """
        if not self._loaded:
            raise RuntimeError("Models not loaded. Call load_all_models() first.")

        predictions = {}

        # NN 예측
        nn_preds = self.predict_nn(task, input1, lengths, input2)
        for name, tensor in nn_preds.items():
            predictions[name] = tensor.cpu().numpy().squeeze(-1)  # (batch,)

        # Tree 예측 (flat_features 필요)
        if flat_features is None:
            # WARNING: input1의 마지막 행은 변곡점 전날 데이터이므로
            # 변곡점 당일 데이터로 학습된 Tree 모델과 불일치.
            # StockDataset.flat_features를 사용할 것을 권장.
            import warnings
            warnings.warn(
                "flat_features가 None입니다. input1의 마지막 행(변곡점 전날)을 "
                "사용하면 Tree 모델 성능이 크게 저하됩니다. "
                "StockDataset.flat_features를 전달하세요.",
                UserWarning,
                stacklevel=2,
            )
            input1_np = input1.cpu().numpy()
            batch_size = input1_np.shape[0]
            lengths_np = lengths.cpu().numpy()
            flat_features = np.zeros((batch_size, input1_np.shape[2]), dtype=np.float32)
            for i in range(batch_size):
                seq_len = int(lengths_np[i])
                flat_features[i] = input1_np[i, seq_len - 1, :]

        tree_preds = self.predict_tree(task, flat_features)
        for name, arr in tree_preds.items():
            predictions[name] = arr.astype(np.float32)  # (batch,)

        return predictions

    def get_loaded_models(self, task: str) -> list[str]:
        """특정 task에 로드된 모델 이름 반환."""
        return list(self.models[task].keys())

    def get_all_loaded_models(self) -> dict[str, list[str]]:
        """모든 task의 로드된 모델 이름 반환."""
        return {task: self.get_loaded_models(task) for task in TASKS}

    def is_loaded(self) -> bool:
        """모델 로드 여부."""
        return self._loaded
