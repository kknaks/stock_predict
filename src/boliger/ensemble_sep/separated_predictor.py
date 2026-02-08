"""Separated Predictor - 분리 학습된 모델의 최종 예측기.

BaseEnsemble + MetaLearner를 결합하여 최종 예측 제공.
백테스트 및 실시간 추론에 사용.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from boliger.ensemble_sep.base_ensemble import BaseEnsemble, TASKS
from boliger.ensemble_sep.meta_learner import (
    MetaLearner,
    WeightedAverageMetaLearner,
    StackingMetaLearner,
    create_meta_learner,
)
from boliger.data.dataset import StockDataset, collate_fn, FEATURE_COLS, TARGET_COLS


class SeparatedPredictor:
    """분리 학습된 모델 Predictor.

    9개 모델 (3 task × 5 arch) + Meta Learner로 앙상블 예측 수행.

    Args:
        ensemble_mode: 앙상블 방식.
            - 'simple_avg': 단순 평균
            - 'weighted_avg': 가중 평균 (config 가중치)
            - 'learned_avg': 검증셋 기반 가중치 자동 계산
            - 'stacking_ridge': Ridge 스태킹
            - 'stacking_xgb': XGBoost 스태킹
        model_dir: 모델 파일 디렉토리.
        config_dir: config 파일 디렉토리.
        ensemble_config_path: 앙상블 config 경로.
        device: torch device.
    """

    def __init__(
        self,
        ensemble_mode: str = "weighted_avg",
        model_dir: str = "models_sep",
        config_dir: str = "configs/sep",
        ensemble_config_path: Optional[str] = None,
        device: str = "cpu",
    ):
        self.ensemble_mode = ensemble_mode
        self.model_dir = Path(model_dir)
        self.config_dir = Path(config_dir)
        self.device = device

        # Ensemble config 로드
        if ensemble_config_path:
            self.ensemble_cfg = OmegaConf.load(ensemble_config_path)
        else:
            default_path = self.config_dir / "ensemble" / "default.yaml"
            if default_path.exists():
                self.ensemble_cfg = OmegaConf.load(default_path)
            else:
                self.ensemble_cfg = OmegaConf.create({})

        # Base Ensemble 초기화
        self.base_ensemble = BaseEnsemble(
            model_dir=str(self.model_dir),
            config_dir=str(self.config_dir),
            device=device,
        )

        # Task별 Meta Learner 초기화
        self.meta_learners: dict[str, MetaLearner] = {}
        self._init_meta_learners()

        # 로드 상태
        self._loaded = False

    def _init_meta_learners(self) -> None:
        """Task별 Meta Learner 초기화."""
        for task in TASKS:
            # Task별 가중치 로드
            task_weights = {}
            if "weights" in self.ensemble_cfg and task in self.ensemble_cfg.weights:
                task_weights = dict(self.ensemble_cfg.weights[task])

            # Meta Learner 생성
            if self.ensemble_mode in ("simple_avg", "learned_avg"):
                ml = create_meta_learner(self.ensemble_mode, self.ensemble_cfg)
            elif self.ensemble_mode == "weighted_avg":
                ml = WeightedAverageMetaLearner(weights=task_weights)
            elif self.ensemble_mode in ("stacking_ridge", "stacking_xgb"):
                ml = create_meta_learner(self.ensemble_mode, self.ensemble_cfg)
            else:
                raise ValueError(f"Unknown ensemble mode: {self.ensemble_mode}")

            self.meta_learners[task] = ml

    def load_models(self) -> "SeparatedPredictor":
        """모든 모델 로드."""
        self.base_ensemble.load_all_models()
        self._loaded = True
        return self

    def set_weights(self, task: str, weights: dict[str, float]) -> None:
        """특정 Task의 모델별 가중치 수동 설정.

        Args:
            task: "cls", "high", "low".
            weights: {"lstm": 0.2, "cnn": 0.2, "mlp": 0.2, "xgboost": 0.3, "rf": 0.1}.
        """
        if task not in TASKS:
            raise ValueError(f"Unknown task: {task}. Must be one of {TASKS}.")

        ml = self.meta_learners[task]
        if isinstance(ml, WeightedAverageMetaLearner):
            ml.set_weights(weights)
        else:
            print(f"Warning: set_weights() only works with weighted_avg mode. "
                  f"Current mode: {self.ensemble_mode}")

    def get_weights(self, task: str) -> dict[str, float]:
        """현재 가중치 조회."""
        if task not in TASKS:
            raise ValueError(f"Unknown task: {task}.")

        ml = self.meta_learners[task]
        if isinstance(ml, WeightedAverageMetaLearner):
            return ml.get_weights()
        elif isinstance(ml, StackingMetaLearner) and ml._is_fitted:
            return ml.get_coefficients()
        return {}

    def train_meta_learner(
        self,
        task: str,
        val_dataset: StockDataset,
        flat_val_features: Optional[np.ndarray] = None,
        batch_size: int = 512,
    ) -> None:
        """검증셋으로 메타 러너 학습 (stacking용).

        Args:
            task: "cls", "high", "low".
            val_dataset: 검증 데이터셋.
            flat_val_features: Tree 모델용 플랫 피처.
            batch_size: 배치 크기.
        """
        if not self._loaded:
            raise RuntimeError("Models not loaded. Call load_models() first.")

        print(f"\n=== Training Meta Learner for {task} ===")

        # 데이터 로드
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )

        # 모든 샘플에 대해 base 예측 수집
        all_predictions = {name: [] for name in self.base_ensemble.get_loaded_models(task)}
        all_targets = []

        for batch_idx, (padded, lengths, inp2, targets) in enumerate(val_loader):
            # Base 예측 수집
            batch_flat = None
            if flat_val_features is not None:
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(flat_val_features))
                batch_flat = flat_val_features[start_idx:end_idx]

            base_preds = self.base_ensemble.predict_all_base(
                task, padded, lengths, inp2, batch_flat
            )

            for name, pred in base_preds.items():
                all_predictions[name].append(pred)

            # 타겟 수집
            if task == "cls":
                target_vals = targets[:, 0].numpy()  # day_positive
            elif task == "high":
                target_vals = targets[:, 1].numpy()  # day_high_return
            else:
                target_vals = targets[:, 2].numpy()  # day_low_return
            all_targets.append(target_vals)

        # 배열 결합
        for name in all_predictions:
            all_predictions[name] = np.concatenate(all_predictions[name])
        all_targets = np.concatenate(all_targets)

        print(f"  Samples: {len(all_targets):,}")
        print(f"  Models: {list(all_predictions.keys())}")

        # Meta Learner 학습
        ml = self.meta_learners[task]
        if self.ensemble_mode == "learned_avg":
            ml.train(all_predictions, all_targets)
            print(f"  Learned weights: {ml.get_weights()}")
        elif isinstance(ml, StackingMetaLearner):
            ml.train(all_predictions, all_targets)
        else:
            print(f"  Skip training (mode={self.ensemble_mode})")

    @torch.no_grad()
    def predict_tensors(
        self,
        input1: torch.Tensor,
        lengths: torch.Tensor,
        input2: torch.Tensor,
        flat_features: Optional[np.ndarray] = None,
    ) -> dict[str, torch.Tensor]:
        """텐서 입력 예측.

        Args:
            input1: (batch, seq_len, features) 시계열 입력.
            lengths: (batch,) 시퀀스 길이.
            input2: (batch, 12) open_info.
            flat_features: (batch, n_features) Tree용 플랫 피처.

        Returns:
            {
                "prob": (batch, 1) 상승 확률,
                "high_return": (batch, 1) 예상 고점,
                "low_return": (batch, 1) 예상 저점,
            }
        """
        if not self._loaded:
            raise RuntimeError("Models not loaded. Call load_models() first.")

        results = {}

        for task in TASKS:
            # Base 예측 수집
            base_preds = self.base_ensemble.predict_all_base(
                task, input1, lengths, input2, flat_features
            )

            # Meta Learner로 앙상블
            ml = self.meta_learners[task]
            final_pred = ml.predict(base_preds)

            # 출력 키 결정
            if task == "cls":
                results["prob"] = torch.from_numpy(final_pred[:, np.newaxis]).float()
            elif task == "high":
                results["high_return"] = torch.from_numpy(final_pred[:, np.newaxis]).float()
            else:
                results["low_return"] = torch.from_numpy(final_pred[:, np.newaxis]).float()

        return results

    def predict_dataloader(
        self,
        dataloader: DataLoader,
        flat_features: Optional[np.ndarray] = None,
    ) -> dict[str, np.ndarray]:
        """DataLoader로 배치 예측.

        Args:
            dataloader: 데이터 로더.
            flat_features: Tree용 플랫 피처 (전체).

        Returns:
            {
                "prob": (n_samples,) 상승 확률,
                "high_return": (n_samples,) 예상 고점,
                "low_return": (n_samples,) 예상 저점,
            }
        """
        all_probs = []
        all_highs = []
        all_lows = []

        sample_idx = 0
        for padded, lengths, inp2, _ in dataloader:
            actual_batch_size = padded.size(0)

            batch_flat = None
            if flat_features is not None:
                end_idx = min(sample_idx + actual_batch_size, len(flat_features))
                batch_flat = flat_features[sample_idx:end_idx]
                sample_idx = end_idx

            # flat_features가 None이면 base_ensemble에서 자동 생성
            preds = self.predict_tensors(padded, lengths, inp2, batch_flat)
            all_probs.append(preds["prob"].numpy())
            all_highs.append(preds["high_return"].numpy())
            all_lows.append(preds["low_return"].numpy())

        return {
            "prob": np.concatenate(all_probs).squeeze(-1),
            "high_return": np.concatenate(all_highs).squeeze(-1),
            "low_return": np.concatenate(all_lows).squeeze(-1),
        }

    def save_weights(self, path: str) -> None:
        """현재 가중치를 YAML로 저장.

        Args:
            path: 저장 경로.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        weights_dict = {"mode": self.ensemble_mode, "weights": {}}

        for task in TASKS:
            ml = self.meta_learners[task]
            if isinstance(ml, WeightedAverageMetaLearner):
                weights_dict["weights"][task] = ml.get_weights()
            elif isinstance(ml, StackingMetaLearner) and ml._is_fitted:
                weights_dict["weights"][task] = ml.get_coefficients()

        OmegaConf.save(OmegaConf.create(weights_dict), path)
        print(f"Weights saved to: {path}")

    def load_weights(self, path: str) -> None:
        """YAML에서 가중치 로드.

        Args:
            path: 로드 경로.
        """
        path = Path(path)
        cfg = OmegaConf.load(path)

        for task in TASKS:
            if task in cfg.get("weights", {}):
                weights = dict(cfg.weights[task])
                self.set_weights(task, weights)

        print(f"Weights loaded from: {path}")

    def save_meta_learners(self, save_dir: str) -> None:
        """Meta Learner 모델 저장 (stacking용).

        Args:
            save_dir: 저장 디렉토리.
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        for task in TASKS:
            ml = self.meta_learners[task]
            if isinstance(ml, StackingMetaLearner):
                ml.save(save_dir / f"{ml.method}_{task}.pkl")
            elif isinstance(ml, WeightedAverageMetaLearner):
                ml.save(save_dir / f"{task}_weights.yaml")

        print(f"Meta learners saved to: {save_dir}")

    def load_meta_learners(self, save_dir: str) -> None:
        """Meta Learner 모델 로드 (stacking용).

        Args:
            save_dir: 로드 디렉토리.
        """
        save_dir = Path(save_dir)

        for task in TASKS:
            ml = self.meta_learners[task]
            if isinstance(ml, StackingMetaLearner):
                path = save_dir / f"{ml.method}_{task}.pkl"
                if path.exists():
                    ml.load(path)
            elif isinstance(ml, WeightedAverageMetaLearner):
                path = save_dir / f"{task}_weights.yaml"
                if path.exists():
                    ml.load(path)

        print(f"Meta learners loaded from: {save_dir}")

    def get_model_info(self) -> dict:
        """모델 정보 요약."""
        return {
            "ensemble_mode": self.ensemble_mode,
            "loaded": self._loaded,
            "loaded_models": self.base_ensemble.get_all_loaded_models() if self._loaded else {},
            "weights": {task: self.get_weights(task) for task in TASKS},
        }
