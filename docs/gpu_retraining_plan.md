# 리트레이닝 GPU 사용 계획

리트레이닝을 GPU(XGBoost/LightGBM)로 돌리기 위한 **방법과 수정 계획**만 정리합니다. 코드 수정은 이 계획 승인 후 진행합니다.

---

## 1. 현황

| 항목 | 내용 |
|------|------|
| 학습 라이브러리 | **XGBoost**, **LightGBM**, scikit-learn(RandomForest, Ridge 등) |
| GPU 지원 | **XGBoost**: CUDA 지원 (`device='cuda'`) / **LightGBM**: GPU 지원 (`device='gpu'`) |
| 호출 경로 | `retraining/trainer_wrapper.train_versioned_model()` → `StackingModelTrainer` → `StackingClassifierModel` / `StackingRegressor*` → `BaseStackingModel` |
| 현재 | 모든 학습이 **CPU + n_jobs** 로만 동작 |

RandomForest·Ridge 등은 GPU 미지원이므로, **XGBoost·LightGBM만 GPU**로 돌리고 나머지는 그대로 CPU 사용합니다.

---

## 2. 환경 조건 (GPU 쓸 수 있어야 할 것)

### 2.1 호스트에서 직접 Python 실행 시

- **NVIDIA 드라이버**: `nvidia-smi` 가 정상 동작
- **CUDA**: XGBoost GPU 빌드가 CUDA를 사용 (pip/poetry 기본 빌드는 대부분 CUDA 포함 wheel 설치)
- **LightGBM**: CUDA 또는 OpenCL; 일반 pip 설치도 GPU 옵션 지원

즉, **`nvidia-smi` 가 성공하면** 같은 환경에서 Python 리트레이닝을 GPU로 돌리는 건 보통 가능합니다.

### 2.2 Docker에서 리트레이닝 실행 시

- 호스트에 **nvidia-container-toolkit** 설치
- 컨테이너에 GPU 노출: `docker run --gpus all` 또는 compose 에 `deploy.resources.reservations.devices` (GPU 디바이스) 설정
- 이미지에 **CUDA 런타임**이 있거나, pip 설치한 XGBoost/LightGBM이 호스트 GPU에 접근할 수 있어야 함 (nvidia-docker가 호스트 드라이버/라이브러리를 넘겨줌)

현재 Dockerfile은 `python:3.11-slim` 기반이라 **CUDA 베이스 이미지가 아님**. GPU를 쓰려면:

- **옵션 A**: 그대로 두고, 호스트의 nvidia 드라이버만 컨테이너에 넘겨주기 (nvidia-container-toolkit). 많은 경우 pip 설치된 xgboost/lightgbm이 동작함.
- **옵션 B**: CUDA 포함 베이스 이미지로 변경 (이미지 크기·빌드 복잡도 증가).

우선 **옵션 A**로, 코드만 GPU 옵션을 넣고 Docker는 “GPU 노출만” 해주는 방식으로 계획합니다.

---

## 3. 코드 수정 계획 (어디를 어떻게 바꿀지)

### 3.1 GPU 사용 여부 결정 방식

- **환경 변수 우선**: `USE_GPU=1` (또는 `RETRAINING_USE_GPU=1`) 이 있으면 GPU 사용 시도.
- **없으면 자동 감지**: `nvidia-smi` 를 한 번 실행해서 성공하면 GPU 사용, 실패하면 CPU.
- **명시적 끄기**: `USE_GPU=0` 이면 무조건 CPU.

이렇게 하면 “리눅스 시스템 정보(실제로는 nvidia-smi)를 보고 판단”하는 요구를 만족합니다.

### 3.2 수정할 파일 및 내용

| 순서 | 파일 | 변경 내용 |
|------|------|-----------|
| 1 | `src/models/base.py` | ① GPU 사용 가능 여부 감지 함수 추가 (subprocess로 `nvidia-smi` 호출 또는 env 확인). ② `BaseStackingModel.__init__` 에 인자 `use_gpu: Optional[bool] = None` 추가. `None` 이면 위 규칙으로 자동 감지, `True`/`False` 이면 그대로 사용. ③ `self.use_gpu` 저장. |
| 2 | `src/models/classifier.py` | `_create_base_learners()` 에서 XGBoost/LightGBM 생성 시 `self.use_gpu` 이 True 면 **XGBoost**: `device='cuda'` 추가, **LightGBM**: `device='gpu'` 추가. GPU 사용 시 `n_jobs` 는 1로 두는 것이 일반적(선택). |
| 3 | `src/models/regressor_up.py` | 위와 동일하게 XGB/LGB 생성부에 `device='cuda'` / `device='gpu'` 조건부 추가. |
| 4 | `src/models/regressor_down.py` | 동일. |
| 5 | `src/models/regressor_high.py` | 동일. |
| 6 | `src/models/trainer.py` | ① `StackingModelTrainer.__init__` 에 `use_gpu: Optional[bool] = None` 추가. ② `train_classifier` / `train_regressor_up` / `train_regressor_high` / `train_regressor_down` 에서 각 모델 생성 시 `use_gpu=self.use_gpu` 전달 (BaseStackingModel 쪽으로 넘어가도록). |
| 7 | `retraining/trainer_wrapper.py` | ① `train_versioned_model()` 에 인자 `use_gpu: Optional[bool] = None` 추가. ② `StackingModelTrainer(..., use_gpu=use_gpu)` 로 전달. ③ (선택) 환경 변수 `USE_GPU` / `RETRAINING_USE_GPU` 를 읽어서 기본값으로 사용. |
| 8 | (선택) `retraining/handler.py` 등 | 리트레이닝을 트리거하는 진입점에서 `train_versioned_model(..., use_gpu=...)` 에 값을 넘기거나, env만으로 제어. |

**공통 규칙**

- GPU 사용 시에도 **RandomForest, Ridge 등은 그대로 CPU** (변경 없음).
- `use_gpu` 가 True 인데 실제로 GPU가 없거나 실패하면: XGB/LGB 생성 시 예외가 날 수 있으므로, 필요 시 try/except 로 감싸서 “GPU 실패 시 CPU로 fallback” 또는 “에러 메시지로 GPU 불가 안내” 중 하나 선택해서 처리.

---

## 4. XGBoost / LightGBM 파라미터 (참고)

- **XGBoost (2.x/3.x)**  
  - `device='cuda'`  
  - (구 API) `tree_method='gpu_hist'` 등은 버전에 따라 선택.

- **LightGBM**  
  - `device='gpu'`  
  - 필요 시 `gpu_use_dp=False` 등 옵션 추가 가능 (문제 생기면 그때 문서 참고).

---

## 5. 실행·검증 방법

1. **GPU 감지 확인**  
   - 리눅스에서 `nvidia-smi` 성공한 뒤, `USE_GPU=1` 또는 자동 감지로 리트레이닝 실행.  
   - 로그에 “Using GPU for XGBoost/LightGBM” 같은 한 줄이라도 찍어두면 확인하기 좋음.

2. **Docker에서 GPU 사용**  
   - compose 에서 retraining 서비스에만 GPU 할당:
     - `deploy.resources.reservations.devices: [driver: nvidia, count: 1, capabilities: [gpu]]` (compose v2 형식)
   - 실행 후 컨테이너 안에서 `nvidia-smi` 가 되면, 같은 컨테이너에서 리트레이닝도 GPU 사용 가능하다고 보면 됨.

3. **성능 확인**  
   - 동일 데이터로 `USE_GPU=0` vs `USE_GPU=1` 각각 실행해 보면서 학습 시간 비교.

---

## 6. 정리

- **목표**: 리트레이닝을 **GPU로 돌릴 수 있게** 하는 것.
- **방법**:  
  - 환경 변수 + `nvidia-smi` 로 **GPU 사용 여부를 먼저 계획·판단**하고,  
  - **BaseStackingModel → Classifier/Regressor → Trainer → trainer_wrapper** 순으로 `use_gpu` 를 한 겹씩 넘겨서,  
  - XGBoost/LightGBM 생성 시에만 `device='cuda'` / `device='gpu'` 를 넣는 방식.
- **수정 범위**: 위 3.2 표의 7~8개 파일. 새 의존성 추가는 없음 (기존 xgboost/lightgbm 그대로 사용).

이 계획대로 수정 진행해도 될지 알려주시면, 그 다음에 실제 패치 단계로 들어가겠습니다.
