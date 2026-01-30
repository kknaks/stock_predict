# 월간 모델 재학습 시스템 구현 계획

## 범위

AI 서버(`stock_predict`)에 월간 재학습 핵심 로직 구현. Airflow DAG는 나중에 별도 작업.

## 아키텍처

```
Kafka: model_retrain_command → AI Server 수신
  → DB에서 최신 데이터 로드 (src/data/)
  → 전처리 + 피처 엔지니어링 (src/features/)
  → 모델 학습 (src/models/trainer.py)
  → Champion-Challenger 비교
  → 통과 시 model_registry DB 업데이트 + 버전별 모델 저장
  → Kafka: model_retrain_result 발행
```

## 구현 단계

### Step 1: Kafka 메시지 스키마
**파일**: `app/kafka/retraining_schemas.py` (신규)
- `ModelRetrainCommandMessage`: trigger_type, config (data_start_date, threshold, n_estimators 등)
- `ModelRetrainResultMessage`: status, new_version, is_deployed, test_metrics, comparison_result

### Step 2: DB 모델 - model_registry 테이블
**파일**: `app/database/database/model_registry.py` (신규)
- SQLAlchemy 모델: version, model_path, training 메타데이터, test 메트릭 (accuracy, roc_auc, f1, mae 등), status (candidate/active/retired), comparison_result (JSONB)
- 기존 `app/database/database/base.py`의 Base, TimestampMixin 활용

### Step 3: Champion-Challenger 비교 로직
**파일**: `app/retraining/champion_challenger.py` (신규)
- ROC AUC 절대 최소값 체크 (< 0.60이면 reject)
- ROC AUC 하락폭 체크 (> 2% 하락이면 reject)
- F1 하락폭 체크 (> 5% 하락 시, ROC 개선 없으면 reject)
- 첫 모델은 auto-deploy

### Step 4: Model Registry 서비스
**파일**: `app/retraining/model_registry_service.py` (신규)
- CRUD: create_entry, get_active_model, activate_model, update_comparison_result
- 활성화 시 기존 모델 retired 처리

### Step 5: 데이터 파이프라인 (재학습용)
**파일**: `app/retraining/data_pipeline.py` (신규)
- 기존 `src/data/data_loader.py` + `src/data/preprocessor.py` + `src/features/` 활용
- DB → OHLCV + 시장지수 + 시가총액 로드 → 전처리 → 기술지표 → 시장 피처 → 학습용 DataFrame 반환

### Step 6: Trainer Wrapper (버전 관리 포함)
**파일**: `app/retraining/trainer_wrapper.py` (신규)
- `src/models/trainer.py`의 `StackingModelTrainer` 래핑
- 버전별 디렉토리 생성: `models/stacking/{version}/`
- 학습 → 저장 → 메트릭 반환

### Step 7: 재학습 핸들러 (오케스트레이터)
**파일**: `app/handler/model_retrain_handler.py` (신규)
- 전체 플로우 조율: 데이터 로드 → 학습 → 비교 → 배포 결정 → 레지스트리 저장
- 버전 생성: `v1.{YYMM}.{DD}` 형식
- active symlink 업데이트: `models/stacking/active -> {version}/`

### Step 8: Kafka Consumer/Producer 추가
**수정 파일**: `app/kafka/consumer.py` - `RetrainingConsumer` 클래스 추가
**수정 파일**: `app/kafka/producer.py` - `RetrainingProducer` 클래스 추가
**수정 파일**: `app/config/settings.py` - topic_retrain_command, topic_retrain_result 추가

### Step 9: main.py에 재학습 consumer 통합
**수정 파일**: `app/main.py`
- 별도 스레드에서 `RetrainingConsumer` 실행
- 기존 gap_candidate consumer와 병렬 동작

### Step 10: 기존 모델을 v1.0.0으로 마이그레이션
- `models/stacking/stacking_hybrid_model.pkl` → `models/stacking/v1.0.0/`
- `models/stacking/active` symlink 생성
- predictor가 active symlink 경로에서 모델 로드하도록 수정

## 수정 대상 파일 요약

| 파일 | 작업 |
|------|------|
| `app/kafka/retraining_schemas.py` | 신규 |
| `app/database/database/model_registry.py` | 신규 |
| `app/retraining/__init__.py` | 신규 |
| `app/retraining/champion_challenger.py` | 신규 |
| `app/retraining/model_registry_service.py` | 신규 |
| `app/retraining/data_pipeline.py` | 신규 |
| `app/retraining/trainer_wrapper.py` | 신규 |
| `app/handler/model_retrain_handler.py` | 신규 |
| `app/kafka/consumer.py` | 수정 - RetrainingConsumer 추가 |
| `app/kafka/producer.py` | 수정 - RetrainingProducer 추가 |
| `app/config/settings.py` | 수정 - 토픽명 추가 |
| `app/main.py` | 수정 - 재학습 consumer 스레드 추가 |
| `app/handler/gap_predict_handler.py` | 수정 - active symlink 경로 사용 |

## 노트북 vs src 코드 비교 결과

**핵심 로직 동일 확인됨.** src 코드를 그대로 활용 가능.

차이점 2개 (모두 실질 영향 없음):
1. Model 2-1 분할: 노트북이 별도 split 후 교정하는 반면, src는 인덱스 재사용 (src가 더 깔끔)
2. 저장 포맷 키 이름: `regressor_up_max` vs `stacking_reg_up_max` (로더가 양쪽 처리)

## 검증 방법

1. 단위 테스트: champion_challenger 비교 로직
2. 통합 테스트: 소량 데이터로 전체 파이프라인 (데이터 로드 → 학습 → 저장 → 레지스트리)
3. Kafka 테스트: command 메시지 발행 → result 메시지 수신 확인
