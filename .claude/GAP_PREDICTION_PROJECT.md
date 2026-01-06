# 주식 갭 상승 예측 AI 모델 - 프로젝트 지침서

## 1. 프로젝트 개요

### 1.1 가설
> **"갭 상승으로 출발한 종목은 당일 장마감까지 상승할 것이다"**

### 1.2 목표
- 갭 상승 종목의 당일 종가 방향 예측 (상승/하락)
- 갭 상승 유형별 패턴 분석
- 실제 트레이딩 적용 가능한 모델 개발

### 1.3 범위
| 항목 | 내용 |
|------|------|
| 대상 시장 | NYSE, NASDAQ, AMEX |
| 데이터 기간 | 20년 (약 2005 ~ 2024) |
| 데이터 소스 | Refinitiv (DB) |
| 개발 환경 | Python |

---

## 2. 용어 정의

### 2.1 갭 상승 (Gap Up)
```
갭 상승률 = (당일 시가 - 전일 종가) / 전일 종가 × 100

데이터 수집 조건: 갭 상승률 > 0% (시가 > 전일 종가)
모델 학습 시 필터링 가능: 1%, 2%, 3% 등 실험
```

### 2.2 타겟 라벨 (예측 대상)
```
성공(1): 당일 종가 > 당일 시가
실패(0): 당일 종가 ≤ 당일 시가
```

### 2.3 갭 유형 분류
| 유형 | 정의 | 식별 방법 |
|------|------|-----------|
| Earnings Gap | 실적 발표로 인한 갭 | 실적 발표일 캘린더 |
| News Gap | 뉴스/이벤트로 인한 갭 | 뉴스 건수 > 0 |
| Market Gap | 시장 동반 상승 갭 | SPY 갭과 상관관계 |
| Technical Gap | 특별한 촉매 없음 | 위 조건 모두 해당 없음 |

---

## 3. 데이터 요구사항

### 3.1 필수 데이터 (Tier 1) - Phase 1

#### 가격 데이터 (OHLCV)
```sql
-- 필요 컬럼
symbol          -- 종목 티커
date            -- 거래일
open            -- 시가
high            -- 고가
low             -- 저가
close           -- 종가
adj_close       -- 수정 종가
volume          -- 거래량
```

#### 실적 데이터
```sql
earnings_date           -- 실적 발표일
eps_actual              -- 실제 EPS
eps_estimate            -- 예상 EPS
earnings_surprise_pct   -- 서프라이즈 (%)
revenue_actual          -- 실제 매출
revenue_estimate        -- 예상 매출
```

#### 뉴스/센티먼트 데이터
```sql
news_datetime       -- 뉴스 발행 시간
headline            -- 헤드라인
sentiment_score     -- 센티먼트 스코어 (-1 ~ 1)
relevance_score     -- 관련성 점수
```

#### 종목 메타 정보
```sql
symbol              -- 티커
exchange            -- 거래소
sector              -- 섹터
industry            -- 산업
market_cap          -- 시가총액
status              -- 상장/상장폐지 여부
delist_date         -- 상장폐지일 (해당 시)
```

### 3.2 보조 데이터 (Tier 2) - 고도화 단계

#### 프리마켓 데이터 ⭐ 고도화
```sql
premarket_open      -- 프리마켓 시가
premarket_high      -- 프리마켓 고가
premarket_low       -- 프리마켓 저가
premarket_close     -- 프리마켓 종가
premarket_volume    -- 프리마켓 거래량
```

#### 애널리스트 데이터
```sql
rating_date         -- 레이팅 날짜
rating_current      -- 현재 레이팅
rating_prior        -- 이전 레이팅
target_price        -- 목표가
target_price_prior  -- 이전 목표가
```

#### 시장 컨텍스트
```sql
-- 별도 수집 필요
spy_open, spy_close     -- S&P 500 ETF
qqq_open, qqq_close     -- NASDAQ 100 ETF
vix_close               -- VIX 지수
sector_etf_data         -- 섹터별 ETF
```

#### 종목 메타 정보
```sql
symbol              -- 티커
exchange            -- 거래소
sector              -- 섹터
industry            -- 산업
market_cap          -- 시가총액
```

---

## 4. Feature Engineering

### 4.1 가격 기반 Features
```python
features_price = {
    # 갭 관련
    "gap_pct": "(open - prev_close) / prev_close * 100",
    "gap_size_category": "small/medium/large (1-2%, 2-5%, 5%+)",
    
    # 전일 패턴
    "prev_return": "(prev_close - prev_open) / prev_open * 100",
    "prev_range_pct": "(prev_high - prev_low) / prev_close * 100",
    "prev_upper_shadow": "(prev_high - max(prev_open, prev_close)) / prev_close",
    "prev_lower_shadow": "(min(prev_open, prev_close) - prev_low) / prev_close",
    
    # 거래량
    "volume_ratio": "prev_volume / avg_volume_20d",
}

# ⭐ 고도화 단계에서 추가
features_price_advanced = {
    "premarket_volume_ratio": "premarket_volume / avg_volume_20d",
    "premarket_change": "(premarket_close - prev_close) / prev_close * 100",
    "premarket_range": "(premarket_high - premarket_low) / prev_close * 100",
}
```

### 4.2 기술적 지표 Features
```python
features_technical = {
    # 이동평균
    "above_ma5": "prev_close > MA_5",
    "above_ma20": "prev_close > MA_20",
    "above_ma50": "prev_close > MA_50",
    "ma5_ma20_cross": "MA_5 > MA_20",
    
    # 모멘텀
    "rsi_14": "RSI 14일",
    "rsi_category": "oversold(<30) / neutral / overbought(>70)",
    
    # 변동성
    "atr_14": "ATR 14일",
    "atr_ratio": "gap_size / ATR_14",
    "bollinger_position": "(prev_close - BB_lower) / (BB_upper - BB_lower)",
    
    # 추세
    "return_5d": "5일 수익률",
    "return_20d": "20일 수익률",
    "consecutive_up_days": "연속 상승일 수",
}
```

### 4.3 이벤트/뉴스 Features
```python
features_event = {
    # 실적
    "is_earnings_day": "당일 실적 발표 여부 (0/1)",
    "is_earnings_window": "실적 발표 ±2일 이내 (0/1)",
    "earnings_surprise_pct": "EPS 서프라이즈 (%)",
    "earnings_surprise_category": "beat / meet / miss",
    
    # 뉴스
    "news_count_24h": "24시간 내 뉴스 건수",
    "news_sentiment_avg": "평균 센티먼트 스코어",
    "news_sentiment_max": "최대 센티먼트 스코어",
    "has_positive_news": "긍정 뉴스 존재 여부",
    
    # 애널리스트
    "rating_change": "레이팅 변경 (upgrade:1, none:0, downgrade:-1)",
    "target_price_change_pct": "목표가 변경률",
}
```

### 4.4 시장 컨텍스트 Features
```python
features_market = {
    # 시장 방향
    "spy_gap_pct": "SPY 갭 (%)",
    "qqq_gap_pct": "QQQ 갭 (%)",
    "market_gap_diff": "종목 갭 - SPY 갭 (상대 강도)",
    
    # 변동성 환경
    "vix_level": "VIX 수준",
    "vix_category": "low(<15) / normal / high(>25)",
    "vix_change": "VIX 전일 대비 변화",
    
    # 섹터
    "sector_gap_pct": "섹터 ETF 갭 (%)",
    "sector_relative_gap": "종목 갭 - 섹터 갭",
}
```

### 4.5 시간 Features
```python
features_time = {
    "day_of_week": "요일 (0-4)",
    "month": "월 (1-12)",
    "is_month_start": "월초 여부 (첫 3거래일)",
    "is_month_end": "월말 여부 (마지막 3거래일)",
    "is_quarter_end": "분기말 여부",
    "days_since_earnings": "실적 발표 후 경과일",
}
```

---

## 5. 데이터 필터링 기준

### 5.1 종목 필터
```python
filters_symbol = {
    "min_price": 5.0,           # $5 이상 (페니스탁 제외)
    "min_market_cap": 500e6,    # $500M 이상
    "min_avg_volume": 100000,   # 일평균 거래량 10만주 이상
    "exchanges": ["NYSE", "NASDAQ", "AMEX"],
    "include_delisted": True,   # ✅ 상장폐지 종목 포함 (생존자 편향 제거)
}
```

### 5.2 갭 필터
```python
filters_gap = {
    # 데이터 수집 단계
    "collection_min_gap_pct": 0.0,   # 0% 초과 (시가 > 전일 종가) 모두 수집
    
    # 모델 학습 단계 (실험용)
    "training_gap_thresholds": [1.0, 2.0, 3.0, 5.0],  # 다양한 기준 실험
    
    # 공통
    "max_gap_pct": 50.0,        # 최대 50% 갭 (극단값 제외)
}
```

### 5.3 데이터 품질 필터
```python
filters_quality = {
    "exclude_first_trading_day": True,   # IPO 첫날 제외
    "exclude_halt_resume": True,         # 거래정지 후 재개일 제외
    "require_premarket_data": False,     # 프리마켓 데이터 불필요 (고도화에서 사용)
    "handle_delisted": "include",        # 상장폐지 종목 처리: 포함
}
```

### 5.4 상장폐지 종목 처리
```python
# 상장폐지 종목 포함 이유:
# - 생존자 편향(Survivorship Bias) 제거
# - 실패한 종목의 갭 패턴도 학습에 포함
# - 보다 현실적인 모델 성능 평가

# 주의사항:
# - 상장폐지 전 마지막 거래일 데이터 품질 확인
# - 거래정지 기간 처리 필요
```

---

## 6. 모델링 전략

### 6.1 문제 정의 - 하이브리드 접근법

#### 최종 출력 목표
```
입력: 갭 상승 종목 정보
출력:
  - 상승 확률: 72%
  - 상승 시 예상 수익률: +2.5%
  - 하락 시 예상 손실률: -1.8%
  - 기대 수익률: +1.3%
```

#### 3개 모델 구성
```
┌─────────────────────────────────────────────────────────────┐
│                    하이브리드 모델 구조                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  [입력: 갭 상승 종목 Features]                               │
│              │                                              │
│              ▼                                              │
│  ┌───────────────────────┐                                  │
│  │  Model 1: 분류 모델    │ → 상승 확률 P(up)                │
│  │  (Classification)     │                                  │
│  └───────────────────────┘                                  │
│              │                                              │
│              ▼                                              │
│  ┌───────────────────────┐                                  │
│  │  Model 2: 회귀 모델    │ → 상승 시 예상 수익률 E[r|up]    │
│  │  (상승 케이스만 학습)   │                                  │
│  └───────────────────────┘                                  │
│              │                                              │
│              ▼                                              │
│  ┌───────────────────────┐                                  │
│  │  Model 3: 회귀 모델    │ → 하락 시 예상 손실률 E[r|down]  │
│  │  (하락 케이스만 학습)   │                                  │
│  └───────────────────────┘                                  │
│              │                                              │
│              ▼                                              │
│  ┌───────────────────────┐                                  │
│  │      최종 계산         │                                  │
│  │  기대 수익률 =         │                                  │
│  │  P(up) × E[r|up] +    │                                  │
│  │  P(down) × E[r|down]  │                                  │
│  └───────────────────────┘                                  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

#### 모델별 상세

| 모델 | 타입 | 학습 데이터 | 타겟 | 출력 |
|------|------|-------------|------|------|
| Model 1 | 분류 | 전체 갭 상승 케이스 | 상승(1)/하락(0) | 상승 확률 (0~1) |
| Model 2 | 회귀 | 상승한 케이스만 | 수익률 (%) | 상승 시 예상 수익률 |
| Model 3 | 회귀 | 하락한 케이스만 | 수익률 (%) | 하락 시 예상 손실률 |

#### 타겟 변수 정의
```python
# 공통
intraday_return = (close - open) / open * 100  # 당일 수익률 (%)

# Model 1: 분류 타겟
target_direction = 1 if intraday_return > 0 else 0

# Model 2: 상승 케이스 수익률 (양수)
target_up_return = intraday_return  # where intraday_return > 0

# Model 3: 하락 케이스 수익률 (음수)
target_down_return = intraday_return  # where intraday_return <= 0
```

#### 최종 출력 계산
```python
def predict(features):
    # 각 모델 예측
    prob_up = model_classifier.predict_proba(features)[1]  # 상승 확률
    return_if_up = model_regressor_up.predict(features)    # 상승 시 수익률
    return_if_down = model_regressor_down.predict(features) # 하락 시 손실률
    
    # 기대 수익률 계산
    prob_down = 1 - prob_up
    expected_return = (prob_up * return_if_up) + (prob_down * return_if_down)
    
    return {
        "up_probability": prob_up,
        "expected_return_if_up": return_if_up,
        "expected_return_if_down": return_if_down,
        "expected_return": expected_return,
    }
```

### 6.2 데이터 분할
```python
# 시계열 특성 고려 - 미래 데이터 누출 방지
train_period = "2005-01-01 ~ 2019-12-31"  # 15년
valid_period = "2020-01-01 ~ 2021-12-31"  # 2년
test_period  = "2022-01-01 ~ 2024-12-31"  # 3년

# 또는 Rolling Window 방식
# 학습: 과거 5년, 검증: 다음 1년, 반복
```

### 6.3 모델 후보

#### Model 1: 분류 모델 (상승 확률)
```python
classifiers = {
    "baseline": "Logistic Regression",
    "tree_based": ["Random Forest", "XGBoost", "LightGBM", "CatBoost"],
    "neural": ["MLP", "TabNet"],  # 선택적
}
# 핵심: predict_proba()로 확률 출력 필요
# 캘리브레이션 필수 (Platt Scaling, Isotonic Regression)
```

#### Model 2 & 3: 회귀 모델 (수익률 예측)
```python
regressors = {
    "baseline": "Linear Regression, Ridge, Lasso",
    "tree_based": ["Random Forest Regressor", "XGBoost Regressor", "LightGBM Regressor"],
    "quantile": "Quantile Regression",  # 범위 예측 시 유용
}
# 핵심: 예측 신뢰구간 제공 가능하면 좋음
```

#### 고급 옵션: 단일 모델 접근
```python
# 대안 1: Multi-output 모델
# 하나의 모델이 [방향, 수익률] 동시 예측

# 대안 2: Quantile Regression
# 수익률의 10%, 50%, 90% 분위수 예측
# → 자연스럽게 범위와 방향 정보 제공
```

### 6.4 클래스 불균형 처리
```python
# 갭 상승 후 상승/하락 비율 확인 필요
# 예상: 약 50:50 또는 약간의 불균형

imbalance_strategies = [
    "class_weight='balanced'",
    "SMOTE",
    "Undersampling",
    "Threshold 조정",
]
```

---

## 7. 평가 지표

### 7.1 Model 1: 분류 성능
```python
metrics_classification = {
    "accuracy": "전체 정확도",
    "precision": "예측 상승 중 실제 상승 비율",
    "recall": "실제 상승 중 예측 상승 비율",
    "f1_score": "Precision-Recall 조화평균",
    "auc_roc": "ROC 곡선 아래 면적",
    "auc_pr": "Precision-Recall 곡선 아래 면적",
    "brier_score": "확률 예측 정확도 (낮을수록 좋음)",
    "calibration_error": "예측 확률 vs 실제 확률 오차",
}
```

### 7.2 Model 2 & 3: 회귀 성능
```python
metrics_regression = {
    "mae": "Mean Absolute Error (평균 절대 오차)",
    "rmse": "Root Mean Squared Error",
    "mape": "Mean Absolute Percentage Error",
    "r2": "결정 계수",
    "direction_accuracy": "예측 부호와 실제 부호 일치율",
}
```

### 7.3 통합 시스템 성능
```python
metrics_integrated = {
    "expected_return_mae": "기대 수익률 예측 MAE",
    "expected_return_correlation": "기대 수익률 vs 실제 수익률 상관계수",
    "ranking_quality": "기대 수익률 순위와 실제 순위 일치도 (Spearman)",
}
```

### 7.4 트레이딩 성능 (백테스팅)
```python
metrics_trading = {
    "win_rate": "승률 (%)",
    "avg_return": "평균 수익률 (%)",
    "profit_factor": "총이익 / 총손실",
    "sharpe_ratio": "샤프 비율",
    "max_drawdown": "최대 낙폭 (%)",
    "total_return": "누적 수익률 (%)",
}
```

### 7.3 신뢰도 기반 평가
```python
# 예측 확률(confidence)에 따른 성능 분석
confidence_analysis = {
    "high_confidence": "확률 > 0.7인 경우 승률",
    "low_confidence": "확률 0.5-0.6인 경우 승률",
    "calibration": "예측 확률과 실제 확률의 일치도",
}
```

---

## 8. 구현 단계

### Phase 1: 데이터 파이프라인 (Week 1-2)
```
1.1 DB 연결 및 스키마 확인
1.2 OHLCV 데이터 추출 쿼리 작성
1.3 실적 데이터 추출 및 매핑
1.4 뉴스/센티먼트 데이터 추출
1.5 데이터 병합 및 정합성 검증
1.6 갭 상승 이벤트 식별 및 라벨링
```

### Phase 2: EDA 및 Feature Engineering (Week 2-3)
```
2.1 갭 상승 기초 통계 분석
    - 연도별/월별 갭 발생 빈도
    - 갭 크기 분포
    - 갭 유형별 분포
    
2.2 타겟 분포 분석
    - 전체 갭 상승 후 승률
    - 갭 크기별 승률
    - 갭 유형별 승률
    
2.3 Feature 생성 및 선택
    - 상관관계 분석
    - Feature Importance 예비 분석
```

### Phase 3: 모델 개발 (Week 3-4)
```
3.1 Model 1: 분류 모델 (상승 확률)
    - 베이스라인: Logistic Regression
    - Tree 기반: XGBoost, LightGBM
    - 확률 캘리브레이션
    
3.2 Model 2: 상승 케이스 회귀 모델
    - 상승한 케이스만 필터링
    - 수익률 예측 모델 학습
    
3.3 Model 3: 하락 케이스 회귀 모델
    - 하락한 케이스만 필터링
    - 손실률 예측 모델 학습
    
3.4 통합 시스템 구축
    - 3개 모델 파이프라인 연결
    - 기대 수익률 계산 로직
    
3.5 하이퍼파라미터 튜닝
3.6 Feature Selection 반복
```

### Phase 4: 평가 및 백테스팅 (Week 4-5)
```
4.1 Hold-out 테스트셋 평가
4.2 시간별 성능 안정성 분석
4.3 백테스팅 시뮬레이션
4.4 리스크 분석
```

### Phase 5: 고도화 (선택)
```
5.1 프리마켓 데이터 추가 ⭐
    - 프리마켓 거래량, 변동률 Feature 추가
    - 모델 재학습 및 성능 비교
    
5.2 갭 유형별 개별 모델
    - Earnings Gap 전용 모델
    - News Gap 전용 모델
    
5.3 확률 캘리브레이션 고도화
5.4 실시간 예측 파이프라인
5.5 애널리스트 데이터 통합
```

---

## 9. 디렉토리 구조

```
gap_prediction/
├── config/
│   ├── config.yaml          # 설정 파일
│   └── features.yaml        # Feature 정의
│
├── data/
│   ├── raw/                 # DB에서 추출한 원본 데이터
│   ├── processed/           # 전처리된 데이터
│   └── features/            # Feature 데이터셋
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_modeling_classifier.ipynb    # Model 1: 분류
│   ├── 05_modeling_regressor.ipynb     # Model 2 & 3: 회귀
│   ├── 06_model_integration.ipynb      # 통합 시스템
│   └── 07_backtesting.ipynb
│
├── src/
│   ├── data/
│   │   ├── db_connector.py      # DB 연결
│   │   ├── data_loader.py       # 데이터 로드
│   │   └── preprocessor.py      # 전처리
│   │
│   ├── features/
│   │   ├── price_features.py    # 가격 기반 Feature
│   │   ├── technical_features.py # 기술적 지표
│   │   ├── event_features.py    # 이벤트/뉴스 Feature
│   │   └── market_features.py   # 시장 컨텍스트
│   │
│   ├── models/
│   │   ├── classifier.py        # Model 1: 방향 분류
│   │   ├── regressor_up.py      # Model 2: 상승 수익률
│   │   ├── regressor_down.py    # Model 3: 하락 손실률
│   │   ├── ensemble.py          # 통합 예측 시스템
│   │   ├── calibration.py       # 확률 캘리브레이션
│   │   └── trainer.py           # 공통 학습 유틸
│   │
│   ├── prediction/
│   │   ├── predictor.py         # 통합 예측기
│   │   └── output_formatter.py  # 출력 포맷팅
│   │
│   ├── evaluation/
│   │   ├── classifier_metrics.py
│   │   ├── regressor_metrics.py
│   │   └── integrated_metrics.py
│   │
│   └── backtesting/
│       ├── simulator.py         # 백테스팅 시뮬레이터
│       └── metrics.py           # 트레이딩 지표
│
├── models/                  # 학습된 모델 저장
│   ├── classifier/
│   ├── regressor_up/
│   └── regressor_down/
│
├── reports/                 # 분석 리포트
├── requirements.txt
└── README.md
```

---

## 10. 주요 리스크 및 고려사항

### 10.1 데이터 관련
- **Look-ahead Bias**: 미래 데이터가 Feature에 포함되지 않도록 주의
- **Survivorship Bias**: ✅ 상장폐지 종목 포함으로 해결
- **데이터 품질**: 수정주가, 주식분할 등 반영 확인
- **상장폐지 처리**: 마지막 거래일 데이터 품질, 거래정지 기간 확인

### 10.2 모델 관련
- **과적합**: 시계열 교차검증으로 검증
- **Regime Change**: 시장 환경 변화에 따른 성능 저하 가능
- **클래스 불균형**: 불균형 심할 경우 처리 필요

### 10.3 실전 적용
- **슬리피지**: 실제 진입가와 시가 차이
- **유동성**: 거래량 부족 시 진입 어려움
- **거래비용**: 수수료 반영한 실질 수익률

---

## 11. 성공 기준

### Model 1 (분류) 최소 목표
```
- 테스트셋 정확도: > 55%
- AUC-ROC: > 0.58
- Brier Score: < 0.24
- 캘리브레이션: 예측 확률과 실제 확률 오차 < 5%
```

### Model 2 & 3 (회귀) 최소 목표
```
- 수익률 예측 MAE: < 2.0%
- 방향 일치율: > 60%
- R²: > 0.05 (주가 예측은 매우 어려우므로 낮은 기준)
```

### 통합 시스템 목표
```
- 기대 수익률 상관계수: > 0.15
- 상위 20% 기대 수익률 종목 실제 평균 수익률: > 1.5%
- 하위 20% 기대 수익률 종목 실제 평균 수익률: < 0.5%
```

### 트레이딩 목표
```
- 기대 수익률 > 1% 종목만 매매 시 승률: > 60%
- 연간 샤프비율: > 1.0
- 백테스팅 누적 수익률: > Buy & Hold
```

---

## 12. 참고사항

### 변경 이력
| 날짜 | 버전 | 변경 내용 |
|------|------|-----------|
| 2025-01-06 | 0.1 | 초안 작성 |
| 2025-01-06 | 0.2 | 하이브리드 모델 구조로 변경 (분류 + 회귀) |
| 2025-01-06 | 0.3 | 갭 기준 0% 초과로 변경, 상장폐지 포함, 프리마켓 고도화로 이동 |

### 미결정 사항
- [x] ~~갭 상승 기준~~ → 0% 초과로 수집, 학습 시 실험
- [x] ~~상장폐지 종목 포함 여부~~ → 포함 (생존자 편향 제거)
- [x] ~~프리마켓 데이터 필수 여부~~ → Phase 1 제외, 고도화에서 추가
- [ ] 구체적인 DB 테이블/컬럼명

---

*이 문서는 프로젝트 진행에 따라 업데이트됩니다.*
