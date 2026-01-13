"""
Model 2-1 (고가 예측) 디버깅 스크립트
"""
import pandas as pd
import numpy as np
from pathlib import Path

# 데이터 로드
data_path = Path(__file__).parent.parent / "data/processed/preprocessed_df_full.parquet"
df = pd.read_parquet(data_path)

print("=" * 80)
print("Model 2-1 디버깅: 고가 수익률 예측 문제 분석")
print("=" * 80)

# 1. target_max_return 컬럼 확인
print("\n1. target_max_return 컬럼 기본 정보")
print(f"   컬럼 존재 여부: {'target_max_return' in df.columns}")

if 'target_max_return' in df.columns:
    print(f"\n   통계:")
    print(df['target_max_return'].describe())

    print(f"\n   NaN 개수: {df['target_max_return'].isna().sum():,} ({df['target_max_return'].isna().sum()/len(df)*100:.2f}%)")

    # 2. target_return vs target_max_return 비교
    print("\n2. target_return vs target_max_return 비교 (상승 케이스만)")
    df_up = df[df['target_direction'] == 1].copy()

    print(f"\n   상승 케이스 개수: {len(df_up):,}")
    print(f"\n   target_return (종가) 통계:")
    print(df_up['target_return'].describe())

    print(f"\n   target_max_return (고가) 통계:")
    print(df_up['target_max_return'].describe())

    # 차이 계산
    df_up['diff'] = df_up['target_max_return'] - df_up['target_return']
    print(f"\n   차이 (고가 - 종가) 통계:")
    print(df_up['diff'].describe())

    print(f"\n   평균:")
    print(f"   - 종가 수익률: {df_up['target_return'].mean():.2f}%")
    print(f"   - 고가 수익률: {df_up['target_max_return'].mean():.2f}%")
    print(f"   - 차이: {df_up['diff'].mean():.2f}%p")

    # 3. 이상값 확인
    print("\n3. 이상값 확인")
    print(f"   고가 < 종가인 케이스: {(df_up['target_max_return'] < df_up['target_return']).sum():,}")
    print(f"   고가 = 종가인 케이스: {(df_up['target_max_return'] == df_up['target_return']).sum():,}")

    # 이상한 케이스 샘플
    weird = df_up[df_up['target_max_return'] < df_up['target_return']].head(10)
    if len(weird) > 0:
        print(f"\n   고가 < 종가 샘플:")
        print(weird[['date', 'InfoCode', 'target_return', 'target_max_return', 'diff']].to_string())

    # 4. 원본 데이터 확인 (high, close, open)
    print("\n4. 원본 OHLC 데이터 확인 (상승 케이스 샘플 10개)")
    sample = df_up.sample(min(10, len(df_up)), random_state=42)
    print(sample[['date', 'InfoCode', 'open', 'high', 'close', 'target_return', 'target_max_return']].to_string())

    # 5. target_max_return 계산 검증
    print("\n5. target_max_return 계산 검증")
    print("   예상 계산식: (high - open) / open * 100")

    sample_verify = df_up.sample(min(5, len(df_up)), random_state=42).copy()
    sample_verify['calculated_max_return'] = (sample_verify['high'] - sample_verify['open']) / sample_verify['open'] * 100
    sample_verify['match'] = np.isclose(sample_verify['target_max_return'], sample_verify['calculated_max_return'], rtol=0.01)

    print(sample_verify[['date', 'InfoCode', 'open', 'high', 'target_max_return', 'calculated_max_return', 'match']].to_string())

    print(f"\n   계산식 일치 비율: {sample_verify['match'].sum()}/{len(sample_verify)}")

    # 6. 분할 문제 확인
    print("\n6. 데이터 분할 시뮬레이션")
    from sklearn.model_selection import train_test_split

    # 전체 데이터에서 NaN 제거
    df_clean = df.dropna(subset=['target_direction', 'target_return', 'target_max_return'])

    # 상승 케이스
    df_up_clean = df_clean[df_clean['target_direction'] == 1].copy()

    print(f"\n   전체 데이터: {len(df_clean):,}")
    print(f"   상승 케이스: {len(df_up_clean):,}")

    # 분할 시뮬레이션
    X = df_up_clean[['gap_pct', 'prev_return', 'market_gap_diff']]  # 임시 features
    y_return = df_up_clean['target_return']
    y_max_return = df_up_clean['target_max_return']

    # 원본 노트북과 동일하게 분할
    X_temp, X_test, y_ret_temp, y_ret_test = train_test_split(
        X, y_return, test_size=0.1, random_state=42
    )

    # 고가 수익률도 동일한 인덱스로 분할
    y_max_ret_temp = y_max_return.loc[X_temp.index]
    y_max_ret_test = y_max_return.loc[X_test.index]

    print(f"\n   Test Set 크기: {len(X_test):,}")
    print(f"   종가 수익률 평균: {y_ret_test.mean():.2f}%")
    print(f"   고가 수익률 평균: {y_max_ret_test.mean():.2f}%")
    print(f"   차이: {y_max_ret_test.mean() - y_ret_test.mean():.2f}%p")

    # 노트북 코드의 문제 재현
    print("\n7. 노트북 코드 문제 재현 (잘못된 분할)")
    print("   원본 코드:")
    print("   _, _, _, _, _, y_up_max_test = train_test_split(")
    print("       X_up, y_up_max, y_direction[df_up.index], test_size=0.1, random_state=42, stratify=y_direction[df_up.index]")
    print("   )")

    print(f"\n   ⚠️ 문제: stratify=y_direction[df_up.index]")
    print(f"   df_up은 이미 target_direction==1만 필터링됨")
    print(f"   따라서 y_direction[df_up.index]는 모두 1")
    print(f"   stratify에 모두 같은 값 → 에러 또는 이상한 분할")

    print("\n   ⚠️ 또 다른 문제: X_up_test와 y_up_max_test의 인덱스 불일치")
    print(f"   X_up_test는 첫 번째 split으로 생성")
    print(f"   y_up_max_test는 두 번째 독립된 split으로 생성")
    print(f"   → 같은 random_state여도 stratify가 다르면 다른 결과!")

else:
    print("   ⚠️ target_max_return 컬럼이 없습니다!")

print("\n" + "=" * 80)
print("디버깅 완료")
print("=" * 80)
