"""
월간 모델 재학습 시스템

Kafka consumer 기반 독립 컨테이너로 운영
- model_retrain_command 토픽 수신
- parquet(과거) + DB(신규) 데이터 병합
- 모델 학습 + Champion-Challenger 비교
- model_retrain_result 토픽 발행
"""
