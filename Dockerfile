FROM python:3.11-slim

# 작업 디렉토리 설정
WORKDIR /app

# 시스템 패키지 설치
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Poetry 설치
RUN pip install --no-cache-dir poetry==1.8.3

# Poetry 설정 (가상환경 비활성화)
RUN poetry config virtualenvs.create false

# 의존성 파일 복사
COPY pyproject.toml poetry.lock* ./

# Lock 파일이 없거나 호환되지 않으면 재생성
RUN poetry lock --no-update || poetry lock

# 의존성만 설치 (패키지 자체는 설치하지 않음, --no-root 옵션)
RUN poetry install --only main --no-root --no-interaction --no-ansi

# 애플리케이션 코드 복사
COPY app/ ./app/
COPY src/ ./src/
COPY config/ ./config/
# models는 볼륨 마운트로 사용 (docker-compose.yml에서)

# 환경 변수 설정
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# 메인 엔트리 포인트
CMD ["python", "-m", "app.main"]
