"""
PostgreSQL 데이터베이스 연결

동기/비동기 연결 모두 지원:
- 비동기: asyncpg + SQLAlchemy async (FastAPI, 실시간 처리)
- 동기: psycopg2 + SQLAlchemy (배치 처리, Feature 계산)
"""

import logging
from typing import AsyncGenerator, Generator

from sqlalchemy import create_engine, Engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
    AsyncEngine,
)

from .settings import settings

logger = logging.getLogger(__name__)

# 싱글톤 인스턴스
_async_engine: AsyncEngine | None = None
_sync_engine: Engine | None = None
_async_session_factory: async_sessionmaker[AsyncSession] | None = None
_sync_session_factory: sessionmaker[Session] | None = None


# ========================================
# 비동기 연결 (AsyncPG)
# ========================================

def get_async_engine() -> AsyncEngine:
    """비동기 엔진 반환 (싱글톤)"""
    global _async_engine

    if _async_engine is None:
        _async_engine = create_async_engine(
            settings.async_database_url,
            echo=settings.db_echo,
            pool_size=5,
            max_overflow=10,
            pool_pre_ping=True,
        )
        logger.info(f"✓ Async DB engine created: {settings.async_database_url}")

    return _async_engine


def get_async_session_factory() -> async_sessionmaker[AsyncSession]:
    """비동기 세션 팩토리 반환"""
    global _async_session_factory

    if _async_session_factory is None:
        _async_session_factory = async_sessionmaker(
            bind=get_async_engine(),
            class_=AsyncSession,
            expire_on_commit=False,
            autocommit=False,
            autoflush=False,
        )

    return _async_session_factory


async def get_async_db() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI Dependency용 비동기 세션 제공

    Usage:
        @app.get("/users")
        async def get_users(db: AsyncSession = Depends(get_async_db)):
            ...
    """
    session_factory = get_async_session_factory()
    session = session_factory()

    try:
        yield session
        await session.commit()
    except Exception as e:
        await session.rollback()
        logger.error(f"Async DB session error: {e}")
        raise
    finally:
        await session.close()


async def init_async_db():
    """비동기 데이터베이스 연결 테스트"""
    from sqlalchemy import text

    engine = get_async_engine()

    async with engine.begin() as conn:
        await conn.execute(text("SELECT 1"))

    logger.info("✓ Async database connection verified")


async def close_async_db():
    """비동기 데이터베이스 연결 종료"""
    global _async_engine, _async_session_factory

    if _async_engine:
        await _async_engine.dispose()
        _async_engine = None
        _async_session_factory = None
        logger.info("Async database connection closed")


# ========================================
# 동기 연결 (Psycopg2) - Feature 계산용
# ========================================

def get_sync_engine() -> Engine:
    """동기 엔진 반환 (싱글톤)"""
    global _sync_engine

    if _sync_engine is None:
        _sync_engine = create_engine(
            settings.database_url,
            echo=settings.db_echo,
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True,
        )
        logger.info(f"✓ Sync DB engine created: {settings.database_url}")

    return _sync_engine


def get_sync_session_factory() -> sessionmaker[Session]:
    """동기 세션 팩토리 반환"""
    global _sync_session_factory

    if _sync_session_factory is None:
        _sync_session_factory = sessionmaker(
            bind=get_sync_engine(),
            autocommit=False,
            autoflush=False,
        )

    return _sync_session_factory


def get_sync_db() -> Generator[Session, None, None]:
    """
    동기 세션 제공 (Context Manager 사용)

    Usage:
        with next(get_sync_db()) as session:
            result = session.execute(query)
    """
    session_factory = get_sync_session_factory()
    session = session_factory()

    try:
        yield session
        session.commit()
    except Exception as e:
        session.rollback()
        logger.error(f"Sync DB session error: {e}")
        raise
    finally:
        session.close()


def init_sync_db():
    """동기 데이터베이스 연결 테스트"""
    from sqlalchemy import text

    engine = get_sync_engine()

    with engine.begin() as conn:
        conn.execute(text("SELECT 1"))

    logger.info("✓ Sync database connection verified")


def close_sync_db():
    """동기 데이터베이스 연결 종료"""
    global _sync_engine, _sync_session_factory

    if _sync_engine:
        _sync_engine.dispose()
        _sync_engine = None
        _sync_session_factory = None
        logger.info("Sync database connection closed")


# ========================================
# 통합 함수
# ========================================

async def init_db():
    """모든 데이터베이스 연결 초기화"""
    await init_async_db()
    init_sync_db()
    logger.info("✓ All database connections initialized")


async def close_db():
    """모든 데이터베이스 연결 종료"""
    await close_async_db()
    close_sync_db()
    logger.info("All database connections closed")
