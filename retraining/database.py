"""
재학습 서버 DB 연결 (app 미의존)

동기 연결만 사용 (배치 학습용)
"""

import logging
from typing import Generator

from sqlalchemy import create_engine, Engine
from sqlalchemy.orm import sessionmaker, Session

from .config import settings

logger = logging.getLogger(__name__)

_engine: Engine | None = None
_session_factory: sessionmaker[Session] | None = None


def get_engine() -> Engine:
    """동기 엔진 반환 (싱글톤)"""
    global _engine

    if _engine is None:
        _engine = create_engine(
            settings.database_url,
            echo=settings.db_echo,
            pool_size=5,
            max_overflow=10,
            pool_pre_ping=True,
        )
        logger.info(f"Retraining DB engine created: {settings.database_url}")

    return _engine


def get_session_factory() -> sessionmaker[Session]:
    """세션 팩토리 반환"""
    global _session_factory

    if _session_factory is None:
        _session_factory = sessionmaker(
            bind=get_engine(),
            autocommit=False,
            autoflush=False,
        )

    return _session_factory


def get_session() -> Generator[Session, None, None]:
    """세션 제공 (context manager)"""
    factory = get_session_factory()
    session = factory()
    try:
        yield session
        session.commit()
    except Exception as e:
        session.rollback()
        logger.error(f"DB session error: {e}")
        raise
    finally:
        session.close()


def close_db():
    """DB 연결 종료"""
    global _engine, _session_factory
    if _engine:
        _engine.dispose()
        _engine = None
        _session_factory = None
        logger.info("Retraining DB connection closed")
