import yaml
from pathlib import Path
from typing import Optional, Dict, Any, Type, List
import logging
from urllib.parse import quote_plus
from sqlalchemy import create_engine, Engine, text
from sqlalchemy.orm import sessionmaker, Session, DeclarativeBase
from sqlalchemy.pool import QueuePool

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    """SQLAlchemy ORM 모델의 기본 클래스"""
    pass


class DatabaseConnector:
    """MSSQL 데이터베이스 ORM 연결 관리 클래스"""

    def __init__(self, config_path: Optional[str] = None):
        """
        Args:
            config_path: config.yaml 파일 경로. None이면 프로젝트 루트의 config/config.yaml 사용
        """
        if config_path is None:
            project_root = Path(__file__).parent.parent.parent
            config_path = project_root / "config" / "config.yaml"

        self.config = self._load_config(config_path)
        self.engine: Optional[Engine] = None
        self.SessionLocal: Optional[sessionmaker] = None

    def _load_config(self, config_path: Path) -> Dict[str, Any]:
        """config.yaml 파일에서 데이터베이스 설정 로드"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            return config['database']
        except FileNotFoundError:
            logger.error(f"Config file not found: {config_path}")
            raise
        except KeyError:
            logger.error("'database' key not found in config file")
            raise
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            raise

    def _create_connection_string(self) -> str:
        """SQLAlchemy 연결 문자열 생성"""
        server = self.config['server']
        database = self.config['database']
        username = self.config['username']
        password = self.config['password']

        # 서버 주소에서 콤마를 콜론으로 변경 (MSSQL 형식 -> URL 형식)
        # 예: "4.230.9.13,1433" -> "4.230.9.13:1433"
        server = server.replace(',', ':')
        
        # 비밀번호 특수문자 URL 인코딩
        password_encoded = quote_plus(password)

        # pymssql 드라이버 사용
        connection_string = (
            f"mssql+pymssql://{username}:{password_encoded}@{server}/{database}"
        )
        return connection_string

    def connect(self) -> Engine:
        """데이터베이스 엔진 생성 및 연결"""
        try:
            connection_string = self._create_connection_string()

            self.engine = create_engine(
                connection_string,
                poolclass=QueuePool,
                pool_size=5,
                max_overflow=10,
                pool_pre_ping=True,  # 연결 유효성 체크
                echo=False  # SQL 로깅 (디버깅 시 True로 변경)
            )

            # SessionLocal 생성
            self.SessionLocal = sessionmaker(
                autocommit=False,
                autoflush=False,
                bind=self.engine
            )

            # 연결 테스트
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))

            logger.info(f"Successfully connected to database: {self.config['database']}")
            return self.engine

        except Exception as e:
            logger.error(f"Database connection error: {e}")
            raise

    def get_session(self) -> Session:
        """
        새로운 데이터베이스 세션 생성

        Returns:
            SQLAlchemy Session 객체
        """
        if not self.SessionLocal:
            self.connect()

        return self.SessionLocal()

    def create_tables(self, base: Type[DeclarativeBase] = Base):
        """
        정의된 모든 ORM 모델의 테이블 생성

        Args:
            base: DeclarativeBase 클래스 (기본값: Base)
        """
        if not self.engine:
            self.connect()

        base.metadata.create_all(bind=self.engine)
        logger.info("All tables created successfully")

    def drop_tables(self, base: Type[DeclarativeBase] = Base):
        """
        정의된 모든 ORM 모델의 테이블 삭제

        Args:
            base: DeclarativeBase 클래스 (기본값: Base)
        """
        if not self.engine:
            self.connect()

        base.metadata.drop_all(bind=self.engine)
        logger.info("All tables dropped successfully")

    def execute_raw_sql(self, query: str, params: Optional[Dict[str, Any]] = None):
        """
        Raw SQL 쿼리 실행 (ORM 외부에서 직접 SQL 실행 필요 시)

        Args:
            query: 실행할 SQL 쿼리
            params: 쿼리 파라미터 (선택사항)

        Returns:
            쿼리 결과
        """
        if not self.engine:
            self.connect()

        with self.engine.connect() as conn:
            if params:
                result = conn.execute(text(query), params)
            else:
                result = conn.execute(text(query))
            conn.commit()
            return result

    def get_table_list(self) -> List[str]:
        """데이터베이스의 모든 테이블 목록 조회"""
        query = """
        SELECT TABLE_NAME
        FROM INFORMATION_SCHEMA.TABLES
        WHERE TABLE_TYPE = 'BASE TABLE'
        ORDER BY TABLE_NAME
        """
        result = self.execute_raw_sql(query)
        return [row[0] for row in result]

    def dispose(self):
        """데이터베이스 엔진 및 연결 풀 종료"""
        if self.engine:
            self.engine.dispose()
            logger.info("Database connection pool disposed")
            self.engine = None
            self.SessionLocal = None

    def __enter__(self):
        """Context manager 진입"""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager 종료"""
        self.dispose()
        return False


class SessionManager:
    """세션 관리를 위한 Context Manager"""

    def __init__(self, db_connector: DatabaseConnector):
        """
        Args:
            db_connector: DatabaseConnector 인스턴스
        """
        self.db_connector = db_connector
        self.session: Optional[Session] = None

    def __enter__(self) -> Session:
        """Context manager 진입 - 세션 생성"""
        self.session = self.db_connector.get_session()
        return self.session

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager 종료 - 세션 정리"""
        if self.session:
            if exc_type is not None:
                # 예외 발생 시 롤백
                self.session.rollback()
                logger.error(f"Session rolled back due to error: {exc_val}")
            else:
                # 정상 종료 시 커밋
                self.session.commit()
            self.session.close()
        return False


# 전역 데이터베이스 커넥터 인스턴스 (선택적)
_db_connector: Optional[DatabaseConnector] = None


def get_db_connector() -> DatabaseConnector:
    """전역 데이터베이스 커넥터 인스턴스 반환 (싱글톤 패턴)"""
    global _db_connector
    if _db_connector is None:
        _db_connector = DatabaseConnector()
        _db_connector.connect()
    return _db_connector


def get_session() -> Session:
    """새로운 데이터베이스 세션 생성 (의존성 주입용)"""
    db = get_db_connector()
    return db.get_session()


# 사용 예시
if __name__ == "__main__":
    # 방법 1: Context manager 사용
    with DatabaseConnector() as db:
        # 테이블 목록 조회
        tables = db.get_table_list()
        print(f"Available tables: {tables}")

        # 세션을 사용한 작업
        with SessionManager(db) as session:
            # ORM 쿼리 예시 (모델이 정의된 경우)
            # results = session.query(YourModel).filter_by(id=1).all()
            pass

    # 방법 2: 전역 인스턴스 사용
    db = get_db_connector()
    session = get_session()
    try:
        # 작업 수행
        # results = session.query(YourModel).all()
        session.commit()
    except Exception as e:
        session.rollback()
        raise
    finally:
        session.close()
