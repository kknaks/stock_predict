"""
재학습 오케스트레이터

전체 플로우 조율:
1. 데이터 로드 (parquet + DB)
2. 모델 학습
3. Champion-Challenger 비교
4. 배포 결정
5. 레지스트리 저장 (skip_registry=False일 때만)
6. Active symlink 업데이트
7. Markdown 리포트 생성
"""

import logging
import time
from datetime import datetime
from pathlib import Path

from .config import settings
from .database import get_session_factory
from .schemas import (
    ModelRetrainCommandMessage,
    ModelRetrainResultMessage,
)
from .data_pipeline import build_training_data
from .trainer_wrapper import train_versioned_model
from .champion_challenger import compare_models
from .report import generate_report
from .backtester import run_backtest

logger = logging.getLogger(__name__)


def _generate_version() -> str:
    """버전 생성: v1.{YY}.{MMDD}"""
    now = datetime.now()
    return f"v1.{now.strftime('%y')}.{now.strftime('%m%d')}"


def _update_active_symlink(version: str):
    """active symlink 업데이트"""
    model_dir = Path(settings.model_base_dir)
    version_dir = model_dir / version
    symlink_path = model_dir / settings.active_symlink_name

    if symlink_path.is_symlink() or symlink_path.exists():
        symlink_path.unlink()

    symlink_path.symlink_to(version_dir.resolve())
    logger.info(f"Active symlink updated: {symlink_path} -> {version_dir}")


def handle_retrain_command(
    command: ModelRetrainCommandMessage,
    skip_registry: bool = False,
) -> ModelRetrainResultMessage:
    """
    재학습 커맨드 처리 (오케스트레이터)

    Args:
        command: 재학습 커맨드 메시지
        skip_registry: True면 model_registry DB 저장 스킵
                       (로컬 테스트: pkl + report.md 생성만 확인)

    Returns:
        재학습 결과 메시지
    """
    start_time = time.time()
    version = _generate_version()
    config = command.config

    logger.info(
        f"Starting retraining: version={version}, trigger={command.trigger_type}, "
        f"skip_registry={skip_registry}"
    )

    try:
        session_factory = get_session_factory()

        with session_factory() as session:
            # ========================================
            # 1. 데이터 로드 (parquet + DB)
            # ========================================
            df_model, total_samples, merged_raw_df = build_training_data(
                session=session,
                parquet_path=settings.parquet_path,
                db_start_date=config.data_start_date,
                db_end_date=config.data_end_date,
            )

            if df_model.empty:
                return ModelRetrainResultMessage(
                    status="failed",
                    new_version=version,
                    trigger_type=command.trigger_type,
                    error_message="No training data available",
                    training_duration_seconds=time.time() - start_time,
                )

            # ========================================
            # 2. 모델 학습
            # ========================================
            train_result = train_versioned_model(
                df_model=df_model,
                version=version,
                model_base_dir=settings.model_base_dir,
                threshold=config.threshold,
                n_estimators=config.n_estimators,
                test_size=config.test_size,
                valid_size=config.valid_size,
                test_recent=config.test_recent,
            )

            test_metrics = train_result["test_metrics"]
            model_path = train_result["model_path"]

            # 학습 데이터 parquet을 버전 디렉토리에 저장 (원본 보존)
            version_dir = Path(settings.model_base_dir) / version
            version_dir.mkdir(parents=True, exist_ok=True)
            training_data_path = version_dir / "training_data.parquet"
            df_model.to_parquet(training_data_path, index=False)
            logger.info(f"Training data saved: {training_data_path}")

            hyperparams = {
                "threshold": config.threshold,
                "n_estimators": config.n_estimators,
                "test_size": config.test_size,
                "valid_size": config.valid_size,
            }

            data_dates = _get_data_date_range(df_model)

            # ========================================
            # 2.5 백테스트 실행
            # ========================================
            backtest_result = None
            try:
                backtest_result = run_backtest(
                    model_path=model_path,
                    df_model=df_model,
                    df_raw=merged_raw_df,
                    threshold=config.threshold,
                )
                if backtest_result:
                    logger.info(
                        f"Backtest: {backtest_result['metrics'].get('n_trades', 0)} trades, "
                        f"return={backtest_result['metrics'].get('total_return_pct', 0):+.2f}%"
                    )
            except Exception as e:
                logger.warning(f"Backtest failed: {e}")

            # ========================================
            # 3. Champion-Challenger 비교
            # ========================================
            # skip_registry면 champion 없이 비교 (첫 모델 취급)
            champion_metrics = None
            champion_version = None
            active_model = None

            if not skip_registry:
                from . import model_registry_service as registry
                from app.database.database.model_registry import ModelStatus

                # champion 정보를 먼저 조회 (create_entry가 같은 버전을 삭제하기 전에)
                active_model = registry.get_active_model(session)
                if active_model and active_model.test_metrics and active_model.version != version:
                    champion_metrics = active_model.test_metrics.get("classifier")
                    champion_version = active_model.version

                # 레지스트리에 candidate로 등록 (같은 버전이면 삭제 후 재등록)
                registry.create_entry(
                    session=session,
                    version=version,
                    model_path=model_path,
                    training_data_start=data_dates[0],
                    training_data_end=data_dates[1],
                    training_samples=total_samples,
                    training_duration_seconds=time.time() - start_time,
                    trigger_type=command.trigger_type,
                    hyperparameters=hyperparams,
                    test_metrics=test_metrics,
                    status=ModelStatus.CANDIDATE,
                )

            comparison = compare_models(
                champion_metrics=champion_metrics,
                challenger_metrics=test_metrics.get("classifier", {}),
                champion_version=champion_version,
                challenger_version=version,
                min_roc_auc=config.min_roc_auc,
                max_roc_auc_drop=config.max_roc_auc_drop,
                max_f1_drop=config.max_f1_drop,
                backtest_metrics=backtest_result["metrics"] if backtest_result else None,
                min_backtest_return=config.min_backtest_return,
                min_sharpe_ratio=config.min_sharpe_ratio,
                min_profit_factor=config.min_profit_factor,
            )

            logger.info(f"Comparison: decision={comparison.decision}, reason={comparison.reason}")

            # ========================================
            # 4. 배포 결정
            # ========================================
            is_deployed = False
            if comparison.decision == "deploy":
                if not skip_registry:
                    registry.update_comparison_result(
                        session=session,
                        version=version,
                        comparison_result=comparison.model_dump(),
                    )
                    registry.activate_model(session, version)
                    session.commit()

                _update_active_symlink(version)
                is_deployed = True
                logger.info(f"Model deployed: {version}")
            else:
                if not skip_registry:
                    registry.update_comparison_result(
                        session=session,
                        version=version,
                        comparison_result=comparison.model_dump(),
                    )
                    registry.reject_model(session, version)
                    session.commit()

                logger.info(f"Model rejected: {version}")

            # ========================================
            # 5. Markdown 리포트 + 비교 차트 생성
            # ========================================
            duration = time.time() - start_time

            # champion 메트릭 (차트 비교용)
            champion_test_metrics = None
            if not skip_registry and active_model and active_model.test_metrics:
                champion_test_metrics = active_model.test_metrics

            try:
                report_path = generate_report(
                    version=version,
                    trigger_type=command.trigger_type,
                    training_duration_seconds=duration,
                    training_samples=total_samples,
                    data_date_range=data_dates,
                    test_metrics=test_metrics,
                    comparison=comparison,
                    is_deployed=is_deployed,
                    model_path=model_path,
                    hyperparameters=hyperparams,
                    model_base_dir=settings.model_base_dir,
                    champion_test_metrics=champion_test_metrics,
                    backtest_result=backtest_result,
                )
                logger.info(f"Report generated: {report_path}")
            except Exception as e:
                logger.warning(f"Report generation failed: {e}")

            return ModelRetrainResultMessage(
                status="success" if is_deployed else "rejected",
                new_version=version,
                is_deployed=is_deployed,
                test_metrics=test_metrics,
                comparison_result=comparison,
                trigger_type=command.trigger_type,
                training_duration_seconds=duration,
            )

    except Exception as e:
        logger.error(f"Retraining failed: {e}", exc_info=True)
        return ModelRetrainResultMessage(
            status="failed",
            new_version=version,
            trigger_type=command.trigger_type,
            error_message=str(e),
            training_duration_seconds=time.time() - start_time,
        )


def _get_data_date_range(df):
    """DataFrame에서 날짜 범위 추출"""
    if "date" not in df.columns or df.empty:
        return (None, None)
    dates = df["date"]
    return (str(dates.min())[:10], str(dates.max())[:10])
