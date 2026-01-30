"""
로컬 테스트 스크립트

DB에서 학습 데이터는 가져오지만, model_registry 메타데이터는 저장하지 않음
pkl 파일과 report.md가 생성되는지만 확인

Usage:
    python -m retraining.test_local
"""

import sys
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger(__name__)


def main():
    from pathlib import Path
    from retraining.schemas import ModelRetrainCommandMessage, RetrainConfig
    from retraining.handler import handle_retrain_command

    logger.info("=== Local Retraining Test (skip_registry=True) ===")

    # 커맨드 생성
    command = ModelRetrainCommandMessage(
        trigger_type="local_test",
        config=RetrainConfig(
            threshold=0.4,
            n_estimators=300,
            test_size=0.1,
            valid_size=0.2,
            test_recent=True,
        ),
    )

    # 실행 (DB registry 사용 → champion 비교 활성화)
    result = handle_retrain_command(command, skip_registry=False)

    # 결과 출력
    logger.info("=" * 60)
    logger.info(f"Status:   {result.status}")
    logger.info(f"Version:  {result.new_version}")
    logger.info(f"Deployed: {result.is_deployed}")
    logger.info(f"Duration: {result.training_duration_seconds:.1f}s")

    if result.error_message:
        logger.error(f"Error: {result.error_message}")
        return

    if result.comparison_result:
        logger.info(f"Decision: {result.comparison_result.decision}")
        logger.info(f"Reason:   {result.comparison_result.reason}")

    # 파일 확인
    version = result.new_version
    model_dir = Path(f"./models/stacking/{version}")
    pkl_path = model_dir / "stacking_hybrid_model.pkl"
    report_path = model_dir / "report.md"

    logger.info("")
    logger.info("=== Generated Files ===")
    logger.info(f"pkl:    {pkl_path} (exists={pkl_path.exists()}, "
                f"size={pkl_path.stat().st_size / 1024 / 1024:.1f}MB)" if pkl_path.exists() else f"pkl:    {pkl_path} (NOT FOUND)")
    logger.info(f"report: {report_path} (exists={report_path.exists()})" if report_path.exists() else f"report: {report_path} (NOT FOUND)")

    if report_path.exists():
        logger.info("")
        logger.info("=== Report Preview ===")
        print(report_path.read_text()[:2000])


if __name__ == "__main__":
    main()
