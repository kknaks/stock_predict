"""
Champion-Challenger 모델 비교

신규 모델(Challenger)이 기존 모델(Champion)보다 나은지 평가
"""

import logging
from typing import Optional, Dict, Any

from .schemas import ComparisonResult

logger = logging.getLogger(__name__)


def compare_models(
    champion_metrics: Optional[Dict[str, Any]],
    challenger_metrics: Dict[str, Any],
    champion_version: Optional[str],
    challenger_version: str,
    min_roc_auc: float = 0.55,
    max_roc_auc_drop: float = 0.02,
    max_f1_drop: float = 0.05,
    backtest_metrics: Optional[Dict[str, Any]] = None,
    min_backtest_return: float = 0.0,
    min_sharpe_ratio: float = 1.0,
    min_profit_factor: float = 1.0,
) -> ComparisonResult:
    """
    Champion-Challenger 비교 수행

    Rules:
    1. 챔피언이 없으면 (첫 모델) → auto-deploy (ROC AUC >= min_roc_auc 시)
    2. ROC AUC < min_roc_auc → reject
    3. 백테스트 수익률 < min_backtest_return → reject
    4. 백테스트 Sharpe < min_sharpe_ratio → reject
    5. 백테스트 Profit Factor < min_profit_factor → reject
    6. ROC AUC 하락 > max_roc_auc_drop → reject
    7. F1 하락 > max_f1_drop AND ROC AUC 개선 없음 → reject
    8. 그 외 → deploy
    """
    challenger_roc_auc = challenger_metrics.get("roc_auc", 0.0)
    challenger_f1 = challenger_metrics.get("f1_score", 0.0)

    def _result(decision: str, reason: str, champ_roc=None, champ_f1=None) -> ComparisonResult:
        return ComparisonResult(
            champion_version=champion_version,
            challenger_version=challenger_version,
            champion_roc_auc=champ_roc,
            challenger_roc_auc=challenger_roc_auc,
            champion_f1=champ_f1,
            challenger_f1=challenger_f1,
            decision=decision,
            reason=reason,
        )

    # Rule 1: 첫 모델 (챔피언 없음)
    if champion_metrics is None:
        if challenger_roc_auc < min_roc_auc:
            return _result("reject", f"First model but ROC AUC {challenger_roc_auc:.4f} < {min_roc_auc}")

        # 백테스트 기준 체크
        reject_reason = _check_backtest(backtest_metrics, min_backtest_return, min_sharpe_ratio, min_profit_factor)
        if reject_reason:
            return _result("reject", reject_reason)

        return _result("deploy", f"First model, ROC AUC {challenger_roc_auc:.4f} >= {min_roc_auc}")

    champion_roc_auc = champion_metrics.get("roc_auc", 0.0)
    champion_f1 = champion_metrics.get("f1_score", 0.0)

    # Rule 2: ROC AUC 절대 최소값
    if challenger_roc_auc < min_roc_auc:
        return _result(
            "reject",
            f"ROC AUC {challenger_roc_auc:.4f} below minimum {min_roc_auc}",
            champion_roc_auc, champion_f1,
        )

    # Rule 3-5: 백테스트 기준
    reject_reason = _check_backtest(backtest_metrics, min_backtest_return, min_sharpe_ratio, min_profit_factor)
    if reject_reason:
        return _result("reject", reject_reason, champion_roc_auc, champion_f1)

    # Rule 6: ROC AUC 하락폭
    roc_drop = champion_roc_auc - challenger_roc_auc
    if roc_drop > max_roc_auc_drop:
        return _result(
            "reject",
            f"ROC AUC dropped {roc_drop:.4f} ({champion_roc_auc:.4f} -> {challenger_roc_auc:.4f}), exceeds max drop {max_roc_auc_drop}",
            champion_roc_auc, champion_f1,
        )

    # Rule 7: F1 하락 + ROC 개선 없음
    f1_drop = champion_f1 - challenger_f1
    roc_improved = challenger_roc_auc > champion_roc_auc
    if f1_drop > max_f1_drop and not roc_improved:
        return _result(
            "reject",
            f"F1 dropped {f1_drop:.4f} ({champion_f1:.4f} -> {challenger_f1:.4f}) without ROC AUC improvement",
            champion_roc_auc, champion_f1,
        )

    # 통과
    return _result(
        "deploy",
        f"Challenger passed: ROC AUC {champion_roc_auc:.4f} -> {challenger_roc_auc:.4f}, F1 {champion_f1:.4f} -> {challenger_f1:.4f}",
        champion_roc_auc, champion_f1,
    )


def _check_backtest(
    backtest_metrics: Optional[Dict[str, Any]],
    min_return: float,
    min_sharpe: float,
    min_profit_factor: float,
) -> Optional[str]:
    """백테스트 기준 체크. 통과하면 None, 실패하면 reject 사유 반환."""
    if backtest_metrics is None:
        return None  # 백테스트 결과 없으면 스킵

    total_return = backtest_metrics.get("total_return_pct", 0.0)
    sharpe = backtest_metrics.get("sharpe_ratio", 0.0)
    profit_factor = backtest_metrics.get("profit_factor", 0.0)

    if total_return < min_return:
        return f"Backtest return {total_return:+.2f}% below minimum {min_return}%"

    if sharpe < min_sharpe:
        return f"Backtest Sharpe {sharpe:.3f} below minimum {min_sharpe}"

    if profit_factor < min_profit_factor:
        return f"Backtest Profit Factor {profit_factor:.3f} below minimum {min_profit_factor}"

    return None
