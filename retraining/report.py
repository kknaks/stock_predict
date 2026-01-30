"""
재학습 결과 Markdown 리포트 생성

models/stacking/{version}/report.md + 비교 차트 PNG
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from .schemas import ComparisonResult

logger = logging.getLogger(__name__)


# ============================================================
# 차트 생성
# ============================================================

def _generate_classifier_comparison_chart(
    champion_metrics: Optional[Dict[str, Any]],
    challenger_metrics: Dict[str, Any],
    champion_version: Optional[str],
    challenger_version: str,
    save_path: Path,
):
    """Classifier 메트릭 비교 바 차트"""
    metric_keys = ["accuracy", "roc_auc", "f1_score", "precision", "recall"]
    labels = ["Accuracy", "ROC AUC", "F1", "Precision", "Recall"]

    challenger_vals = [challenger_metrics.get(k, 0) for k in metric_keys]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(labels))
    width = 0.35

    if champion_metrics:
        champion_vals = [champion_metrics.get(k, 0) for k in metric_keys]
        bars1 = ax.bar(x - width / 2, champion_vals, width,
                       label=f"Champion ({champion_version})",
                       color="#5B9BD5", edgecolor="white")
        bars2 = ax.bar(x + width / 2, challenger_vals, width,
                       label=f"Challenger ({challenger_version})",
                       color="#ED7D31", edgecolor="white")

        # Delta 표시
        for i, (cv, chv) in enumerate(zip(champion_vals, challenger_vals)):
            delta = chv - cv
            sign = "+" if delta >= 0 else ""
            color = "#2E7D32" if delta >= 0 else "#C62828"
            ax.annotate(f"{sign}{delta:.4f}",
                        xy=(x[i] + width / 2, chv),
                        xytext=(0, 8), textcoords="offset points",
                        ha="center", fontsize=9, fontweight="bold", color=color)
    else:
        bars2 = ax.bar(x, challenger_vals, width * 1.5,
                       label=f"Challenger ({challenger_version})",
                       color="#ED7D31", edgecolor="white")

    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Classifier Metrics Comparison", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Chart saved: {save_path}")


def _generate_regressor_comparison_chart(
    champion_metrics: Optional[Dict[str, Any]],
    challenger_metrics: Dict[str, Any],
    champion_version: Optional[str],
    challenger_version: str,
    model_name: str,
    save_path: Path,
):
    """Regressor 메트릭 비교 바 차트"""
    metric_keys = ["mae", "rmse", "r2"]
    labels = ["MAE", "RMSE", "R²"]

    challenger_vals = [challenger_metrics.get(k, 0) for k in metric_keys]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # MAE, RMSE (낮을수록 좋음)
    ax1 = axes[0]
    x1 = np.arange(2)
    width = 0.35
    c_vals_err = challenger_vals[:2]

    if champion_metrics:
        ch_vals_err = [champion_metrics.get(k, 0) for k in metric_keys[:2]]
        ax1.bar(x1 - width / 2, ch_vals_err, width,
                label=f"Champion ({champion_version})", color="#5B9BD5", edgecolor="white")
        bars = ax1.bar(x1 + width / 2, c_vals_err, width,
                       label=f"Challenger ({challenger_version})", color="#ED7D31", edgecolor="white")
        for i, (cv, chv) in enumerate(zip(ch_vals_err, c_vals_err)):
            delta = chv - cv
            sign = "+" if delta >= 0 else ""
            color = "#C62828" if delta >= 0 else "#2E7D32"  # 에러는 낮을수록 좋음
            ax1.annotate(f"{sign}{delta:.4f}",
                         xy=(x1[i] + width / 2, chv),
                         xytext=(0, 8), textcoords="offset points",
                         ha="center", fontsize=9, fontweight="bold", color=color)
    else:
        ax1.bar(x1, c_vals_err, width * 1.5,
                label=f"Challenger ({challenger_version})", color="#ED7D31", edgecolor="white")

    ax1.set_title(f"{model_name} - Error Metrics (lower=better)", fontsize=11, fontweight="bold")
    ax1.set_xticks(x1)
    ax1.set_xticklabels(labels[:2], fontsize=11)
    ax1.legend(fontsize=9)
    ax1.grid(axis="y", alpha=0.3)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # R² (높을수록 좋음)
    ax2 = axes[1]
    x2 = np.arange(1)
    c_r2 = [challenger_vals[2]]

    if champion_metrics:
        ch_r2 = [champion_metrics.get("r2", 0)]
        ax2.bar(x2 - width / 2, ch_r2, width,
                label=f"Champion ({champion_version})", color="#5B9BD5", edgecolor="white")
        ax2.bar(x2 + width / 2, c_r2, width,
                label=f"Challenger ({challenger_version})", color="#ED7D31", edgecolor="white")
        delta = c_r2[0] - ch_r2[0]
        sign = "+" if delta >= 0 else ""
        color = "#2E7D32" if delta >= 0 else "#C62828"
        ax2.annotate(f"{sign}{delta:.4f}",
                     xy=(x2[0] + width / 2, c_r2[0]),
                     xytext=(0, 8), textcoords="offset points",
                     ha="center", fontsize=9, fontweight="bold", color=color)
    else:
        ax2.bar(x2, c_r2, width * 1.5,
                label=f"Challenger ({challenger_version})", color="#ED7D31", edgecolor="white")

    ax2.set_title(f"{model_name} - R² (higher=better)", fontsize=11, fontweight="bold")
    ax2.set_xticks(x2)
    ax2.set_xticklabels(["R²"], fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(axis="y", alpha=0.3)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Chart saved: {save_path}")


def _generate_decision_summary_chart(
    comparison: ComparisonResult,
    save_path: Path,
):
    """배포 결정 요약 차트 (ROC AUC + F1 비교 + 기준선)"""
    fig, ax = plt.subplots(figsize=(8, 5))

    metrics = ["ROC AUC", "F1 Score"]
    challenger_vals = [comparison.challenger_roc_auc, comparison.challenger_f1]

    x = np.arange(len(metrics))
    width = 0.3

    if comparison.champion_version and comparison.champion_roc_auc is not None:
        champion_vals = [comparison.champion_roc_auc, comparison.champion_f1]
        ax.bar(x - width / 2, champion_vals, width,
               label=f"Champion ({comparison.champion_version})",
               color="#5B9BD5", edgecolor="white", zorder=3)
        ax.bar(x + width / 2, challenger_vals, width,
               label=f"Challenger ({comparison.challenger_version})",
               color="#ED7D31", edgecolor="white", zorder=3)
    else:
        ax.bar(x, challenger_vals, width * 1.5,
               label=f"Challenger ({comparison.challenger_version})",
               color="#ED7D31", edgecolor="white", zorder=3)

    # 최소 기준선 (ROC AUC >= 0.60)
    ax.axhline(y=0.60, color="#C62828", linestyle="--", linewidth=1.5,
               label="Min ROC AUC (0.60)", zorder=2)

    # 결정 표시
    decision_color = "#2E7D32" if comparison.decision == "deploy" else "#C62828"
    decision_text = f"Decision: {comparison.decision.upper()}"
    ax.text(0.98, 0.95, decision_text,
            transform=ax.transAxes, fontsize=13, fontweight="bold",
            color=decision_color, ha="right", va="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor=decision_color, alpha=0.9))

    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Deployment Decision Summary", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # 사유 표시
    ax.text(0.5, -0.12, comparison.reason,
            transform=ax.transAxes, fontsize=9, ha="center", va="top",
            style="italic", color="#555555")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Chart saved: {save_path}")


# ============================================================
# 백테스트 차트 생성
# ============================================================

def _generate_equity_curve_chart(equity_df, initial_capital: float, save_path: Path):
    """Equity Curve 차트"""
    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(equity_df["date"], equity_df["equity"], linewidth=2, color="#ED7D31")
    ax.axhline(initial_capital, color="gray", linestyle="--", alpha=0.5, label=f"Initial (${initial_capital:,.0f})")
    ax.fill_between(equity_df["date"], initial_capital, equity_df["equity"],
                    where=equity_df["equity"] >= initial_capital, alpha=0.15, color="#2E7D32")
    ax.fill_between(equity_df["date"], initial_capital, equity_df["equity"],
                    where=equity_df["equity"] < initial_capital, alpha=0.15, color="#C62828")

    final = equity_df["equity"].iloc[-1]
    total_ret = (final / initial_capital - 1) * 100
    ax.set_title(f"Equity Curve (Total Return: {total_ret:+.2f}%)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Capital ($)")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Chart saved: {save_path}")


def _generate_trade_analysis_chart(trades_df, save_path: Path):
    """거래 수익률 분포 + 청산 이유별 차트"""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # 1. 수익률 분포
    ax = axes[0]
    ax.hist(trades_df["return_pct"], bins=40, alpha=0.7, edgecolor="black", color="#5B9BD5")
    ax.axvline(0, color="red", linestyle="--", linewidth=2)
    ax.set_title("Return Distribution", fontsize=12, fontweight="bold")
    ax.set_xlabel("Return (%)")
    ax.set_ylabel("Frequency")
    ax.grid(alpha=0.3)

    # 2. 청산 이유별 수익률
    ax = axes[1]
    exit_groups = trades_df.groupby("exit_reason")["return_pct"]
    labels = []
    data = []
    for reason, group in exit_groups:
        labels.append(reason)
        data.append(group.values)
    if data:
        bp = ax.boxplot(data, labels=labels, patch_artist=True)
        colors = {"take_profit": "#2E7D32", "stop_loss": "#C62828", "close": "#5B9BD5"}
        for patch, label in zip(bp["boxes"], labels):
            patch.set_facecolor(colors.get(label, "#888888"))
            patch.set_alpha(0.7)
    ax.axhline(0, color="red", linestyle="--", alpha=0.5)
    ax.set_title("Return by Exit Reason", fontsize=12, fontweight="bold")
    ax.set_ylabel("Return (%)")
    ax.grid(alpha=0.3, axis="y")

    # 3. 청산 이유 파이 차트
    ax = axes[2]
    exit_counts = trades_df["exit_reason"].value_counts()
    colors_pie = [colors.get(r, "#888888") for r in exit_counts.index]
    ax.pie(exit_counts.values, labels=exit_counts.index, autopct="%1.1f%%",
           colors=colors_pie, startangle=90)
    ax.set_title("Exit Reason Ratio", fontsize=12, fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Chart saved: {save_path}")


# ============================================================
# 리포트 생성
# ============================================================

def generate_report(
    version: str,
    trigger_type: str,
    training_duration_seconds: float,
    training_samples: int,
    data_date_range: tuple,
    test_metrics: Dict[str, Any],
    comparison: ComparisonResult,
    is_deployed: bool,
    model_path: str,
    hyperparameters: Dict[str, Any],
    model_base_dir: str = "./models/stacking",
    champion_test_metrics: Optional[Dict[str, Any]] = None,
    backtest_result: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Markdown 리포트 + 비교 차트 생성

    Args:
        version: 모델 버전
        trigger_type: 트리거 유형
        training_duration_seconds: 학습 소요 시간
        training_samples: 학습 샘플 수
        data_date_range: (시작일, 종료일)
        test_metrics: 챌린저 테스트 메트릭 (모델별)
        comparison: Champion-Challenger 비교 결과
        is_deployed: 배포 여부
        model_path: 모델 저장 경로
        hyperparameters: 학습 하이퍼파라미터
        model_base_dir: 모델 베이스 디렉토리
        champion_test_metrics: 챔피언 테스트 메트릭 (모델별, None이면 첫 모델)

    Returns:
        리포트 파일 경로
    """
    report_dir = Path(model_base_dir) / version
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "report.md"

    # ========================================
    # 1. 차트 생성
    # ========================================
    chart_files = []

    # Decision Summary 차트
    decision_chart = report_dir / "decision_summary.png"
    try:
        _generate_decision_summary_chart(comparison, decision_chart)
        chart_files.append(("decision_summary.png", "Deployment Decision"))
    except Exception as e:
        logger.warning(f"Decision chart failed: {e}")

    # Classifier 비교 차트
    clf_metrics = test_metrics.get("classifier", {})
    champion_clf = champion_test_metrics.get("classifier") if champion_test_metrics else None
    if clf_metrics:
        clf_chart = report_dir / "classifier_comparison.png"
        try:
            _generate_classifier_comparison_chart(
                champion_metrics=champion_clf,
                challenger_metrics=clf_metrics,
                champion_version=comparison.champion_version,
                challenger_version=version,
                save_path=clf_chart,
            )
            chart_files.append(("classifier_comparison.png", "Classifier Metrics"))
        except Exception as e:
            logger.warning(f"Classifier chart failed: {e}")

    # Regressor 비교 차트
    for model_key, model_name in [
        ("regressor_up", "Regressor Up (Close)"),
        ("regressor_high", "Regressor High (Max)"),
        ("regressor_down", "Regressor Down (Loss)"),
    ]:
        reg_metrics = test_metrics.get(model_key, {})
        champion_reg = champion_test_metrics.get(model_key) if champion_test_metrics else None
        if reg_metrics:
            chart_path = report_dir / f"{model_key}_comparison.png"
            try:
                _generate_regressor_comparison_chart(
                    champion_metrics=champion_reg,
                    challenger_metrics=reg_metrics,
                    champion_version=comparison.champion_version,
                    challenger_version=version,
                    model_name=model_name,
                    save_path=chart_path,
                )
                chart_files.append((f"{model_key}_comparison.png", model_name))
            except Exception as e:
                logger.warning(f"{model_key} chart failed: {e}")

    # 백테스트 차트
    if backtest_result and backtest_result.get("equity_df") is not None:
        equity_chart = report_dir / "equity_curve.png"
        try:
            _generate_equity_curve_chart(
                backtest_result["equity_df"],
                backtest_result["config"].initial_capital,
                equity_chart,
            )
            chart_files.append(("equity_curve.png", "Equity Curve"))
        except Exception as e:
            logger.warning(f"Equity curve chart failed: {e}")

        trade_chart = report_dir / "trade_analysis.png"
        try:
            _generate_trade_analysis_chart(
                backtest_result["trades_df"],
                trade_chart,
            )
            chart_files.append(("trade_analysis.png", "Trade Analysis"))
        except Exception as e:
            logger.warning(f"Trade analysis chart failed: {e}")

    # ========================================
    # 2. Markdown 작성
    # ========================================
    lines = []

    # 헤더
    lines.append(f"# Model Retraining Report - {version}")
    lines.append("")
    lines.append(f"- **Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- **Trigger**: {trigger_type}")
    lines.append(f"- **Duration**: {training_duration_seconds:.1f}s ({training_duration_seconds/60:.1f}min)")
    lines.append(f"- **Status**: {'DEPLOYED' if is_deployed else 'REJECTED'}")
    lines.append("")

    # Decision Summary 차트 (맨 위)
    if decision_chart.exists():
        lines.append("![Decision Summary](./decision_summary.png)")
        lines.append("")

    # 데이터 요약
    lines.append("---")
    lines.append("## 1. Training Data")
    lines.append("")
    lines.append(f"| Item | Value |")
    lines.append(f"|------|-------|")
    lines.append(f"| Samples | {training_samples:,} |")
    lines.append(f"| Date Range | {data_date_range[0]} ~ {data_date_range[1]} |")
    lines.append("")

    # 하이퍼파라미터
    lines.append("## 2. Hyperparameters")
    lines.append("")
    lines.append(f"| Parameter | Value |")
    lines.append(f"|-----------|-------|")
    for k, v in hyperparameters.items():
        lines.append(f"| {k} | {v} |")
    lines.append("")

    # 테스트 메트릭 + 차트
    lines.append("---")
    lines.append("## 3. Test Metrics")
    lines.append("")

    # Classifier
    if clf_metrics:
        lines.append("### Model 1: Classifier (Direction)")
        lines.append("")
        if (report_dir / "classifier_comparison.png").exists():
            lines.append("![Classifier Comparison](./classifier_comparison.png)")
            lines.append("")
        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|-------|")
        for key in ["accuracy", "roc_auc", "f1_score", "precision", "recall"]:
            val = clf_metrics.get(key)
            if val is not None:
                lines.append(f"| {key} | {val:.4f} |")
        lines.append("")

    # Regressors
    for model_key, title in [
        ("regressor_up", "Model 2: Regressor Up (Close Return)"),
        ("regressor_high", "Model 2-1: Regressor High (Max Return)"),
        ("regressor_down", "Model 3: Regressor Down (Loss)"),
    ]:
        reg = test_metrics.get(model_key, {})
        if reg:
            lines.append(f"### {title}")
            lines.append("")
            chart_file = f"{model_key}_comparison.png"
            if (report_dir / chart_file).exists():
                lines.append(f"![{model_key}](./{chart_file})")
                lines.append("")
            lines.append(f"| Metric | Value |")
            lines.append(f"|--------|-------|")
            for key in ["mae", "rmse", "r2"]:
                val = reg.get(key)
                if val is not None:
                    lines.append(f"| {key} | {val:.4f} |")
            lines.append("")

    # Champion-Challenger 비교
    lines.append("---")
    lines.append("## 4. Champion-Challenger Comparison")
    lines.append("")

    if comparison.champion_version is None:
        lines.append("> First model - no champion to compare against")
        lines.append("")
    else:
        lines.append(f"| | Champion ({comparison.champion_version}) | Challenger ({comparison.challenger_version}) | Delta |")
        lines.append(f"|---|---|---|---|")
        if comparison.champion_roc_auc is not None:
            delta_roc = comparison.challenger_roc_auc - comparison.champion_roc_auc
            sign = "+" if delta_roc >= 0 else ""
            lines.append(f"| ROC AUC | {comparison.champion_roc_auc:.4f} | {comparison.challenger_roc_auc:.4f} | {sign}{delta_roc:.4f} |")
        if comparison.champion_f1 is not None:
            delta_f1 = comparison.challenger_f1 - comparison.champion_f1
            sign = "+" if delta_f1 >= 0 else ""
            lines.append(f"| F1 Score | {comparison.champion_f1:.4f} | {comparison.challenger_f1:.4f} | {sign}{delta_f1:.4f} |")
        lines.append("")

    lines.append(f"**Decision**: `{comparison.decision.upper()}`")
    lines.append("")
    lines.append(f"**Reason**: {comparison.reason}")
    lines.append("")

    # 배포 기준 체크리스트
    lines.append("### Deployment Criteria Checklist")
    lines.append("")
    lines.append("| Criteria | Threshold | Actual | Result |")
    lines.append("|----------|-----------|--------|--------|")

    roc_auc_val = comparison.challenger_roc_auc
    roc_pass = roc_auc_val >= 0.55
    lines.append(f"| ROC AUC >= 0.55 | 0.55 | {roc_auc_val:.4f} | {'✅ PASS' if roc_pass else '❌ FAIL'} |")

    if backtest_result and backtest_result.get("metrics"):
        bt = backtest_result["metrics"]
        bt_return = bt.get("total_return_pct", 0.0)
        bt_sharpe = bt.get("sharpe_ratio", 0.0)
        bt_pf = bt.get("profit_factor", 0.0)

        lines.append(f"| Backtest Return >= 0% | 0.00% | {bt_return:+.2f}% | {'✅ PASS' if bt_return >= 0.0 else '❌ FAIL'} |")
        lines.append(f"| Sharpe Ratio >= 1.0 | 1.000 | {bt_sharpe:.3f} | {'✅ PASS' if bt_sharpe >= 1.0 else '❌ FAIL'} |")
        lines.append(f"| Profit Factor >= 1.0 | 1.000 | {bt_pf:.3f} | {'✅ PASS' if bt_pf >= 1.0 else '❌ FAIL'} |")
    else:
        lines.append("| Backtest Return >= 0% | 0.00% | N/A | ⏭️ SKIP |")
        lines.append("| Sharpe Ratio >= 1.0 | 1.000 | N/A | ⏭️ SKIP |")
        lines.append("| Profit Factor >= 1.0 | 1.000 | N/A | ⏭️ SKIP |")

    lines.append("")

    # 백테스트
    lines.append("---")
    lines.append("## 5. Backtest Results")
    lines.append("")

    if backtest_result and backtest_result.get("metrics"):
        bt = backtest_result["metrics"]

        if (report_dir / "equity_curve.png").exists():
            lines.append("![Equity Curve](./equity_curve.png)")
            lines.append("")

        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| Total Return | {bt.get('total_return_pct', 0):+.2f}% |")
        lines.append(f"| CAGR | {bt.get('cagr_pct', 0):+.2f}% |")
        lines.append(f"| Sharpe Ratio | {bt.get('sharpe_ratio', 0):.3f} |")
        lines.append(f"| Sortino Ratio | {bt.get('sortino_ratio', 0):.3f} |")
        lines.append(f"| Max Drawdown | {bt.get('max_drawdown_pct', 0):.2f}% |")
        lines.append(f"| Win Rate | {bt.get('win_rate_pct', 0):.2f}% |")
        lines.append(f"| Profit Factor | {bt.get('profit_factor', 0):.3f} |")
        lines.append(f"| Trades | {bt.get('n_trades', 0)} |")
        lines.append(f"| Expectancy | ${bt.get('expectancy', 0):,.2f} |")
        lines.append("")

        # 청산 이유별 통계
        exit_reasons = bt.get("exit_reasons", {})
        avg_by_exit = bt.get("avg_return_by_exit", {})
        if exit_reasons:
            lines.append("### Exit Reason Breakdown")
            lines.append("")

            if (report_dir / "trade_analysis.png").exists():
                lines.append("![Trade Analysis](./trade_analysis.png)")
                lines.append("")

            lines.append("| Reason | Count | Avg Return |")
            lines.append("|--------|-------|------------|")
            total_trades = bt.get("n_trades", 1)
            for reason, count in exit_reasons.items():
                pct = count / total_trades * 100
                avg_ret = avg_by_exit.get(reason, 0)
                lines.append(f"| {reason} | {count} ({pct:.1f}%) | {avg_ret:+.2f}% |")
            lines.append("")
    else:
        lines.append("> Backtest not available")
        lines.append("")

    # Artifacts
    lines.append("---")
    lines.append("## 6. Artifacts")
    lines.append("")
    lines.append(f"- **Model**: `{model_path}`")
    lines.append(f"- **Report**: `{report_path}`")
    if chart_files:
        lines.append(f"- **Charts**: {', '.join(f'`{c[0]}`' for c in chart_files)}")
    if is_deployed:
        lines.append(f"- **Active Symlink**: `{model_base_dir}/active` -> `{version}/`")
    lines.append("")

    content = "\n".join(lines)
    report_path.write_text(content, encoding="utf-8")
    logger.info(f"Report saved: {report_path}")

    return report_path
