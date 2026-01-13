"""
백테스팅 성과 지표 계산

- 수익률 지표: 총수익률, 연간수익률, CAGR
- 리스크 지표: 샤프비율, 최대낙폭, 변동성
- 승률/손익비: 승률, 평균손익, Profit Factor
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional


def calculate_returns(equity_series: pd.Series) -> pd.Series:
    """일일 수익률 계산"""
    return equity_series.pct_change().fillna(0)


def calculate_total_return(initial_capital: float, final_capital: float) -> float:
    """총 수익률 (%)"""
    return (final_capital / initial_capital - 1) * 100


def calculate_cagr(initial_capital: float, final_capital: float, years: float) -> float:
    """연평균 성장률 (CAGR)"""
    if years <= 0:
        return 0.0
    return (np.power(final_capital / initial_capital, 1 / years) - 1) * 100


def calculate_sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.02,
    periods_per_year: int = 252
) -> float:
    """
    샤프 비율

    Args:
        returns: 일일 수익률 시리즈
        risk_free_rate: 무위험 수익률 (연율)
        periods_per_year: 연간 거래일 수

    Returns:
        샤프 비율
    """
    if len(returns) == 0 or returns.std() == 0:
        return 0.0

    excess_returns = returns - (risk_free_rate / periods_per_year)
    sharpe = excess_returns.mean() / returns.std()
    return sharpe * np.sqrt(periods_per_year)


def calculate_sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.02,
    periods_per_year: int = 252
) -> float:
    """
    소르티노 비율 (하방 위험만 고려)

    Args:
        returns: 일일 수익률 시리즈
        risk_free_rate: 무위험 수익률 (연율)
        periods_per_year: 연간 거래일 수

    Returns:
        소르티노 비율
    """
    if len(returns) == 0:
        return 0.0

    excess_returns = returns - (risk_free_rate / periods_per_year)
    downside_returns = returns[returns < 0]

    if len(downside_returns) == 0 or downside_returns.std() == 0:
        return 0.0

    sortino = excess_returns.mean() / downside_returns.std()
    return sortino * np.sqrt(periods_per_year)


def calculate_max_drawdown(equity_series: pd.Series) -> Dict:
    """
    최대 낙폭 (Maximum Drawdown)

    Returns:
        {
            'max_drawdown_pct': 최대 낙폭 (%),
            'max_drawdown_duration': 회복 기간 (일),
            'peak_date': 고점 날짜,
            'trough_date': 저점 날짜,
            'recovery_date': 회복 날짜 (None if not recovered)
        }
    """
    if len(equity_series) == 0:
        return {
            'max_drawdown_pct': 0.0,
            'max_drawdown_duration': 0,
            'peak_date': None,
            'trough_date': None,
            'recovery_date': None
        }

    # 누적 최대값
    cummax = equity_series.expanding().max()

    # Drawdown (%)
    drawdown = (equity_series / cummax - 1) * 100

    # 최대 낙폭
    max_dd = drawdown.min()
    max_dd_idx = drawdown.idxmin()

    # 고점 (최대 낙폭 발생 전 마지막 고점)
    peak_idx = equity_series[:max_dd_idx].idxmax()

    # 회복 날짜 (고점을 넘어선 첫 날)
    recovery_idx = None
    peak_value = equity_series[peak_idx]
    after_trough = equity_series[max_dd_idx:]
    recovery_mask = after_trough >= peak_value

    if recovery_mask.any():
        recovery_idx = after_trough[recovery_mask].index[0]

    # 회복 기간 (일)
    if recovery_idx is not None:
        duration = (recovery_idx - peak_idx).days if hasattr(peak_idx, 'days') else len(equity_series[peak_idx:recovery_idx])
    else:
        duration = len(equity_series[peak_idx:])

    return {
        'max_drawdown_pct': max_dd,
        'max_drawdown_duration': duration,
        'peak_date': peak_idx,
        'trough_date': max_dd_idx,
        'recovery_date': recovery_idx
    }


def calculate_win_rate(trades_df: pd.DataFrame) -> float:
    """승률 (%)"""
    if len(trades_df) == 0:
        return 0.0
    return (trades_df['pnl'] > 0).mean() * 100


def calculate_profit_factor(trades_df: pd.DataFrame) -> float:
    """
    Profit Factor (총이익 / 총손실)

    - > 1.0: 수익
    - < 1.0: 손실
    """
    if len(trades_df) == 0:
        return 0.0

    gross_profit = trades_df[trades_df['pnl'] > 0]['pnl'].sum()
    gross_loss = abs(trades_df[trades_df['pnl'] < 0]['pnl'].sum())

    if gross_loss == 0:
        return np.inf if gross_profit > 0 else 0.0

    return gross_profit / gross_loss


def calculate_avg_win_loss(trades_df: pd.DataFrame) -> Dict:
    """평균 승/패 금액 및 비율"""
    if len(trades_df) == 0:
        return {
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'win_loss_ratio': 0.0
        }

    wins = trades_df[trades_df['pnl'] > 0]['pnl']
    losses = trades_df[trades_df['pnl'] < 0]['pnl']

    avg_win = wins.mean() if len(wins) > 0 else 0.0
    avg_loss = losses.mean() if len(losses) > 0 else 0.0
    win_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0.0

    return {
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'win_loss_ratio': win_loss_ratio
    }


def calculate_expectancy(trades_df: pd.DataFrame) -> float:
    """
    기대값 (Expectancy) = (승률 × 평균수익) - (패율 × 평균손실)

    - > 0: 장기적으로 수익 기대
    - < 0: 장기적으로 손실 기대
    """
    if len(trades_df) == 0:
        return 0.0

    win_rate = calculate_win_rate(trades_df) / 100
    avg_stats = calculate_avg_win_loss(trades_df)

    expectancy = (win_rate * avg_stats['avg_win']) + ((1 - win_rate) * avg_stats['avg_loss'])
    return expectancy


def calculate_all_metrics(
    trades_df: pd.DataFrame,
    equity_df: pd.DataFrame,
    initial_capital: float,
    risk_free_rate: float = 0.02
) -> Dict:
    """
    모든 성과 지표 계산

    Args:
        trades_df: 거래 내역 DataFrame
        equity_df: 일일 자본 DataFrame (date, equity)
        initial_capital: 초기 자본
        risk_free_rate: 무위험 수익률 (연율)

    Returns:
        성과 지표 딕셔너리
    """
    if len(equity_df) == 0 or len(trades_df) == 0:
        return {}

    # Equity 시리즈
    equity_series = equity_df.set_index('date')['equity'] if 'date' in equity_df.columns else equity_df['equity']
    final_capital = equity_series.iloc[-1]

    # 일일 수익률
    daily_returns = calculate_returns(equity_series)

    # 기간 계산 (년)
    if hasattr(equity_series.index, 'to_series'):
        days = (equity_series.index[-1] - equity_series.index[0]).days
    else:
        days = len(equity_series)
    years = days / 365.25

    # 수익률 지표
    total_return = calculate_total_return(initial_capital, final_capital)
    cagr = calculate_cagr(initial_capital, final_capital, years) if years > 0 else 0.0

    # 리스크 지표
    sharpe = calculate_sharpe_ratio(daily_returns, risk_free_rate)
    sortino = calculate_sortino_ratio(daily_returns, risk_free_rate)
    max_dd = calculate_max_drawdown(equity_series)
    volatility = daily_returns.std() * np.sqrt(252) * 100  # 연율화된 변동성 (%)

    # 거래 통계
    win_rate = calculate_win_rate(trades_df)
    profit_factor = calculate_profit_factor(trades_df)
    avg_stats = calculate_avg_win_loss(trades_df)
    expectancy = calculate_expectancy(trades_df)

    # 통합 결과
    metrics = {
        # 기본 정보
        'initial_capital': initial_capital,
        'final_capital': final_capital,
        'total_pnl': final_capital - initial_capital,
        'n_trades': len(trades_df),
        'n_days': len(equity_series),
        'years': years,

        # 수익률
        'total_return_pct': total_return,
        'cagr_pct': cagr,
        'daily_return_mean': daily_returns.mean() * 100,
        'daily_return_std': daily_returns.std() * 100,

        # 리스크 조정 수익률
        'sharpe_ratio': sharpe,
        'sortino_ratio': sortino,

        # 리스크
        'max_drawdown_pct': max_dd['max_drawdown_pct'],
        'max_drawdown_duration': max_dd['max_drawdown_duration'],
        'volatility_annual_pct': volatility,

        # 거래 통계
        'win_rate_pct': win_rate,
        'profit_factor': profit_factor,
        'avg_win': avg_stats['avg_win'],
        'avg_loss': avg_stats['avg_loss'],
        'win_loss_ratio': avg_stats['win_loss_ratio'],
        'expectancy': expectancy,

        # 청산 이유별 통계
        'exit_reasons': trades_df['exit_reason'].value_counts().to_dict(),
        'avg_return_by_exit': trades_df.groupby('exit_reason')['return_pct'].mean().to_dict()
    }

    return metrics


def print_metrics_report(metrics: Dict, title: str = "백테스트 성과 지표"):
    """성과 지표 출력"""
    print("=" * 80)
    print(f"{title}")
    print("=" * 80)

    print(f"\n1. 기본 정보")
    print(f"   초기 자본: ${metrics['initial_capital']:,.0f}")
    print(f"   최종 자본: ${metrics['final_capital']:,.0f}")
    print(f"   총 손익: ${metrics['total_pnl']:,.0f}")
    print(f"   거래 횟수: {metrics['n_trades']:,}회")
    print(f"   거래 일수: {metrics['n_days']:,}일 ({metrics['years']:.2f}년)")

    print(f"\n2. 수익률 지표")
    print(f"   총 수익률: {metrics['total_return_pct']:+.2f}%")
    print(f"   연평균 수익률 (CAGR): {metrics['cagr_pct']:+.2f}%")
    print(f"   일평균 수익률: {metrics['daily_return_mean']:+.4f}%")

    print(f"\n3. 리스크 조정 수익률")
    print(f"   샤프 비율: {metrics['sharpe_ratio']:.3f}")
    print(f"   소르티노 비율: {metrics['sortino_ratio']:.3f}")

    print(f"\n4. 리스크 지표")
    print(f"   최대 낙폭 (MDD): {metrics['max_drawdown_pct']:.2f}%")
    print(f"   MDD 회복 기간: {metrics['max_drawdown_duration']}일")
    print(f"   연간 변동성: {metrics['volatility_annual_pct']:.2f}%")

    print(f"\n5. 거래 통계")
    print(f"   승률: {metrics['win_rate_pct']:.2f}%")
    print(f"   Profit Factor: {metrics['profit_factor']:.3f}")
    print(f"   평균 수익: ${metrics['avg_win']:,.2f}")
    print(f"   평균 손실: ${metrics['avg_loss']:,.2f}")
    print(f"   손익비: {metrics['win_loss_ratio']:.3f}")
    print(f"   기대값: ${metrics['expectancy']:,.2f}")

    print(f"\n6. 청산 이유별 통계")
    for reason, count in metrics['exit_reasons'].items():
        pct = count / metrics['n_trades'] * 100
        avg_ret = metrics['avg_return_by_exit'].get(reason, 0)
        print(f"   {reason}: {count}회 ({pct:.1f}%) - 평균 수익률 {avg_ret:+.2f}%")

    print("\n" + "=" * 80)
