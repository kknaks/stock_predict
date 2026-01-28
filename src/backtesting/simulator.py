"""
백테스팅 시뮬레이터

갭 상승 종목 일중 트레이딩 전략 백테스트
- 시가 진입, 익절가/손절가/장마감 청산
- Model 2-1 (고가 예측)을 활용한 동적 익절 전략
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime


@dataclass
class TradeResult:
    """개별 거래 결과"""
    date: datetime
    symbol: str
    entry_price: float
    exit_price: float
    exit_reason: str  # 'take_profit', 'stop_loss', 'close'
    return_pct: float
    position_size: float
    pnl: float
    prob_up: float
    expected_return: float
    predicted_high: Optional[float] = None
    actual_high: Optional[float] = None


@dataclass
class BacktestConfig:
    """백테스트 설정"""
    initial_capital: float = 100000.0
    max_positions: int = 20
    position_sizing: str = 'equal'  # 'equal', 'expected_return_weighted', 'prob_weighted'
    commission_rate: float = 0.001  # 0.1%
    slippage_rate: float = 0.0005  # 0.05%

    # 필터링 기준
    min_expected_return: float = 1.0  # 최소 기대 수익률 (%)
    min_prob_up: float = 0.4  # 최소 상승 확률

    # 익절/손절 전략
    take_profit_strategy: str = 'model_2_1'  # 'fixed', 'model_2_1', 'none'
    take_profit_ratio: float = 0.8  # Model 2-1 예측 고가의 80%
    stop_loss_strategy: str = 'model_3'  # 'fixed', 'model_3', 'none'
    stop_loss_ratio: float = 0.5  # Model 3 예측 손실의 50%
    fixed_stop_loss_pct: float = -3.0  # 고정 손절 (%)


class GapTradingSimulator:
    """갭 상승 종목 일중 트레이딩 시뮬레이터"""

    def __init__(self, config: BacktestConfig = None):
        self.config = config or BacktestConfig()
        self.trades: List[TradeResult] = []
        self.daily_equity: List[Dict] = []
        self.capital = self.config.initial_capital

    def filter_signals(self, predictions: pd.DataFrame) -> pd.DataFrame:
        """
        신호 필터링

        Args:
            predictions: 모델 예측 결과
                - prob_up: 상승 확률
                - expected_return: 기대 수익률
                - return_if_up: 상승 시 예상 수익률
                - return_if_down: 하락 시 예상 손실률
                - predicted_high (optional): 예측 고가 수익률

        Returns:
            필터링된 신호
        """
        signals = predictions.copy()

        # 기본 필터
        mask = (
            (signals['expected_return'] >= self.config.min_expected_return) &
            (signals['prob_up'] >= self.config.min_prob_up)
        )

        signals = signals[mask].copy()

        # 기대 수익률 기준 정렬 (상위 N개 선택)
        signals = signals.sort_values('expected_return', ascending=False)

        return signals

    def calculate_position_size(
        self,
        n_positions: int,
        expected_returns: Optional[pd.Series] = None,
        probs_up: Optional[pd.Series] = None
    ) -> pd.Series:
        """
        포지션 크기 계산

        Args:
            n_positions: 포지션 수
            expected_returns: 기대 수익률 (가중 방식 사용 시)
            probs_up: 상승 확률 (가중 방식 사용 시)

        Returns:
            각 포지션의 자본 비중 (합 = 1.0)
        """
        if self.config.position_sizing == 'equal':
            # 동일 가중
            return pd.Series([1.0 / n_positions] * n_positions)

        elif self.config.position_sizing == 'expected_return_weighted':
            # 기대 수익률 가중 (양수만)
            weights = expected_returns.clip(lower=0)
            weights = weights / weights.sum()
            return weights

        elif self.config.position_sizing == 'prob_weighted':
            # 상승 확률 가중
            weights = probs_up
            weights = weights / weights.sum()
            return weights

        else:
            raise ValueError(f"Unknown position sizing: {self.config.position_sizing}")

    def calculate_exit_targets(
        self,
        entry_price: float,
        return_if_up: float,
        return_if_down: float,
        predicted_high: Optional[float] = None
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        익절가/손절가 계산

        Args:
            entry_price: 진입가 (시가)
            return_if_up: Model 2 예측 (종가 수익률)
            return_if_down: Model 3 예측 (손실률)
            predicted_high: Model 2-1 예측 (고가 수익률)

        Returns:
            (take_profit_price, stop_loss_price)
        """
        take_profit_price = None
        stop_loss_price = None

        # 익절가 계산
        if self.config.take_profit_strategy == 'model_2_1' and predicted_high is not None:
            # Model 2-1 예측의 80% (예: 예측 고가 +10% → 익절 +8%)
            target_return = predicted_high * self.config.take_profit_ratio
            take_profit_price = entry_price * (1 + target_return / 100)

        elif self.config.take_profit_strategy == 'fixed':
            # Model 2 예측 기준 고정 비율
            target_return = return_if_up * self.config.take_profit_ratio
            take_profit_price = entry_price * (1 + target_return / 100)

        # 손절가 계산
        if self.config.stop_loss_strategy == 'model_3':
            # Model 3 예측의 50% (예: 예측 -6% → 손절 -3%)
            target_loss = return_if_down * self.config.stop_loss_ratio
            stop_loss_price = entry_price * (1 + target_loss / 100)

        elif self.config.stop_loss_strategy == 'fixed':
            # 고정 손절 (예: -3%)
            stop_loss_price = entry_price * (1 + self.config.fixed_stop_loss_pct / 100)

        return take_profit_price, stop_loss_price

    def simulate_trade(
        self,
        entry_price: float,
        high: float,
        low: float,
        close: float,
        take_profit_price: Optional[float],
        stop_loss_price: Optional[float]
    ) -> Tuple[float, str]:
        """
        개별 거래 시뮬레이션 (당일 체결 로직)

        Args:
            entry_price: 시가 (진입가)
            high: 고가
            low: 저가
            close: 종가
            take_profit_price: 익절가
            stop_loss_price: 손절가

        Returns:
            (exit_price, exit_reason)
        """
        # 슬리피지 적용 (진입)
        entry_with_slippage = entry_price * (1 + self.config.slippage_rate)

        # 손절가 먼저 체크 (저가가 손절가 이하인지)
        if stop_loss_price and low <= stop_loss_price:
            exit_price = stop_loss_price * (1 - self.config.slippage_rate)
            return exit_price, 'stop_loss'

        # 익절가 체크 (고가가 익절가 이상인지)
        if take_profit_price and high >= take_profit_price:
            exit_price = take_profit_price * (1 - self.config.slippage_rate)
            return exit_price, 'take_profit'

        # 둘 다 체결 안 되면 종가 청산
        exit_price = close * (1 - self.config.slippage_rate)
        return exit_price, 'close'

    def run(
        self,
        data: pd.DataFrame,
        predictions: pd.DataFrame
    ) -> Dict:
        """
        백테스트 실행

        Args:
            data: 가격 데이터 (open, high, low, close, date, symbol 등)
            predictions: 모델 예측 결과
                - prob_up, expected_return, return_if_up, return_if_down
                - predicted_high (optional): Model 2-1 예측

        Returns:
            백테스트 결과 딕셔너리
        """
        # 데이터 병합
        df = data.merge(predictions, left_index=True, right_index=True, how='inner')

        # 날짜별로 그룹화
        if 'date' not in df.columns:
            raise ValueError("Data must have 'date' column")

        dates = sorted(df['date'].unique())

        for date in dates:
            daily_data = df[df['date'] == date].copy()

            # 신호 필터링
            signals = self.filter_signals(daily_data)

            if len(signals) == 0:
                # 거래 없음, equity 기록만
                self.daily_equity.append({
                    'date': date,
                    'equity': self.capital,
                    'n_trades': 0
                })
                continue

            # 최대 포지션 수 제한
            n_positions = min(len(signals), self.config.max_positions)
            signals = signals.head(n_positions)

            # 포지션 크기 계산
            position_weights = self.calculate_position_size(
                n_positions=n_positions,
                expected_returns=signals['expected_return'],
                probs_up=signals['prob_up']
            )

            # 각 포지션 거래
            daily_pnl = 0.0
            for idx, (i, row) in enumerate(signals.iterrows()):
                # 포지션 크기 (달러)
                position_size = self.capital * position_weights.iloc[idx]

                # 진입가
                entry_price = row['open']

                # 익절/손절가 계산
                predicted_high = row.get('predicted_high', None)
                take_profit_price, stop_loss_price = self.calculate_exit_targets(
                    entry_price=entry_price,
                    return_if_up=row['return_if_up'],
                    return_if_down=row['return_if_down'],
                    predicted_high=predicted_high
                )

                # 거래 시뮬레이션
                exit_price, exit_reason = self.simulate_trade(
                    entry_price=entry_price,
                    high=row['high'],
                    low=row['low'],
                    close=row['close'],
                    take_profit_price=take_profit_price,
                    stop_loss_price=stop_loss_price
                )

                # 수익률 계산 (슬리피지 이미 반영됨)
                return_pct = (exit_price / entry_price - 1) * 100

                # 수수료 차감
                commission = position_size * self.config.commission_rate * 2  # 매수/매도
                gross_pnl = position_size * (exit_price / entry_price - 1)
                net_pnl = gross_pnl - commission

                daily_pnl += net_pnl

                # 거래 기록
                trade = TradeResult(
                    date=date,
                    symbol=row.get('symbol', row.get('InfoCode', 'UNKNOWN')),
                    entry_price=entry_price,
                    exit_price=exit_price,
                    exit_reason=exit_reason,
                    return_pct=return_pct,
                    position_size=position_size,
                    pnl=net_pnl,
                    prob_up=row['prob_up'],
                    expected_return=row['expected_return'],
                    predicted_high=predicted_high,
                    actual_high=row['high']
                )
                self.trades.append(trade)

            # 자본 업데이트
            self.capital += daily_pnl

            # 일일 equity 기록
            self.daily_equity.append({
                'date': date,
                'equity': self.capital,
                'n_trades': len(signals),
                'pnl': daily_pnl
            })

        # 결과 반환
        return self.get_results()

    def get_results(self) -> Dict:
        """백테스트 결과 반환"""
        trades_df = pd.DataFrame([
            {
                'date': t.date,
                'symbol': t.symbol,
                'entry_price': t.entry_price,
                'exit_price': t.exit_price,
                'exit_reason': t.exit_reason,
                'return_pct': t.return_pct,
                'position_size': t.position_size,
                'pnl': t.pnl,
                'prob_up': t.prob_up,
                'expected_return': t.expected_return,
                'predicted_high': t.predicted_high,
                'actual_high': t.actual_high
            }
            for t in self.trades
        ])

        equity_df = pd.DataFrame(self.daily_equity)

        return {
            'trades': trades_df,
            'equity': equity_df,
            'config': self.config,
            'initial_capital': self.config.initial_capital,
            'final_capital': self.capital,
            'total_return_pct': (self.capital / self.config.initial_capital - 1) * 100
        }
