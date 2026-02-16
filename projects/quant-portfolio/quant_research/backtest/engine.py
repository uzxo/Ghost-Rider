"""
Walk-Forward Backtest Engine
==============================
Implements expanding/rolling window backtesting with
proper purging, embargo, and point-in-time discipline.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Optional, List, Callable
from datetime import timedelta

logger = logging.getLogger(__name__)


class BacktestEngine:
    """
    Walk-forward backtesting engine.

    At each rebalance date:
    1. Construct universe (point-in-time)
    2. Compute features (using only data available at that date)
    3. Train/retrain models (expanding or rolling window)
    4. Generate signals
    5. Optimize portfolio
    6. Record positions and returns
    """

    def __init__(self, config: dict):
        self.config = config
        self.rebalance_freq = config.get('backtest', {}).get(
            'rebalance_frequency', 'monthly'
        )
        self.embargo_days = config.get('backtest', {}).get('embargo_days', 5)
        self.walk_forward = config.get('backtest', {}).get(
            'walk_forward', 'expanding'
        )

        # Results storage
        self.positions_history: List[Dict] = []
        self.returns_history: List[Dict] = []
        self.weights_history: Dict[pd.Timestamp, pd.Series] = {}
        self.turnover_history: List[float] = []
        self.cost_history: List[float] = []

    def generate_rebalance_dates(self, start: str, end: str) -> List[pd.Timestamp]:
        """Generate rebalance dates based on frequency."""
        if self.rebalance_freq == 'monthly':
            dates = pd.date_range(start, end, freq='BME')
        elif self.rebalance_freq == 'quarterly':
            dates = pd.date_range(start, end, freq='BQE')
        elif self.rebalance_freq == 'weekly':
            dates = pd.date_range(start, end, freq='W-FRI')
        else:
            dates = pd.date_range(start, end, freq='BME')

        return list(dates)

    def run(self, close_prices: pd.DataFrame,
            signal_func: Callable,
            portfolio_func: Callable,
            universe_func: Callable = None,
            cost_func: Callable = None,
            start: str = None,
            end: str = None) -> Dict:
        """
        Execute the walk-forward backtest.

        Parameters
        ----------
        close_prices : pd.DataFrame
            Adjusted close prices (DatetimeIndex, tickers as columns).
        signal_func : callable
            Function(date, universe, prices_up_to_date) -> pd.Series of signals.
        portfolio_func : callable
            Function(signals, returns, date) -> pd.Series of weights.
        universe_func : callable, optional
            Function(date) -> list of eligible tickers.
        cost_func : callable, optional
            Function(old_weights, new_weights) -> cost_bps.
        start : str, optional
            Backtest start (default: 2 years after data start for warmup).
        end : str, optional
            Backtest end.

        Returns
        -------
        dict
            Backtest results including portfolio returns, weights, metrics.
        """
        if start is None:
            start = (close_prices.index[0] + timedelta(days=504)).strftime('%Y-%m-%d')
        if end is None:
            end = close_prices.index[-1].strftime('%Y-%m-%d')

        rebalance_dates = self.generate_rebalance_dates(start, end)
        logger.info(f"Backtest: {start} to {end}, "
                     f"{len(rebalance_dates)} rebalance dates")

        # Initialize
        current_weights = pd.Series(dtype=float)
        portfolio_value = 1.0
        daily_returns = []

        all_dates = close_prices.loc[start:end].index
        rebalance_set = set(rebalance_dates)
        reb_idx = 0

        for i, date in enumerate(all_dates):
            # Check if rebalance day
            is_rebalance = date in rebalance_set
            if not is_rebalance and reb_idx == 0:
                # Find first rebalance date
                upcoming = [d for d in rebalance_dates if d <= date]
                if upcoming:
                    is_rebalance = True

            if is_rebalance and reb_idx < len(rebalance_dates):
                reb_date = date
                reb_idx += 1

                # Get universe
                if universe_func:
                    universe = universe_func(date)
                else:
                    universe = [t for t in close_prices.columns
                                if not close_prices[t].loc[:date].dropna().empty]

                # Only use data available up to embargo boundary
                embargo_date = date - timedelta(days=self.embargo_days)
                available_prices = close_prices.loc[:embargo_date]

                try:
                    # Generate signals
                    signals = signal_func(date, universe, available_prices)

                    # Compute returns for covariance
                    returns = available_prices[universe].pct_change().dropna()

                    # Optimize portfolio
                    new_weights = portfolio_func(signals, returns, date)
                    new_weights = new_weights.reindex(universe, fill_value=0)
                    new_weights = new_weights[new_weights > 0]

                    # Compute turnover
                    if not current_weights.empty:
                        all_tickers = set(current_weights.index) | set(new_weights.index)
                        old = current_weights.reindex(all_tickers, fill_value=0)
                        new = new_weights.reindex(all_tickers, fill_value=0)
                        turnover = (new - old).abs().sum() / 2
                        self.turnover_history.append(turnover)

                        # Transaction cost
                        if cost_func:
                            cost = cost_func(current_weights, new_weights)
                            self.cost_history.append(cost)
                            portfolio_value *= (1 - cost / 10000)
                    else:
                        self.turnover_history.append(1.0)

                    current_weights = new_weights
                    self.weights_history[date] = current_weights.copy()

                except Exception as e:
                    logger.warning(f"Rebalance failed on {date}: {e}")
                    continue

            # Compute daily portfolio return
            if not current_weights.empty and i > 0:
                prev_date = all_dates[i - 1]
                day_returns = (
                    close_prices.loc[date] / close_prices.loc[prev_date] - 1
                )
                port_ret = (
                    current_weights *
                    day_returns.reindex(current_weights.index, fill_value=0)
                ).sum()

                daily_returns.append({
                    'date': date,
                    'return': port_ret,
                    'n_holdings': (current_weights > 0).sum(),
                })

        # Compile results
        returns_df = pd.DataFrame(daily_returns).set_index('date')
        portfolio_returns = returns_df['return']

        results = {
            'returns': portfolio_returns,
            'weights_history': self.weights_history,
            'turnover': self.turnover_history,
            'costs': self.cost_history,
            'n_rebalances': reb_idx,
            'avg_turnover': np.mean(self.turnover_history) if self.turnover_history else 0,
            'avg_holdings': returns_df['n_holdings'].mean() if not returns_df.empty else 0,
        }

        logger.info(
            f"Backtest complete: {len(portfolio_returns)} days, "
            f"{reb_idx} rebalances, "
            f"avg turnover: {results['avg_turnover']:.1%}"
        )

        return results

    def purge_train_data(self, train_end: pd.Timestamp,
                          test_start: pd.Timestamp,
                          returns: pd.DataFrame,
                          horizon: int = 21) -> pd.DataFrame:
        """
        Remove training samples that overlap with the test period.

        This prevents label leakage when the prediction target
        uses overlapping return windows.

        Parameters
        ----------
        train_end : pd.Timestamp
            Last date of training data.
        test_start : pd.Timestamp
            First date of test data.
        returns : pd.DataFrame
            Full returns DataFrame.
        horizon : int
            Prediction horizon in days.

        Returns
        -------
        pd.DataFrame
            Purged training data.
        """
        purge_start = test_start - timedelta(days=horizon + self.embargo_days)
        train_data = returns.loc[:min(train_end, purge_start)]
        n_purged = len(returns.loc[:train_end]) - len(train_data)

        if n_purged > 0:
            logger.debug(f"Purged {n_purged} samples from training data")

        return train_data
