"""
Performance & Risk Metrics
===========================
Comprehensive suite of institutional-grade performance analytics.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class PerformanceMetrics:
    """Compute a full suite of performance and risk metrics."""

    def __init__(self, risk_free_rate: float = 0.04):
        self.rf = risk_free_rate

    def compute_all(self, returns: pd.Series,
                     benchmark_returns: pd.Series = None,
                     annualize: bool = True) -> Dict:
        """
        Compute all performance metrics.

        Parameters
        ----------
        returns : pd.Series
            Portfolio daily returns.
        benchmark_returns : pd.Series, optional
            Benchmark daily returns.
        annualize : bool
            Whether to annualize metrics.

        Returns
        -------
        dict
            Comprehensive metrics dictionary.
        """
        returns = returns.dropna()
        ann_factor = 252 if annualize else 1

        metrics = {}

        # ── Return Metrics ──
        metrics['total_return'] = (1 + returns).prod() - 1
        n_years = len(returns) / 252
        metrics['cagr'] = (
            (1 + metrics['total_return']) ** (1 / max(n_years, 0.01)) - 1
        )
        metrics['annualized_return'] = returns.mean() * ann_factor
        metrics['annualized_volatility'] = returns.std() * np.sqrt(ann_factor)

        # ── Risk-Adjusted Returns ──
        daily_rf = self.rf / 252
        excess = returns - daily_rf

        metrics['sharpe_ratio'] = (
            excess.mean() / excess.std() * np.sqrt(ann_factor)
            if excess.std() > 0 else 0
        )

        # Sortino (downside deviation)
        downside = excess[excess < 0]
        downside_std = np.sqrt((downside ** 2).mean()) * np.sqrt(ann_factor)
        metrics['sortino_ratio'] = (
            excess.mean() * ann_factor / downside_std
            if downside_std > 0 else 0
        )

        # ── Drawdown Analysis ──
        cum_returns = (1 + returns).cumprod()
        rolling_max = cum_returns.expanding().max()
        drawdowns = cum_returns / rolling_max - 1

        metrics['max_drawdown'] = drawdowns.min()
        metrics['calmar_ratio'] = (
            metrics['cagr'] / abs(metrics['max_drawdown'])
            if metrics['max_drawdown'] != 0 else 0
        )

        # Drawdown duration
        underwater = drawdowns < 0
        if underwater.any():
            dd_groups = (~underwater).cumsum()
            dd_lengths = underwater.groupby(dd_groups).sum()
            metrics['max_drawdown_duration_days'] = int(dd_lengths.max())
        else:
            metrics['max_drawdown_duration_days'] = 0

        # ── Tail Risk ──
        metrics['skewness'] = returns.skew()
        metrics['kurtosis'] = returns.kurtosis()
        metrics['var_95'] = np.percentile(returns, 5)
        metrics['var_99'] = np.percentile(returns, 1)
        metrics['cvar_95'] = returns[returns <= metrics['var_95']].mean()

        # ── Win/Loss Statistics ──
        metrics['hit_rate_daily'] = (returns > 0).mean()
        monthly = returns.resample('ME').sum()
        metrics['hit_rate_monthly'] = (monthly > 0).mean()
        gains = returns[returns > 0]
        losses = returns[returns < 0]
        metrics['profit_factor'] = (
            gains.sum() / abs(losses.sum()) if losses.sum() != 0 else np.inf
        )
        metrics['avg_win'] = gains.mean() if len(gains) > 0 else 0
        metrics['avg_loss'] = losses.mean() if len(losses) > 0 else 0
        metrics['win_loss_ratio'] = (
            abs(metrics['avg_win'] / metrics['avg_loss'])
            if metrics['avg_loss'] != 0 else np.inf
        )

        # ── Benchmark-Relative Metrics ──
        if benchmark_returns is not None:
            bench = benchmark_returns.reindex(returns.index).dropna()
            common = returns.index.intersection(bench.index)
            active = returns[common] - bench[common]

            metrics['active_return'] = active.mean() * ann_factor
            tracking_error = active.std() * np.sqrt(ann_factor)
            metrics['tracking_error'] = tracking_error
            metrics['information_ratio'] = (
                metrics['active_return'] / tracking_error
                if tracking_error > 0 else 0
            )

            # Beta and Alpha (CAPM)
            cov_matrix = np.cov(returns[common], bench[common])
            beta = cov_matrix[0, 1] / cov_matrix[1, 1] if cov_matrix[1, 1] > 0 else 1
            metrics['beta'] = beta
            metrics['alpha_capm'] = (
                metrics['annualized_return'] -
                self.rf - beta * (bench.mean() * ann_factor - self.rf)
            )

            # Capture ratios
            up_bench = bench[bench > 0]
            down_bench = bench[bench < 0]
            if len(up_bench) > 0:
                up_port = returns.reindex(up_bench.index)
                metrics['upside_capture'] = (
                    up_port.mean() / up_bench.mean()
                )
            if len(down_bench) > 0:
                down_port = returns.reindex(down_bench.index)
                metrics['downside_capture'] = (
                    down_port.mean() / down_bench.mean()
                )

        return metrics

    def compute_rolling(self, returns: pd.Series,
                        window: int = 252) -> pd.DataFrame:
        """
        Compute rolling performance metrics.

        Parameters
        ----------
        returns : pd.Series
            Daily returns.
        window : int
            Rolling window in trading days.

        Returns
        -------
        pd.DataFrame
            Rolling metrics over time.
        """
        daily_rf = self.rf / 252
        rolling = pd.DataFrame(index=returns.index)

        # Rolling return
        rolling['return'] = returns.rolling(window).mean() * 252

        # Rolling volatility
        rolling['volatility'] = returns.rolling(window).std() * np.sqrt(252)

        # Rolling Sharpe
        excess = returns - daily_rf
        rolling['sharpe'] = (
            excess.rolling(window).mean() /
            excess.rolling(window).std() * np.sqrt(252)
        )

        # Rolling drawdown
        cum = (1 + returns).cumprod()
        rolling_max = cum.rolling(window, min_periods=1).max()
        rolling['drawdown'] = cum / rolling_max - 1

        # Rolling skewness
        rolling['skewness'] = returns.rolling(window).skew()

        return rolling

    def generate_report(self, metrics: Dict) -> str:
        """Generate a formatted performance report string."""
        lines = [
            "=" * 60,
            "PERFORMANCE REPORT",
            "=" * 60,
            "",
            "── Return Metrics ──",
            f"  CAGR:                  {metrics.get('cagr', 0):.2%}",
            f"  Annualized Return:     {metrics.get('annualized_return', 0):.2%}",
            f"  Total Return:          {metrics.get('total_return', 0):.2%}",
            f"  Annualized Volatility: {metrics.get('annualized_volatility', 0):.2%}",
            "",
            "── Risk-Adjusted ──",
            f"  Sharpe Ratio:          {metrics.get('sharpe_ratio', 0):.3f}",
            f"  Sortino Ratio:         {metrics.get('sortino_ratio', 0):.3f}",
            f"  Calmar Ratio:          {metrics.get('calmar_ratio', 0):.3f}",
            f"  Information Ratio:     {metrics.get('information_ratio', 0):.3f}",
            "",
            "── Drawdown ──",
            f"  Max Drawdown:          {metrics.get('max_drawdown', 0):.2%}",
            f"  Max DD Duration:       {metrics.get('max_drawdown_duration_days', 0)} days",
            "",
            "── Tail Risk ──",
            f"  VaR (95%):             {metrics.get('var_95', 0):.4f}",
            f"  CVaR (95%):            {metrics.get('cvar_95', 0):.4f}",
            f"  Skewness:              {metrics.get('skewness', 0):.3f}",
            f"  Kurtosis:              {metrics.get('kurtosis', 0):.3f}",
            "",
            "── Win/Loss ──",
            f"  Hit Rate (Daily):      {metrics.get('hit_rate_daily', 0):.2%}",
            f"  Hit Rate (Monthly):    {metrics.get('hit_rate_monthly', 0):.2%}",
            f"  Profit Factor:         {metrics.get('profit_factor', 0):.3f}",
            "",
            "── Benchmark-Relative ──",
            f"  Beta:                  {metrics.get('beta', 'N/A')}",
            f"  CAPM Alpha:            {metrics.get('alpha_capm', 'N/A')}",
            f"  Tracking Error:        {metrics.get('tracking_error', 'N/A')}",
            "=" * 60,
        ]
        return "\n".join(lines)
