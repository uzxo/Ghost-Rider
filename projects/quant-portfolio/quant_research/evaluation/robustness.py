"""
Robustness Testing Framework
==============================
Monte Carlo bootstrap, regime stress testing,
feature stability, and sensitivity analysis.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class RobustnessTests:
    """Comprehensive robustness testing suite."""

    def __init__(self, risk_free_rate: float = 0.04):
        self.rf = risk_free_rate

    # ──────────────────────────────────────────────────────
    # Monte Carlo Bootstrap
    # ──────────────────────────────────────────────────────
    def monte_carlo_bootstrap(self, returns: pd.Series,
                               benchmark_returns: pd.Series = None,
                               n_bootstrap: int = 10000,
                               block_size: int = 21,
                               confidence: float = 0.95) -> Dict:
        """
        Block bootstrap analysis of strategy performance.

        Parameters
        ----------
        returns : pd.Series
            Portfolio daily returns.
        benchmark_returns : pd.Series, optional
            Benchmark returns.
        n_bootstrap : int
            Number of bootstrap samples.
        block_size : int
            Block size for block bootstrap (preserves autocorrelation).
        confidence : float
            Confidence level for intervals.

        Returns
        -------
        dict
            Bootstrap distribution statistics.
        """
        ret = returns.dropna().values
        n = len(ret)
        n_blocks = n // block_size + 1

        sharpes = []
        cagrs = []
        max_dds = []

        for _ in range(n_bootstrap):
            # Block bootstrap
            block_starts = np.random.randint(0, n - block_size, n_blocks)
            sample = np.concatenate([
                ret[s:s + block_size] for s in block_starts
            ])[:n]

            # Compute metrics on bootstrapped sample
            ann_ret = np.mean(sample) * 252
            ann_vol = np.std(sample) * np.sqrt(252)
            sharpe = (
                (ann_ret - self.rf) / ann_vol if ann_vol > 0 else 0
            )
            sharpes.append(sharpe)

            cum = np.cumprod(1 + sample)
            running_max = np.maximum.accumulate(cum)
            dd = cum / running_max - 1
            max_dds.append(np.min(dd))

            n_years = n / 252
            cagr = cum[-1] ** (1 / max(n_years, 0.01)) - 1
            cagrs.append(cagr)

        sharpes = np.array(sharpes)
        cagrs = np.array(cagrs)
        max_dds = np.array(max_dds)

        alpha_half = (1 - confidence) / 2
        results = {
            'sharpe': {
                'mean': np.mean(sharpes),
                'median': np.median(sharpes),
                'std': np.std(sharpes),
                'ci_lower': np.percentile(sharpes, alpha_half * 100),
                'ci_upper': np.percentile(sharpes, (1 - alpha_half) * 100),
                'prob_positive': (sharpes > 0).mean(),
            },
            'cagr': {
                'mean': np.mean(cagrs),
                'median': np.median(cagrs),
                'ci_lower': np.percentile(cagrs, alpha_half * 100),
                'ci_upper': np.percentile(cagrs, (1 - alpha_half) * 100),
            },
            'max_drawdown': {
                'mean': np.mean(max_dds),
                'median': np.median(max_dds),
                'ci_lower': np.percentile(max_dds, alpha_half * 100),
                'ci_upper': np.percentile(max_dds, (1 - alpha_half) * 100),
            },
            'n_bootstrap': n_bootstrap,
        }

        # Probability of outperforming benchmark
        if benchmark_returns is not None:
            bench = benchmark_returns.dropna().values[:n]
            bench_sharpes = []
            for _ in range(min(n_bootstrap, 1000)):
                block_starts = np.random.randint(0, len(bench) - block_size,
                                                  n_blocks)
                sample = np.concatenate([
                    bench[s:s + block_size] for s in block_starts
                ])[:len(bench)]
                ann_ret = np.mean(sample) * 252
                ann_vol = np.std(sample) * np.sqrt(252)
                bench_sharpes.append(
                    (ann_ret - self.rf) / ann_vol if ann_vol > 0 else 0
                )

            results['prob_beat_benchmark'] = (
                sharpes[:len(bench_sharpes)] >
                np.array(bench_sharpes)
            ).mean()

        logger.info(
            f"Bootstrap Sharpe: {results['sharpe']['mean']:.3f} "
            f"[{results['sharpe']['ci_lower']:.3f}, "
            f"{results['sharpe']['ci_upper']:.3f}]"
        )
        return results

    # ──────────────────────────────────────────────────────
    # Regime Stress Testing
    # ──────────────────────────────────────────────────────
    def regime_stress_test(self, returns: pd.Series,
                            benchmark_returns: pd.Series = None
                            ) -> Dict:
        """
        Evaluate strategy during known stress periods.

        Returns performance metrics for each regime period.
        """
        stress_periods = {
            'COVID_Crash': ('2020-02-19', '2020-03-23'),
            'COVID_Recovery': ('2020-03-24', '2020-12-31'),
            'Rate_Hike_2022': ('2022-01-03', '2022-10-12'),
            'Q4_2018_Selloff': ('2018-10-01', '2018-12-24'),
            'Post_Election_2024': ('2024-11-01', '2024-12-31'),
            'Full_Sample': (
                returns.index[0].strftime('%Y-%m-%d'),
                returns.index[-1].strftime('%Y-%m-%d')
            ),
        }

        results = {}
        for name, (start, end) in stress_periods.items():
            try:
                period_ret = returns.loc[start:end]
                if len(period_ret) < 5:
                    continue

                cum_ret = (1 + period_ret).prod() - 1
                ann_vol = period_ret.std() * np.sqrt(252)
                max_dd = (
                    (1 + period_ret).cumprod() /
                    (1 + period_ret).cumprod().expanding().max() - 1
                ).min()

                result = {
                    'cum_return': cum_ret,
                    'ann_volatility': ann_vol,
                    'max_drawdown': max_dd,
                    'n_days': len(period_ret),
                    'hit_rate': (period_ret > 0).mean(),
                }

                if benchmark_returns is not None:
                    bench_period = benchmark_returns.loc[start:end]
                    if len(bench_period) > 0:
                        bench_cum = (1 + bench_period).prod() - 1
                        result['benchmark_return'] = bench_cum
                        result['excess_return'] = cum_ret - bench_cum

                results[name] = result
            except Exception as e:
                logger.debug(f"Stress test {name} skipped: {e}")

        return results

    # ──────────────────────────────────────────────────────
    # Sensitivity Analysis
    # ──────────────────────────────────────────────────────
    def weight_sensitivity(self, signal_func,
                            base_weights: Dict[str, float],
                            returns_data,
                            n_trials: int = 500,
                            perturbation: float = 0.20) -> Dict:
        """
        Test sensitivity to signal combination weights.

        Perturb weights by +/-perturbation and measure
        Sharpe ratio distribution.
        """
        base_sharpes = []

        for _ in range(n_trials):
            perturbed = {}
            for name, w in base_weights.items():
                noise = np.random.uniform(-perturbation, perturbation)
                perturbed[name] = max(w * (1 + noise), 0)

            # Normalize
            total = sum(perturbed.values())
            if total > 0:
                perturbed = {k: v / total for k, v in perturbed.items()}

            try:
                ret = signal_func(perturbed)
                ann_ret = ret.mean() * 252
                ann_vol = ret.std() * np.sqrt(252)
                sharpe = (ann_ret - self.rf) / ann_vol if ann_vol > 0 else 0
                base_sharpes.append(sharpe)
            except Exception:
                continue

        return {
            'sharpe_mean': np.mean(base_sharpes),
            'sharpe_std': np.std(base_sharpes),
            'sharpe_min': np.min(base_sharpes),
            'sharpe_max': np.max(base_sharpes),
            'n_trials': len(base_sharpes),
            'fragile': np.std(base_sharpes) > 0.3,
        }

    # ──────────────────────────────────────────────────────
    # Holdout Validation
    # ──────────────────────────────────────────────────────
    def holdout_test(self, in_sample_sharpe: float,
                      holdout_returns: pd.Series) -> Dict:
        """
        Test for overfitting via holdout comparison.

        If holdout Sharpe < 50% of in-sample Sharpe, strategy is overfit.
        """
        ret = holdout_returns.dropna()
        ann_ret = ret.mean() * 252
        ann_vol = ret.std() * np.sqrt(252)
        holdout_sharpe = (
            (ann_ret - self.rf) / ann_vol if ann_vol > 0 else 0
        )

        ratio = (
            holdout_sharpe / in_sample_sharpe
            if in_sample_sharpe != 0 else 0
        )

        return {
            'in_sample_sharpe': in_sample_sharpe,
            'holdout_sharpe': holdout_sharpe,
            'ratio': ratio,
            'overfit': ratio < 0.50,
            'assessment': (
                'PASS — Holdout performance consistent'
                if ratio >= 0.50
                else 'FAIL — Likely overfit (holdout < 50% of IS)'
            )
        }

    def generate_report(self, bootstrap: Dict = None,
                         stress: Dict = None,
                         holdout: Dict = None) -> str:
        """Generate formatted robustness report."""
        lines = [
            "=" * 60,
            "ROBUSTNESS TESTING REPORT",
            "=" * 60,
        ]

        if bootstrap:
            s = bootstrap['sharpe']
            lines.extend([
                "",
                "── Monte Carlo Bootstrap ──",
                f"  Sharpe: {s['mean']:.3f} [{s['ci_lower']:.3f}, {s['ci_upper']:.3f}]",
                f"  P(Sharpe > 0): {s['prob_positive']:.1%}",
                f"  CAGR: {bootstrap['cagr']['mean']:.2%} "
                f"[{bootstrap['cagr']['ci_lower']:.2%}, {bootstrap['cagr']['ci_upper']:.2%}]",
            ])
            if 'prob_beat_benchmark' in bootstrap:
                lines.append(
                    f"  P(Beat Benchmark): {bootstrap['prob_beat_benchmark']:.1%}"
                )

        if stress:
            lines.extend(["", "── Regime Stress Tests ──"])
            for name, data in stress.items():
                excess = data.get('excess_return', 'N/A')
                if isinstance(excess, float):
                    excess = f"{excess:+.2%}"
                lines.append(
                    f"  {name:25s}: Return={data['cum_return']:+.2%}  "
                    f"DD={data['max_drawdown']:.2%}  Excess={excess}"
                )

        if holdout:
            lines.extend([
                "",
                "── Holdout Validation ──",
                f"  In-Sample Sharpe:  {holdout['in_sample_sharpe']:.3f}",
                f"  Holdout Sharpe:    {holdout['holdout_sharpe']:.3f}",
                f"  Ratio:             {holdout['ratio']:.2%}",
                f"  Assessment:        {holdout['assessment']}",
            ])

        lines.append("=" * 60)
        return "\n".join(lines)
