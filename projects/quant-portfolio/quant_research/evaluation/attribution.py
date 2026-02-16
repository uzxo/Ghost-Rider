"""
Factor Attribution
==================
Decompose portfolio returns into factor contributions
and estimate alpha significance.
"""

import logging
import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class FactorAttribution:
    """
    Factor-based return attribution via time-series regression.

    Decomposes returns using Fama-French 5-factor + momentum model.
    """

    def __init__(self, risk_free_rate: float = 0.04):
        self.rf = risk_free_rate

    def attribute(self, portfolio_returns: pd.Series,
                   factor_returns: pd.DataFrame) -> Dict:
        """
        Run factor attribution regression.

        r_p - r_f = alpha + sum(beta_i * factor_i) + epsilon

        Parameters
        ----------
        portfolio_returns : pd.Series
            Daily portfolio returns.
        factor_returns : pd.DataFrame
            Daily factor returns (in percentage).

        Returns
        -------
        dict
            Regression results including alpha, betas, t-stats, R-squared.
        """
        # Align dates
        common = portfolio_returns.index.intersection(factor_returns.index)
        if len(common) < 60:
            logger.warning("Insufficient data for factor attribution")
            return {}

        y = portfolio_returns[common].values
        rf = factor_returns['RF'].reindex(common).fillna(0).values / 100
        y_excess = y - rf

        factor_cols = [c for c in factor_returns.columns if c != 'RF']
        X = factor_returns[factor_cols].reindex(common).fillna(0).values / 100
        X_const = np.column_stack([np.ones(len(y_excess)), X])

        # OLS regression
        try:
            betas, residuals, rank, sv = np.linalg.lstsq(
                X_const, y_excess, rcond=None
            )
        except np.linalg.LinAlgError:
            logger.error("Factor regression failed")
            return {}

        # Predictions and residuals
        y_pred = X_const @ betas
        resid = y_excess - y_pred
        n, k = X_const.shape

        # R-squared
        ss_res = np.sum(resid ** 2)
        ss_tot = np.sum((y_excess - y_excess.mean()) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        adj_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - k - 1)

        # Standard errors (heteroskedasticity-robust)
        mse = ss_res / (n - k)
        var_betas = mse * np.linalg.inv(X_const.T @ X_const)
        se = np.sqrt(np.diag(var_betas))

        # T-statistics and p-values
        t_stats = betas / se
        p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), n - k))

        # Build results
        results = {
            'alpha_annualized': betas[0] * 252,
            'alpha_daily': betas[0],
            'alpha_t_stat': t_stats[0],
            'alpha_p_value': p_values[0],
            'alpha_significant': p_values[0] < 0.05,
            'r_squared': r_squared,
            'adj_r_squared': adj_r_squared,
            'n_observations': n,
            'factor_betas': {},
            'factor_t_stats': {},
            'factor_p_values': {},
        }

        for i, col in enumerate(factor_cols):
            results['factor_betas'][col] = betas[i + 1]
            results['factor_t_stats'][col] = t_stats[i + 1]
            results['factor_p_values'][col] = p_values[i + 1]

        # Log summary
        logger.info(
            f"Factor Attribution: Alpha={results['alpha_annualized']:.2%} "
            f"(t={results['alpha_t_stat']:.2f}), "
            f"R²={r_squared:.3f}"
        )
        for col in factor_cols:
            logger.info(
                f"  {col}: beta={results['factor_betas'][col]:.3f} "
                f"(t={results['factor_t_stats'][col]:.2f})"
            )

        return results

    def rolling_attribution(self, portfolio_returns: pd.Series,
                              factor_returns: pd.DataFrame,
                              window: int = 252) -> pd.DataFrame:
        """
        Compute rolling factor attribution.

        Returns DataFrame with rolling alpha and betas.
        """
        common = portfolio_returns.index.intersection(factor_returns.index)
        y = portfolio_returns[common]
        rf = factor_returns['RF'].reindex(common).fillna(0) / 100
        y_excess = y - rf

        factor_cols = [c for c in factor_returns.columns if c != 'RF']
        X = factor_returns[factor_cols].reindex(common).fillna(0) / 100

        results = {}
        for i in range(window, len(y_excess)):
            date = y_excess.index[i]
            y_win = y_excess.iloc[i - window:i].values
            X_win = X.iloc[i - window:i].values
            X_const = np.column_stack([np.ones(window), X_win])

            try:
                betas = np.linalg.lstsq(X_const, y_win, rcond=None)[0]
                row = {'alpha': betas[0] * 252}
                for j, col in enumerate(factor_cols):
                    row[f'beta_{col}'] = betas[j + 1]
                results[date] = row
            except np.linalg.LinAlgError:
                continue

        return pd.DataFrame(results).T

    def generate_report(self, results: Dict) -> str:
        """Generate a formatted factor attribution report."""
        lines = [
            "=" * 60,
            "FACTOR ATTRIBUTION REPORT",
            "=" * 60,
            "",
            f"Alpha (annualized): {results.get('alpha_annualized', 0):.2%}",
            f"Alpha t-statistic:  {results.get('alpha_t_stat', 0):.2f}",
            f"Alpha p-value:      {results.get('alpha_p_value', 0):.4f}",
            f"Significant (5%):   {'YES' if results.get('alpha_significant') else 'NO'}",
            f"R-squared:          {results.get('r_squared', 0):.3f}",
            f"Adj R-squared:      {results.get('adj_r_squared', 0):.3f}",
            f"Observations:       {results.get('n_observations', 0)}",
            "",
            "Factor Exposures:",
        ]
        for factor, beta in results.get('factor_betas', {}).items():
            t = results['factor_t_stats'].get(factor, 0)
            lines.append(f"  {factor:10s}: beta={beta:+.3f}  (t={t:.2f})")

        lines.append("=" * 60)
        return "\n".join(lines)
