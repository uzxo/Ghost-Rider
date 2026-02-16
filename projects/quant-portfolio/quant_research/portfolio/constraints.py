"""
Portfolio Constraints
=====================
Sector neutrality, factor neutrality, beta targeting,
and volatility targeting.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class PortfolioConstraints:
    """Manage and enforce portfolio constraints."""

    def __init__(self, config: dict):
        self.config = config.get('constraints', config.get('portfolio', {}))

    def check_constraints(self, weights: pd.Series,
                          sector_map: Dict[str, str],
                          benchmark_sector_weights: Dict[str, float] = None,
                          betas: pd.Series = None,
                          factor_exposures: pd.DataFrame = None
                          ) -> dict:
        """
        Check all constraints and report violations.

        Returns
        -------
        dict
            Constraint check results with pass/fail for each.
        """
        results = {}

        # Position limits
        max_w = self.config.get('max_weight', 0.05)
        min_w = self.config.get('min_weight', 0.005)
        results['max_weight'] = {
            'passed': weights.max() <= max_w + 1e-6,
            'max_observed': weights.max(),
            'limit': max_w,
        }
        active = weights[weights > 0]
        results['min_weight'] = {
            'passed': active.min() >= min_w - 1e-6 if len(active) > 0 else True,
            'min_observed': active.min() if len(active) > 0 else 0,
            'limit': min_w,
        }

        # Sector neutrality
        if benchmark_sector_weights:
            max_dev = self.config.get('max_sector_deviation', 0.03)
            sector_weights = {}
            for ticker, weight in weights.items():
                sector = sector_map.get(ticker, 'Unknown')
                sector_weights[sector] = sector_weights.get(sector, 0) + weight

            max_sector_dev = 0
            for sector, port_w in sector_weights.items():
                bench_w = benchmark_sector_weights.get(sector, 0)
                dev = abs(port_w - bench_w)
                max_sector_dev = max(max_sector_dev, dev)

            results['sector_neutrality'] = {
                'passed': max_sector_dev <= max_dev + 1e-6,
                'max_deviation': max_sector_dev,
                'limit': max_dev,
                'sector_weights': sector_weights,
            }

        # Beta constraint
        if betas is not None:
            beta_range = self.config.get('beta_range', [0.90, 1.10])
            port_beta = (weights * betas.reindex(
                weights.index, fill_value=1.0
            )).sum()
            results['beta'] = {
                'passed': beta_range[0] <= port_beta <= beta_range[1],
                'portfolio_beta': port_beta,
                'range': beta_range,
            }

        # Number of holdings
        n_holdings = (weights > 0).sum()
        target_n = self.config.get('num_holdings', 40)
        results['num_holdings'] = {
            'value': n_holdings,
            'target': target_n,
        }

        return results

    def apply_volatility_targeting(self, weights: pd.Series,
                                    cov: pd.DataFrame,
                                    target_vol: float = 0.12
                                    ) -> pd.Series:
        """
        Scale portfolio weights to target a specific volatility.

        Parameters
        ----------
        weights : pd.Series
            Raw portfolio weights.
        cov : pd.DataFrame
            Covariance matrix.
        target_vol : float
            Target annualized volatility.

        Returns
        -------
        pd.Series
            Volatility-targeted weights.
        """
        common = weights.index.intersection(cov.index)
        w = weights[common].values
        Sigma = cov.loc[common, common].values

        port_vol = np.sqrt(w @ Sigma @ w) * np.sqrt(252)

        if port_vol > 0:
            scale = target_vol / port_vol
            weights = weights * scale
            logger.info(f"Vol targeting: {port_vol:.1%} -> "
                         f"{target_vol:.1%} (scale: {scale:.2f})")

        return weights
