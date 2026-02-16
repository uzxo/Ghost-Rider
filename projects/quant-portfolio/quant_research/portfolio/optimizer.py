"""
Portfolio Optimizer
===================
Multiple portfolio construction methods:
- Black-Litterman
- Hierarchical Risk Parity (HRP)
- Risk Parity
- CVaR Optimization
- Mean-Variance (Ledoit-Wolf)
"""

import logging
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class PortfolioOptimizer:
    """Multi-method portfolio optimizer."""

    def __init__(self, config: dict):
        self.config = config
        self.method = config.get('optimizer', 'hrp')

    def optimize(self, expected_returns: pd.Series,
                 cov_matrix: pd.DataFrame,
                 constraints: dict = None,
                 current_weights: pd.Series = None) -> pd.Series:
        """
        Optimize portfolio weights.

        Parameters
        ----------
        expected_returns : pd.Series
            Expected returns for each asset.
        cov_matrix : pd.DataFrame
            Covariance matrix.
        constraints : dict
            Portfolio constraints.
        current_weights : pd.Series, optional
            Current weights (for turnover constraint).

        Returns
        -------
        pd.Series
            Optimal portfolio weights.
        """
        method = self.method.lower()

        if method == 'black_litterman':
            weights = self._black_litterman(
                expected_returns, cov_matrix, constraints
            )
        elif method == 'hrp':
            weights = self._hrp(cov_matrix, expected_returns)
        elif method == 'risk_parity':
            weights = self._risk_parity(cov_matrix, expected_returns)
        elif method == 'cvar':
            weights = self._cvar_optimization(
                expected_returns, cov_matrix, constraints
            )
        elif method == 'mean_variance':
            weights = self._mean_variance(
                expected_returns, cov_matrix, constraints
            )
        else:
            weights = self._hrp(cov_matrix, expected_returns)

        # Apply constraints
        if constraints:
            weights = self._apply_constraints(
                weights, constraints, current_weights
            )

        return weights

    def estimate_covariance(self, returns: pd.DataFrame,
                             method: str = 'ledoit_wolf'
                             ) -> pd.DataFrame:
        """
        Estimate the covariance matrix.

        Parameters
        ----------
        returns : pd.DataFrame
            Historical returns.
        method : str
            Estimation method.

        Returns
        -------
        pd.DataFrame
            Estimated covariance matrix.
        """
        if method == 'ledoit_wolf':
            return self._ledoit_wolf(returns)
        elif method == 'ewma':
            halflife = self.config.get('covariance', {}).get(
                'ewma_halflife', 60
            )
            return self._ewma_covariance(returns, halflife)
        else:
            # Sample covariance
            return returns.cov()

    def _ledoit_wolf(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Ledoit-Wolf shrinkage estimator."""
        try:
            from sklearn.covariance import LedoitWolf
            lw = LedoitWolf()
            lw.fit(returns.dropna())
            cov = pd.DataFrame(
                lw.covariance_,
                index=returns.columns,
                columns=returns.columns
            )
            logger.info(f"Ledoit-Wolf shrinkage: {lw.shrinkage_:.4f}")
            return cov
        except Exception as e:
            logger.warning(f"Ledoit-Wolf failed: {e}; using sample covariance")
            return returns.cov()

    def _ewma_covariance(self, returns: pd.DataFrame,
                          halflife: int = 60) -> pd.DataFrame:
        """Exponentially-weighted moving average covariance."""
        return returns.ewm(halflife=halflife).cov().iloc[-len(returns.columns):]

    # ──────────────────────────────────────────────────────
    # Black-Litterman
    # ──────────────────────────────────────────────────────
    def _black_litterman(self, views: pd.Series,
                          cov: pd.DataFrame,
                          constraints: dict = None) -> pd.Series:
        """
        Black-Litterman portfolio optimization.

        Parameters
        ----------
        views : pd.Series
            Model-generated alpha views (excess return forecasts).
        cov : pd.DataFrame
            Covariance matrix.
        """
        bl_config = self.config.get('black_litterman', {})
        tau = bl_config.get('tau', 0.05)

        assets = cov.index
        n = len(assets)

        # Market-cap equilibrium prior (assume equal for simplicity)
        # In production, use actual market-cap weights
        w_mkt = pd.Series(1.0 / n, index=assets)

        # Risk aversion
        sigma_mkt = float(np.sqrt(w_mkt @ cov @ w_mkt))
        risk_aversion = 0.10 / (sigma_mkt ** 2)  # Target ~10% return

        # Equilibrium returns
        pi = risk_aversion * cov @ w_mkt

        # Views (P matrix = identity for absolute views)
        P = np.eye(n)
        Q = views.reindex(assets).fillna(0).values

        # Omega (uncertainty on views) — proportional to variance
        omega = np.diag(np.diag(tau * P @ cov.values @ P.T))

        # Black-Litterman posterior
        Sigma = cov.values
        inv_tau_sigma = np.linalg.inv(tau * Sigma)
        inv_omega = np.linalg.inv(omega)

        # Posterior expected returns
        posterior_cov = np.linalg.inv(inv_tau_sigma + P.T @ inv_omega @ P)
        posterior_mean = posterior_cov @ (
            inv_tau_sigma @ pi.values + P.T @ inv_omega @ Q
        )

        # Optimal weights from posterior
        w = risk_aversion * np.linalg.inv(Sigma) @ posterior_mean

        weights = pd.Series(w, index=assets)
        # Normalize for long-only
        weights = weights.clip(lower=0)
        if weights.sum() > 0:
            weights = weights / weights.sum()

        return weights

    # ──────────────────────────────────────────────────────
    # Hierarchical Risk Parity (HRP)
    # ──────────────────────────────────────────────────────
    def _hrp(self, cov: pd.DataFrame,
             alpha_signal: pd.Series = None) -> pd.Series:
        """
        Hierarchical Risk Parity (Lopez de Prado).

        Does not require covariance matrix inversion.
        """
        assets = cov.index.tolist()
        n = len(assets)

        # Correlation matrix
        std = np.sqrt(np.diag(cov.values))
        corr = cov.values / np.outer(std, std)
        corr = np.clip(corr, -1, 1)
        np.fill_diagonal(corr, 1.0)

        # Distance matrix
        dist = np.sqrt(0.5 * (1 - corr))
        np.fill_diagonal(dist, 0)

        # Hierarchical clustering
        condensed = squareform(dist, checks=False)
        link = linkage(condensed, method='single')

        # Quasi-diagonalization
        sort_idx = leaves_list(link).tolist()
        sorted_assets = [assets[i] for i in sort_idx]

        # Recursive bisection
        weights = self._recursive_bisection(
            cov, sorted_assets
        )

        # Optional alpha tilt
        if alpha_signal is not None:
            rp_config = self.config.get('risk_parity', {})
            if rp_config.get('alpha_tilt', False):
                weights = self._alpha_tilt(
                    weights, alpha_signal,
                    tilt_budget=rp_config.get('tilt_budget', 0.03)
                )

        return weights

    def _recursive_bisection(self, cov: pd.DataFrame,
                              sorted_assets: list) -> pd.Series:
        """Recursive bisection for HRP."""
        weights = pd.Series(1.0, index=sorted_assets)

        clusters = [sorted_assets]
        while clusters:
            new_clusters = []
            for cluster in clusters:
                if len(cluster) <= 1:
                    continue

                mid = len(cluster) // 2
                left = cluster[:mid]
                right = cluster[mid:]

                # Cluster variance
                cov_left = cov.loc[left, left]
                cov_right = cov.loc[right, right]

                # Inverse-variance allocation
                inv_var_left = 1.0 / self._cluster_variance(cov_left)
                inv_var_right = 1.0 / self._cluster_variance(cov_right)
                total = inv_var_left + inv_var_right

                alpha_left = inv_var_left / total
                alpha_right = 1 - alpha_left

                weights[left] *= alpha_left
                weights[right] *= alpha_right

                new_clusters.extend([left, right])

            clusters = [c for c in new_clusters if len(c) > 1]

        return weights / weights.sum()

    @staticmethod
    def _cluster_variance(cov: pd.DataFrame) -> float:
        """Compute cluster variance using inverse-variance weights."""
        ivp = 1.0 / np.diag(cov.values)
        ivp /= ivp.sum()
        return float(ivp @ cov.values @ ivp)

    # ──────────────────────────────────────────────────────
    # Risk Parity (Equal Risk Contribution)
    # ──────────────────────────────────────────────────────
    def _risk_parity(self, cov: pd.DataFrame,
                      alpha_signal: pd.Series = None) -> pd.Series:
        """
        Equal Risk Contribution portfolio.
        Each position contributes equally to total portfolio risk.
        """
        assets = cov.index
        n = len(assets)
        Sigma = cov.values

        # Target: equal risk contribution
        target_rc = np.ones(n) / n

        def objective(w):
            w = np.abs(w)
            port_var = w @ Sigma @ w
            if port_var <= 0:
                return 1e10
            marginal_contrib = Sigma @ w
            risk_contrib = w * marginal_contrib / np.sqrt(port_var)
            rc_pct = risk_contrib / risk_contrib.sum()
            return np.sum((rc_pct - target_rc) ** 2)

        w0 = np.ones(n) / n
        bounds = [(0.001, 1.0) for _ in range(n)]
        constraints_list = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]

        result = minimize(objective, w0, method='SLSQP',
                          bounds=bounds, constraints=constraints_list,
                          options={'maxiter': 1000})

        weights = pd.Series(np.abs(result.x), index=assets)
        weights = weights / weights.sum()

        # Alpha tilt
        if alpha_signal is not None:
            rp_config = self.config.get('risk_parity', {})
            if rp_config.get('alpha_tilt', False):
                weights = self._alpha_tilt(
                    weights, alpha_signal,
                    tilt_budget=rp_config.get('tilt_budget', 0.03)
                )

        return weights

    # ──────────────────────────────────────────────────────
    # CVaR Optimization
    # ──────────────────────────────────────────────────────
    def _cvar_optimization(self, expected_returns: pd.Series,
                            cov: pd.DataFrame,
                            constraints: dict = None) -> pd.Series:
        """
        Minimize Conditional Value-at-Risk (Expected Shortfall).
        """
        cvar_config = self.config.get('cvar', {})
        confidence = cvar_config.get('confidence_level', 0.95)
        n = len(cov)
        assets = cov.index

        # Generate scenarios via Monte Carlo
        n_sim = cvar_config.get('n_simulations', 5000)
        mu = expected_returns.reindex(assets).fillna(0).values
        scenarios = np.random.multivariate_normal(mu, cov.values, n_sim)

        def cvar_objective(w):
            port_returns = scenarios @ w
            var_threshold = np.percentile(port_returns,
                                          (1 - confidence) * 100)
            tail_returns = port_returns[port_returns <= var_threshold]
            if len(tail_returns) == 0:
                return 0
            return -np.mean(tail_returns)  # Minimize negative tail

        w0 = np.ones(n) / n
        bounds = [(0, 0.1) for _ in range(n)]
        constraints_list = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]

        result = minimize(cvar_objective, w0, method='SLSQP',
                          bounds=bounds, constraints=constraints_list)

        weights = pd.Series(result.x, index=assets)
        weights = weights.clip(lower=0)
        if weights.sum() > 0:
            weights = weights / weights.sum()
        return weights

    # ──────────────────────────────────────────────────────
    # Mean-Variance
    # ──────────────────────────────────────────────────────
    def _mean_variance(self, expected_returns: pd.Series,
                        cov: pd.DataFrame,
                        constraints: dict = None) -> pd.Series:
        """Maximum Sharpe ratio portfolio."""
        n = len(cov)
        assets = cov.index
        mu = expected_returns.reindex(assets).fillna(0).values
        Sigma = cov.values

        def neg_sharpe(w):
            port_ret = w @ mu
            port_vol = np.sqrt(w @ Sigma @ w)
            if port_vol <= 0:
                return 1e10
            return -port_ret / port_vol

        w0 = np.ones(n) / n
        max_w = constraints.get('max_weight', 0.1) if constraints else 0.1
        bounds = [(0, max_w) for _ in range(n)]
        constraints_list = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]

        result = minimize(neg_sharpe, w0, method='SLSQP',
                          bounds=bounds, constraints=constraints_list)

        weights = pd.Series(result.x, index=assets)
        weights = weights.clip(lower=0)
        if weights.sum() > 0:
            weights = weights / weights.sum()
        return weights

    # ──────────────────────────────────────────────────────
    # Constraints & Adjustments
    # ──────────────────────────────────────────────────────
    def _apply_constraints(self, weights: pd.Series,
                            constraints: dict,
                            current_weights: pd.Series = None
                            ) -> pd.Series:
        """Apply portfolio constraints to raw weights."""
        max_w = constraints.get('max_weight', 0.05)
        min_w = constraints.get('min_weight', 0.005)

        # Position limits
        weights = weights.clip(lower=0, upper=max_w)

        # Remove tiny positions
        weights[weights < min_w] = 0

        # Renormalize
        if weights.sum() > 0:
            weights = weights / weights.sum()

        # Turnover constraint
        if current_weights is not None:
            max_turnover = constraints.get('max_turnover_oneway', 0.30)
            turnover = (weights - current_weights.reindex(
                weights.index, fill_value=0
            )).abs().sum() / 2

            if turnover > max_turnover:
                # Blend old and new weights
                blend = max_turnover / max(turnover, 1e-8)
                current = current_weights.reindex(
                    weights.index, fill_value=0
                )
                weights = blend * weights + (1 - blend) * current
                weights = weights.clip(lower=0)
                if weights.sum() > 0:
                    weights = weights / weights.sum()

        return weights

    def _alpha_tilt(self, base_weights: pd.Series,
                     alpha_signal: pd.Series,
                     tilt_budget: float = 0.03) -> pd.Series:
        """Tilt risk parity weights toward high-alpha names."""
        alpha = alpha_signal.reindex(base_weights.index).fillna(0)
        alpha_norm = alpha / alpha.abs().sum() if alpha.abs().sum() > 0 else alpha

        tilted = base_weights + tilt_budget * alpha_norm
        tilted = tilted.clip(lower=0)
        if tilted.sum() > 0:
            tilted = tilted / tilted.sum()

        return tilted
