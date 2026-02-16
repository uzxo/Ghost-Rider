"""
Signal Combination
==================
IC-weighted signal combination with orthogonalization
and turnover penalty.
"""

import logging
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class SignalCombiner:
    """
    Combine multiple alpha signals using information-coefficient
    weighting with orthogonalization.
    """

    def __init__(self, ic_lookback: int = 12):
        """
        Parameters
        ----------
        ic_lookback : int
            Number of months of trailing IC to use for weighting.
        """
        self.ic_lookback = ic_lookback
        self.ic_history: Dict[str, list] = {}
        self.weights: Dict[str, float] = {}

    def compute_ic(self, signal: pd.Series,
                   forward_returns: pd.Series) -> float:
        """
        Compute Information Coefficient (Spearman rank correlation).

        Parameters
        ----------
        signal : pd.Series
            Cross-sectional signal values.
        forward_returns : pd.Series
            Realized next-period returns.

        Returns
        -------
        float
            Spearman rank correlation.
        """
        common = signal.dropna().index.intersection(
            forward_returns.dropna().index
        )
        if len(common) < 10:
            return np.nan

        ic, _ = spearmanr(signal[common], forward_returns[common])
        return ic

    def update_ic_history(self, signal_name: str, ic: float):
        """Record IC for trailing average computation."""
        if signal_name not in self.ic_history:
            self.ic_history[signal_name] = []
        self.ic_history[signal_name].append(ic)
        # Keep only lookback window
        self.ic_history[signal_name] = (
            self.ic_history[signal_name][-self.ic_lookback:]
        )

    def compute_ic_weights(self) -> Dict[str, float]:
        """
        Compute signal weights proportional to trailing IC.

        Signals with negative IC get zero weight.
        """
        weights = {}
        for name, ic_list in self.ic_history.items():
            valid_ics = [ic for ic in ic_list if not np.isnan(ic)]
            if valid_ics:
                mean_ic = np.mean(valid_ics)
                weights[name] = max(mean_ic, 0)  # Floor at zero
            else:
                weights[name] = 0

        # Normalize to sum to 1
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}
        else:
            # Equal weight fallback
            n = len(weights)
            weights = {k: 1.0 / n for k in weights}

        self.weights = weights
        logger.info(f"IC-weighted signal weights: "
                     f"{ {k: round(v, 3) for k, v in weights.items()} }")
        return weights

    def orthogonalize(self, signals: pd.DataFrame) -> pd.DataFrame:
        """
        Orthogonalize signals to remove redundant information.

        Regress each signal on all others and use residuals.
        This ensures each signal contributes unique alpha.

        Parameters
        ----------
        signals : pd.DataFrame
            Signal values with columns as signal names.

        Returns
        -------
        pd.DataFrame
            Orthogonalized signals.
        """
        result = pd.DataFrame(index=signals.index)
        cols = signals.columns.tolist()

        for i, col in enumerate(cols):
            other_cols = [c for c in cols if c != col]
            if not other_cols:
                result[col] = signals[col]
                continue

            y = signals[col].values
            X = signals[other_cols].values

            # Handle NaN
            mask = ~np.isnan(y) & ~np.any(np.isnan(X), axis=1)
            if mask.sum() < 10:
                result[col] = signals[col]
                continue

            # OLS residuals
            try:
                X_clean = np.column_stack([np.ones(mask.sum()), X[mask]])
                betas = np.linalg.lstsq(X_clean, y[mask], rcond=None)[0]
                predicted = X_clean @ betas
                residuals = np.full(len(y), np.nan)
                residuals[mask] = y[mask] - predicted
                result[col] = residuals
            except np.linalg.LinAlgError:
                result[col] = signals[col]

        return result

    def combine(self, signals: pd.DataFrame,
                weights: Optional[Dict[str, float]] = None,
                orthogonalize: bool = True) -> pd.Series:
        """
        Combine signals into a composite alpha score.

        Parameters
        ----------
        signals : pd.DataFrame
            Normalized signal values.
        weights : dict, optional
            Signal weights. If None, uses IC-derived weights.
        orthogonalize : bool
            Whether to orthogonalize before combining.

        Returns
        -------
        pd.Series
            Composite alpha signal.
        """
        if weights is None:
            weights = self.weights

        if not weights:
            # Equal weight fallback
            weights = {col: 1.0 / len(signals.columns)
                       for col in signals.columns}

        if orthogonalize:
            signals = self.orthogonalize(signals)

        # Weighted combination
        composite = pd.Series(0.0, index=signals.index)
        for col, weight in weights.items():
            if col in signals.columns:
                composite += weight * signals[col].fillna(0)

        return composite

    def compute_ic_decay(self, signal: pd.Series,
                         returns_by_lag: Dict[int, pd.Series]
                         ) -> pd.Series:
        """
        Compute IC at different lags to determine signal decay.

        Parameters
        ----------
        signal : pd.Series
            Signal values.
        returns_by_lag : dict
            {lag_months: forward_returns} for lags 1, 2, ..., 12.

        Returns
        -------
        pd.Series
            IC at each lag.
        """
        decay = {}
        for lag, returns in returns_by_lag.items():
            ic = self.compute_ic(signal, returns)
            decay[lag] = ic

        return pd.Series(decay)

    def compute_turnover_penalty(self, signal_old: pd.Series,
                                 signal_new: pd.Series,
                                 cost_per_unit: float = 0.002
                                 ) -> float:
        """
        Estimate the transaction cost penalty from signal change.

        Parameters
        ----------
        signal_old : pd.Series
            Previous period's signal.
        signal_new : pd.Series
            Current period's signal.
        cost_per_unit : float
            Cost per unit of turnover (default 20 bps).

        Returns
        -------
        float
            Estimated cost penalty.
        """
        common = signal_old.index.intersection(signal_new.index)
        if len(common) == 0:
            return 0

        turnover = (signal_new[common] - signal_old[common]).abs().mean()
        return turnover * cost_per_unit
