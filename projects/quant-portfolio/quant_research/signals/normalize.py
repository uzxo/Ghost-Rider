"""
Signal Normalization
====================
Cross-sectional normalization to make signals comparable
across time and across different alpha sources.
"""

import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class SignalNormalizer:
    """
    Normalize raw signals into comparable, distribution-controlled scores.
    """

    @staticmethod
    def winsorize(signal: pd.Series,
                  lower_pct: float = 1.0,
                  upper_pct: float = 99.0) -> pd.Series:
        """
        Winsorize signal at specified percentiles.

        Parameters
        ----------
        signal : pd.Series
            Raw signal values.
        lower_pct, upper_pct : float
            Percentile bounds for winsorization.

        Returns
        -------
        pd.Series
            Winsorized signal.
        """
        lo = np.nanpercentile(signal.dropna(), lower_pct)
        hi = np.nanpercentile(signal.dropna(), upper_pct)
        return signal.clip(lo, hi)

    @staticmethod
    def zscore(signal: pd.Series) -> pd.Series:
        """
        Cross-sectional z-score normalization.

        z_i = (signal_i - mean) / std
        """
        mean = signal.mean()
        std = signal.std()
        if std == 0 or pd.isna(std):
            return pd.Series(0.0, index=signal.index)
        return (signal - mean) / std

    @staticmethod
    def rank_normalize(signal: pd.Series) -> pd.Series:
        """
        Convert to percentile rank, mapped to [-1, +1].

        More robust to outliers than z-scoring.
        """
        n = signal.notna().sum()
        if n == 0:
            return pd.Series(0.0, index=signal.index)
        ranks = signal.rank(pct=True, na_option='keep')
        return ranks * 2 - 1  # Map [0, 1] to [-1, 1]

    @classmethod
    def normalize_cross_section(cls, signals: pd.DataFrame,
                                 method: str = 'rank',
                                 winsorize_bounds: tuple = (1, 99)
                                 ) -> pd.DataFrame:
        """
        Normalize all signals cross-sectionally.

        Parameters
        ----------
        signals : pd.DataFrame
            Columns are signal names, rows are securities.
        method : str
            'zscore' or 'rank'.
        winsorize_bounds : tuple
            (lower_pct, upper_pct) for winsorization.

        Returns
        -------
        pd.DataFrame
            Normalized signals.
        """
        result = pd.DataFrame(index=signals.index)

        for col in signals.columns:
            raw = signals[col]
            # Winsorize first
            winsorized = cls.winsorize(raw, *winsorize_bounds)
            # Then normalize
            if method == 'zscore':
                result[col] = cls.zscore(winsorized)
            elif method == 'rank':
                result[col] = cls.rank_normalize(winsorized)
            else:
                result[col] = winsorized

        return result

    @staticmethod
    def normalize_panel(panel: pd.DataFrame,
                        date_col: str = 'date',
                        method: str = 'rank') -> pd.DataFrame:
        """
        Normalize signals cross-sectionally at each date.

        Parameters
        ----------
        panel : pd.DataFrame
            Panel data with date column and signal columns.
        date_col : str
            Name of the date column.
        method : str
            Normalization method.

        Returns
        -------
        pd.DataFrame
            Panel with normalized signals.
        """
        signal_cols = [c for c in panel.columns
                       if c not in [date_col, 'ticker']]

        result = panel.copy()
        for date, group in panel.groupby(date_col):
            for col in signal_cols:
                if col in group.columns:
                    raw = group[col]
                    if method == 'rank':
                        normalized = SignalNormalizer.rank_normalize(raw)
                    else:
                        normalized = SignalNormalizer.zscore(
                            SignalNormalizer.winsorize(raw)
                        )
                    result.loc[group.index, col] = normalized

        return result
