"""
Macro Feature Engineering
=========================
Transforms raw macro data into model-ready features.
"""

import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class MacroFeatures:
    """Transform raw macro data into model features."""

    def compute(self, macro_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute macro features from raw macro data.

        Parameters
        ----------
        macro_df : pd.DataFrame
            Raw macro data from MacroLoader.

        Returns
        -------
        pd.DataFrame
            Macro features indexed by date.
        """
        result = macro_df.copy()

        # ── Z-scores of levels (rolling 1-year window) ──
        z_cols = ['vix', 'dxy', 'tnx']
        for col in z_cols:
            if col in result.columns:
                roll_mean = result[col].rolling(252).mean()
                roll_std = result[col].rolling(252).std()
                result[f'{col}_zscore_1y'] = (result[col] - roll_mean) / roll_std

        # ── Yield curve features ──
        if 'yield_curve_2s10s' in result.columns:
            result['yield_curve_inverted'] = (
                result['yield_curve_2s10s'] < 0
            ).astype(float)
            result['yield_curve_zscore'] = (
                (result['yield_curve_2s10s'] -
                 result['yield_curve_2s10s'].rolling(252).mean()) /
                result['yield_curve_2s10s'].rolling(252).std()
            )

        # ── Credit conditions ──
        if 'credit_spread_hy' in result.columns:
            result['credit_spread_zscore'] = (
                (result['credit_spread_hy'] -
                 result['credit_spread_hy'].rolling(252).mean()) /
                result['credit_spread_hy'].rolling(252).std()
            )
            result['credit_tightening'] = (
                result['credit_spread_hy'].diff(21) > 0
            ).astype(float)

        # ── Regime classification ──
        result['regime'] = self._classify_regime(result)

        return result

    def _classify_regime(self, df: pd.DataFrame) -> pd.Series:
        """
        Simple rule-based regime classification.

        Regimes:
        0 = Bull / Low Vol
        1 = Bull / High Vol
        2 = Bear / Low Vol
        3 = Bear / High Vol
        """
        regime = pd.Series(0, index=df.index)

        # Market direction
        is_bull = pd.Series(True, index=df.index)
        if 'sp500_ret_63d' in df.columns:
            is_bull = df['sp500_ret_63d'] > 0
        elif 'sp500_above_200ma' in df.columns:
            is_bull = df['sp500_above_200ma'] > 0.5

        # Volatility level
        is_high_vol = pd.Series(False, index=df.index)
        if 'vix' in df.columns:
            is_high_vol = df['vix'] > 20

        regime = np.where(
            is_bull & ~is_high_vol, 0,
            np.where(
                is_bull & is_high_vol, 1,
                np.where(
                    ~is_bull & ~is_high_vol, 2, 3
                )
            )
        )
        return pd.Series(regime, index=df.index)

    def get_regime_description(self, regime_id: int) -> str:
        """Get human-readable regime description."""
        descriptions = {
            0: "Bull / Low Volatility",
            1: "Bull / High Volatility",
            2: "Bear / Low Volatility",
            3: "Bear / High Volatility",
        }
        return descriptions.get(regime_id, "Unknown")
