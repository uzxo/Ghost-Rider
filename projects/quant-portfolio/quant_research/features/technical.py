"""
Technical Feature Engineering
=============================
Comprehensive technical indicators computed from OHLCV data.
All features are computed using only past data (no look-ahead).
"""

import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class TechnicalFeatures:
    """Compute technical indicators from price/volume data."""

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all technical features for a single ticker.

        Parameters
        ----------
        df : pd.DataFrame
            OHLCV data with columns: Open, High, Low, Close, Volume

        Returns
        -------
        pd.DataFrame
            Original data plus technical feature columns.
        """
        result = df.copy()

        # ── Price Momentum ──
        for period in [5, 10, 21, 63, 126, 252]:
            result[f'ret_{period}d'] = result['Close'].pct_change(period)

        # 12-1 month momentum (skip most recent month)
        result['momentum_12_1'] = (
            result['Close'].pct_change(252) -
            result['Close'].pct_change(21)
        )

        # ── Moving Averages ──
        for window in [5, 10, 20, 50, 200]:
            ma = result['Close'].rolling(window).mean()
            result[f'ma_{window}'] = ma
            result[f'ma_{window}_dist'] = (
                (result['Close'] - ma) / ma
            )

        # MA crossover signals
        result['ma_cross_5_20'] = (
            result['ma_5'] > result['ma_20']
        ).astype(float)
        result['ma_cross_50_200'] = (
            result['ma_50'] > result['ma_200']
        ).astype(float)

        # ── RSI ──
        result['rsi_14'] = self._rsi(result['Close'], 14)
        result['rsi_5'] = self._rsi(result['Close'], 5)

        # ── MACD ──
        ema12 = result['Close'].ewm(span=12, adjust=False).mean()
        ema26 = result['Close'].ewm(span=26, adjust=False).mean()
        result['macd'] = ema12 - ema26
        result['macd_signal'] = result['macd'].ewm(span=9, adjust=False).mean()
        result['macd_hist'] = result['macd'] - result['macd_signal']

        # ── Bollinger Bands ──
        bb_ma = result['Close'].rolling(20).mean()
        bb_std = result['Close'].rolling(20).std()
        result['bb_upper'] = bb_ma + 2 * bb_std
        result['bb_lower'] = bb_ma - 2 * bb_std
        result['bb_width'] = (result['bb_upper'] - result['bb_lower']) / bb_ma
        result['bb_position'] = (
            (result['Close'] - result['bb_lower']) /
            (result['bb_upper'] - result['bb_lower'])
        )

        # ── Volatility ──
        for window in [10, 21, 63]:
            result[f'vol_{window}d'] = (
                result['Close'].pct_change().rolling(window).std() *
                np.sqrt(252)
            )

        # Volatility ratio (short/long)
        result['vol_ratio'] = result['vol_10d'] / result['vol_63d']

        # ── ATR (Average True Range) ──
        result['atr_14'] = self._atr(
            result['High'], result['Low'], result['Close'], 14
        )
        result['atr_pct'] = result['atr_14'] / result['Close']

        # ── Volume Features ──
        result['volume_ma_20'] = result['Volume'].rolling(20).mean()
        result['volume_ratio'] = result['Volume'] / result['volume_ma_20']

        # OBV (On-Balance Volume)
        result['obv'] = self._obv(result['Close'], result['Volume'])
        result['obv_ma_20'] = result['obv'].rolling(20).mean()

        # Dollar volume
        result['dollar_volume'] = result['Close'] * result['Volume']
        result['dollar_volume_ma_20'] = result['dollar_volume'].rolling(20).mean()

        # ── Amihud Illiquidity ──
        result['amihud'] = (
            result['Close'].pct_change().abs() /
            result['dollar_volume']
        ).rolling(21).mean()

        # ── Rate of Change ──
        result['roc_10'] = result['Close'].pct_change(10) * 100
        result['roc_21'] = result['Close'].pct_change(21) * 100

        # ── Drawdown ──
        rolling_max = result['Close'].expanding().max()
        result['drawdown'] = (result['Close'] - rolling_max) / rolling_max

        # ── Stochastic Oscillator ──
        low_14 = result['Low'].rolling(14).min()
        high_14 = result['High'].rolling(14).max()
        result['stoch_k'] = (
            (result['Close'] - low_14) / (high_14 - low_14) * 100
        )
        result['stoch_d'] = result['stoch_k'].rolling(3).mean()

        return result

    @staticmethod
    def _rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """Compute Relative Strength Index."""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
        avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    @staticmethod
    def _atr(high: pd.Series, low: pd.Series,
             close: pd.Series, period: int = 14) -> pd.Series:
        """Compute Average True Range."""
        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs()
        ], axis=1).max(axis=1)
        return tr.rolling(period).mean()

    @staticmethod
    def _obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        """Compute On-Balance Volume."""
        sign = np.sign(close.diff())
        return (sign * volume).cumsum()
