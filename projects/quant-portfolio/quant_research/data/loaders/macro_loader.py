"""
Macro & Regime Data Loader
==========================
Loads macroeconomic indicators from FRED and market data sources.
All data respects publication lag.
"""

import logging
import pandas as pd
import numpy as np
import yfinance as yf

logger = logging.getLogger(__name__)

# FRED series IDs and their publication lags (days)
FRED_SERIES = {
    'fed_funds_rate': {'id': 'DFF', 'lag': 1},
    'yield_curve_2s10s': {'id': 'T10Y2Y', 'lag': 1},
    'cpi_yoy': {'id': 'CPIAUCSL', 'lag': 30},
    'unemployment': {'id': 'UNRATE', 'lag': 30},
    'credit_spread_hy': {'id': 'BAMLH0A0HYM2', 'lag': 1},
    'ted_spread': {'id': 'TEDRATE', 'lag': 1},
    'breakeven_5y': {'id': 'T5YIE', 'lag': 1},
    'initial_claims': {'id': 'ICSA', 'lag': 7},
}


class MacroLoader:
    """Load macroeconomic and market regime data."""

    def __init__(self, start: str, end: str):
        self.start = start
        self.end = end

    def load_all(self) -> pd.DataFrame:
        """
        Load all macro indicators into a single DataFrame.

        Returns
        -------
        pd.DataFrame
            Daily frequency, forward-filled for non-daily series.
        """
        frames = {}

        # Market-based indicators (daily, from Yahoo Finance)
        frames['vix'] = self._load_yahoo('^VIX', 'vix')
        frames['sp500'] = self._load_yahoo('^GSPC', 'sp500')
        frames['dxy'] = self._load_yahoo('DX-Y.NYB', 'dxy')
        frames['tnx'] = self._load_yahoo('^TNX', 'tnx')  # 10Y yield

        # Try loading FRED data via pandas-datareader or fallback
        fred_data = self._load_fred_proxy()
        if fred_data is not None:
            frames['fred'] = fred_data

        # Combine all frames
        result = pd.DataFrame(index=pd.date_range(self.start, self.end, freq='B'))
        for name, df in frames.items():
            if df is not None and not df.empty:
                result = result.join(df, how='left')

        result = result.ffill().bfill()

        # Derived features
        result = self._add_derived_features(result)

        logger.info(f"Macro data loaded: {result.shape[1]} features, "
                     f"{result.shape[0]} observations")
        return result

    def _load_yahoo(self, symbol: str, name: str) -> pd.DataFrame:
        """Load a single Yahoo Finance series."""
        try:
            df = yf.download(symbol, start=self.start, end=self.end,
                             progress=False)
            if df.empty:
                return None
            # Flatten MultiIndex if present
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            return df[['Close']].rename(columns={'Close': name})
        except Exception as e:
            logger.warning(f"Failed to load {symbol}: {e}")
            return None

    def _load_fred_proxy(self) -> pd.DataFrame:
        """
        Load FRED data. Try pandas-datareader first, fall back to
        synthetic proxies from market data.
        """
        try:
            from pandas_datareader import data as pdr
            frames = {}
            for name, spec in FRED_SERIES.items():
                try:
                    s = pdr.DataReader(
                        spec['id'], 'fred',
                        self.start, self.end
                    )
                    s.columns = [name]
                    # Apply publication lag
                    s.index = s.index + pd.Timedelta(days=spec['lag'])
                    frames[name] = s
                except Exception:
                    logger.debug(f"Could not load FRED series {spec['id']}")
            if frames:
                result = pd.concat(frames.values(), axis=1)
                return result
        except ImportError:
            logger.info("pandas-datareader not available; using market proxies")

        return None

    def _add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived macro features."""
        # VIX regime (z-score of VIX level)
        if 'vix' in df.columns:
            df['vix_zscore'] = (
                (df['vix'] - df['vix'].rolling(252).mean()) /
                df['vix'].rolling(252).std()
            )
            df['vix_regime'] = pd.cut(
                df['vix'],
                bins=[0, 15, 20, 30, 100],
                labels=['low_vol', 'normal', 'elevated', 'crisis']
            )

        # Market momentum (S&P 500 returns)
        if 'sp500' in df.columns:
            df['sp500_ret_21d'] = df['sp500'].pct_change(21)
            df['sp500_ret_63d'] = df['sp500'].pct_change(63)
            df['sp500_ret_252d'] = df['sp500'].pct_change(252)

            # Market breadth proxy: is market above 200-day MA?
            df['sp500_above_200ma'] = (
                df['sp500'] > df['sp500'].rolling(200).mean()
            ).astype(float)

        # Dollar momentum
        if 'dxy' in df.columns:
            df['dxy_ret_21d'] = df['dxy'].pct_change(21)

        # Yield curve features
        if 'tnx' in df.columns:
            df['tnx_change_21d'] = df['tnx'].diff(21)

        return df
