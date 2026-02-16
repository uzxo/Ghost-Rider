"""
Price Data Loader
=================
Fetches adjusted OHLCV data from Yahoo Finance with proper
corporate action handling and validation.
"""

import logging
import pandas as pd
import numpy as np
import yfinance as yf

logger = logging.getLogger(__name__)


class PriceLoader:
    """Load and validate adjusted price data."""

    def __init__(self, tickers: list, start: str, end: str):
        self.tickers = tickers
        self.start = start
        self.end = end
        self._cache = {}

    def load(self) -> pd.DataFrame:
        """
        Download adjusted close prices for all tickers.

        Returns
        -------
        pd.DataFrame
            MultiIndex columns (ticker, field) with OHLCV data.
        """
        logger.info(f"Downloading price data for {len(self.tickers)} tickers "
                     f"from {self.start} to {self.end}")

        raw = yf.download(
            self.tickers,
            start=self.start,
            end=self.end,
            auto_adjust=True,       # Use split/dividend-adjusted prices
            group_by='ticker',
            threads=True,
        )

        # Validate and clean
        prices = self._validate(raw)
        logger.info(f"Price data loaded: {prices.shape}")
        return prices

    def load_close_prices(self) -> pd.DataFrame:
        """
        Get a clean DataFrame of adjusted close prices (tickers as columns).

        Returns
        -------
        pd.DataFrame
            Index: DatetimeIndex, Columns: tickers
        """
        raw = self.load()
        close = pd.DataFrame()
        for ticker in self.tickers:
            try:
                if isinstance(raw.columns, pd.MultiIndex):
                    s = raw.xs(ticker, level=0, axis=1)['Close']
                else:
                    s = raw['Close']
                close[ticker] = s
            except (KeyError, Exception) as e:
                logger.warning(f"Could not extract close for {ticker}: {e}")
        close = close.dropna(how='all')
        close = close.ffill().bfill()
        return close

    def load_ohlcv(self, ticker: str) -> pd.DataFrame:
        """Load OHLCV for a single ticker."""
        if ticker in self._cache:
            return self._cache[ticker]

        df = yf.download(
            ticker,
            start=self.start,
            end=self.end,
            auto_adjust=True,
        )
        if df.empty:
            logger.warning(f"No data for {ticker}")
            return pd.DataFrame()

        # Flatten MultiIndex columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
        self._cache[ticker] = df
        return df

    def _validate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Validate price data for anomalies."""
        # Check for suspiciously large single-day moves (> 50%)
        # which may indicate bad split adjustment
        if isinstance(data.columns, pd.MultiIndex):
            for ticker in self.tickers:
                try:
                    close = data.xs(ticker, level=0, axis=1)['Close']
                    rets = close.pct_change().abs()
                    bad_days = rets[rets > 0.5]
                    if len(bad_days) > 0:
                        logger.warning(
                            f"{ticker}: {len(bad_days)} days with >50% move "
                            f"(possible bad adjustment)"
                        )
                except (KeyError, Exception):
                    pass
        return data

    def compute_returns(self, close: pd.DataFrame,
                        period: int = 1) -> pd.DataFrame:
        """
        Compute log returns or simple returns.

        Parameters
        ----------
        close : pd.DataFrame
            Close prices with tickers as columns.
        period : int
            Return period in trading days.

        Returns
        -------
        pd.DataFrame
            Simple returns.
        """
        return close.pct_change(period).dropna(how='all')
