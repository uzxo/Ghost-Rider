"""
Universe Construction
=====================
Point-in-time universe with survivorship bias controls.
"""

import logging
import pandas as pd
import numpy as np
import yfinance as yf
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class UniverseConstructor:
    """
    Construct a point-in-time investable universe.

    In a production system, this would use CRSP or Compustat
    historical constituent lists. Here we implement a practical
    approximation using available data.
    """

    def __init__(self, config: dict):
        self.config = config
        self._sector_cache: Dict[str, str] = {}
        self._ipo_dates: Dict[str, pd.Timestamp] = {}

    def get_universe(self, as_of_date: pd.Timestamp,
                     all_tickers: list) -> list:
        """
        Get the eligible universe as of a specific date.

        Parameters
        ----------
        as_of_date : pd.Timestamp
            The date for which to construct the universe.
        all_tickers : list
            Full candidate list.

        Returns
        -------
        list
            Tickers eligible for investment on this date.
        """
        eligible = []
        buffer_days = self.config.get('listing_buffer_days', 22)
        exclude = set(self.config.get('exclude_types', []))

        for ticker in all_tickers:
            # Check if ticker was listed by as_of_date
            ipo_date = self._get_ipo_date(ticker)
            if ipo_date is not None:
                if as_of_date < ipo_date + pd.Timedelta(days=buffer_days):
                    logger.debug(f"{ticker}: not yet listed on {as_of_date}")
                    continue

            # Exclude ETFs and other non-equity types
            if self._is_excluded_type(ticker, exclude):
                continue

            eligible.append(ticker)

        logger.info(f"Universe on {as_of_date.date()}: "
                     f"{len(eligible)}/{len(all_tickers)} eligible")
        return eligible

    def _get_ipo_date(self, ticker: str) -> Optional[pd.Timestamp]:
        """
        Estimate IPO date from first available price data.
        """
        if ticker in self._ipo_dates:
            return self._ipo_dates[ticker]

        try:
            df = yf.download(ticker, period='max', progress=False)
            if not df.empty:
                first_date = df.index[0]
                self._ipo_dates[ticker] = pd.Timestamp(first_date)
                return self._ipo_dates[ticker]
        except Exception:
            pass
        return None

    def _is_excluded_type(self, ticker: str, exclude_types: set) -> bool:
        """Check if ticker is an excluded type (ETF, REIT, etc.)."""
        etfs = {'SPY', 'QQQ', 'IWM', 'DIA', 'VTI', 'VOO', 'EFA',
                'EEM', 'TLT', 'GLD', 'SLV', 'XLF', 'XLK', 'XLE',
                'XLV', 'XLI', 'XLP', 'XLY', 'XLB', 'XLU', 'XLRE'}
        if 'ETF' in exclude_types and ticker in etfs:
            return True
        return False

    def get_sector(self, ticker: str) -> str:
        """Get sector classification for a ticker."""
        if ticker in self._sector_cache:
            return self._sector_cache[ticker]
        try:
            info = yf.Ticker(ticker).info
            sector = info.get('sector', 'Unknown')
            self._sector_cache[ticker] = sector
            return sector
        except Exception:
            self._sector_cache[ticker] = 'Unknown'
            return 'Unknown'

    def get_sector_map(self, tickers: list) -> Dict[str, str]:
        """Get sector mapping for all tickers."""
        return {t: self.get_sector(t) for t in tickers}

    def get_benchmark_prices(self, benchmark: str = '^GSPC',
                             start: str = None,
                             end: str = None) -> pd.Series:
        """Load benchmark prices."""
        df = yf.download(benchmark, start=start, end=end, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df['Close']
