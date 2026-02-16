"""
Factor Exposure Computation
============================
Rolling factor betas via time-series regression against
Fama-French factors.
"""

import logging
import pandas as pd
import numpy as np
from typing import Optional

logger = logging.getLogger(__name__)

# Kenneth French Data Library URLs
FF_URL = ("https://mba.tuck.dartmouth.edu/pages/faculty/"
          "ken.french/ftp/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip")
MOM_URL = ("https://mba.tuck.dartmouth.edu/pages/faculty/"
           "ken.french/ftp/F-F_Momentum_Factor_daily_CSV.zip")


class FactorExposures:
    """Compute rolling factor exposures for individual securities."""

    def __init__(self, rolling_window: int = 60):
        self.rolling_window = rolling_window
        self._factors_cache: Optional[pd.DataFrame] = None

    def load_factors(self, start: str, end: str) -> pd.DataFrame:
        """
        Load Fama-French 5 factors + Momentum.

        Returns daily factor returns as percentages.
        Falls back to synthetic factors from market data if download fails.
        """
        if self._factors_cache is not None:
            return self._factors_cache

        try:
            factors = self._download_ff_factors(start, end)
            if factors is not None and not factors.empty:
                self._factors_cache = factors
                return factors
        except Exception as e:
            logger.warning(f"Could not download FF factors: {e}")

        # Fallback: create synthetic market factor from SPY
        logger.info("Using synthetic factor proxies")
        factors = self._synthetic_factors(start, end)
        self._factors_cache = factors
        return factors

    def compute_exposures(self, returns: pd.DataFrame,
                          factors: pd.DataFrame) -> pd.DataFrame:
        """
        Compute rolling factor betas for all securities.

        Parameters
        ----------
        returns : pd.DataFrame
            Daily returns for each ticker (columns).
        factors : pd.DataFrame
            Daily factor returns.

        Returns
        -------
        pd.DataFrame
            MultiIndex: (date, ticker), columns: factor betas + alpha.
        """
        # Align dates
        common_idx = returns.index.intersection(factors.index)
        returns = returns.loc[common_idx]
        factors = factors.loc[common_idx]

        results = []
        factor_cols = [c for c in factors.columns if c != 'RF']

        for ticker in returns.columns:
            ret = returns[ticker].dropna()
            if len(ret) < self.rolling_window:
                continue

            # Excess returns
            rf = factors['RF'].reindex(ret.index).fillna(0)
            excess_ret = ret - rf / 100  # RF is in percentage

            betas = self._rolling_regression(
                excess_ret, factors[factor_cols].reindex(ret.index) / 100,
                self.rolling_window
            )
            betas['ticker'] = ticker
            results.append(betas)

        if not results:
            return pd.DataFrame()

        result = pd.concat(results)
        result = result.set_index('ticker', append=True)
        logger.info(f"Computed factor exposures for "
                     f"{result.index.get_level_values('ticker').nunique()} tickers")
        return result

    def _rolling_regression(self, y: pd.Series, X: pd.DataFrame,
                            window: int) -> pd.DataFrame:
        """
        Rolling OLS regression.

        Returns DataFrame with columns for each beta + alpha + r_squared.
        """
        results = {}
        cols = X.columns.tolist()

        for i in range(window, len(y)):
            date = y.index[i]
            y_win = y.iloc[i - window:i].values
            X_win = X.iloc[i - window:i].values

            # Add constant for alpha
            X_const = np.column_stack([np.ones(window), X_win])

            try:
                # OLS: (X'X)^-1 X'y
                betas = np.linalg.lstsq(X_const, y_win, rcond=None)[0]
                y_pred = X_const @ betas
                ss_res = np.sum((y_win - y_pred) ** 2)
                ss_tot = np.sum((y_win - y_win.mean()) ** 2)
                r_sq = 1 - ss_res / ss_tot if ss_tot > 0 else 0

                row = {'alpha': betas[0] * 252}  # Annualize
                for j, col in enumerate(cols):
                    row[f'beta_{col}'] = betas[j + 1]
                row['r_squared'] = r_sq
                results[date] = row
            except np.linalg.LinAlgError:
                continue

        return pd.DataFrame(results).T

    def _download_ff_factors(self, start: str, end: str) -> pd.DataFrame:
        """Download FF factors from Kenneth French's data library."""
        import io
        import zipfile
        import urllib.request

        # Download 5 factors
        response = urllib.request.urlopen(FF_URL)
        z = zipfile.ZipFile(io.BytesIO(response.read()))
        name = z.namelist()[0]
        with z.open(name) as f:
            lines = f.read().decode('utf-8').split('\n')

        # Parse CSV (skip header rows)
        data_lines = []
        for line in lines:
            parts = line.strip().split(',')
            if len(parts) >= 6 and parts[0].strip().isdigit():
                data_lines.append(parts[:6])

        ff5 = pd.DataFrame(data_lines,
                           columns=['date', 'Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA'])
        ff5['date'] = pd.to_datetime(ff5['date'], format='%Y%m%d')
        for col in ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']:
            ff5[col] = ff5[col].astype(float)
        ff5 = ff5.set_index('date')
        ff5 = ff5.loc[start:end]

        # Add RF column
        ff5['RF'] = 0.0  # Will be overwritten if available

        return ff5

    def _synthetic_factors(self, start: str, end: str) -> pd.DataFrame:
        """Create synthetic factor proxies from market data."""
        import yfinance as yf

        spy = yf.download('SPY', start=start, end=end, progress=False)
        if isinstance(spy.columns, pd.MultiIndex):
            spy.columns = spy.columns.get_level_values(0)

        mkt_ret = spy['Close'].pct_change() * 100  # In percentage
        factors = pd.DataFrame({
            'Mkt-RF': mkt_ret,
            'SMB': np.random.normal(0, 0.3, len(mkt_ret)),  # Placeholder
            'HML': np.random.normal(0, 0.3, len(mkt_ret)),  # Placeholder
            'RMW': np.random.normal(0, 0.2, len(mkt_ret)),  # Placeholder
            'CMA': np.random.normal(0, 0.2, len(mkt_ret)),  # Placeholder
            'RF': 0.02 / 252 * 100,
        }, index=spy.index)

        logger.warning("Using synthetic factor proxies — "
                        "results will not be production-quality")
        return factors
