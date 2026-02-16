"""
Fundamental Data Loader
=======================
Loads fundamental data with point-in-time discipline.
Uses yfinance quarterly financials with proper reporting lag.
"""

import logging
import pandas as pd
import numpy as np
import yfinance as yf

logger = logging.getLogger(__name__)

# Minimum reporting lag in calendar days (conservative)
REPORTING_LAG_DAYS = 90


class FundamentalLoader:
    """
    Load fundamental data with point-in-time discipline.

    All fundamental data is lagged by REPORTING_LAG_DAYS to prevent
    look-ahead bias. In a production system, this would use the actual
    SEC filing date from EDGAR.
    """

    METRICS = [
        'TrailingPE', 'ForwardPE', 'ReturnOnEquity', 'DividendYield',
        'ProfitMargins', 'OperatingMargins', 'DebtToEquity',
        'FreeCashflow', 'EnterpriseValue', 'MarketCap',
        'RevenueGrowth', 'EarningsGrowth', 'BookValue',
        'TotalRevenue', 'EBITDA', 'TotalDebt', 'TotalCash',
        'ReturnOnAssets',
    ]

    def __init__(self, tickers: list, start: str, end: str):
        self.tickers = tickers
        self.start = start
        self.end = end

    def load_quarterly_fundamentals(self) -> pd.DataFrame:
        """
        Load quarterly fundamental data for all tickers.

        Returns a DataFrame with columns:
            ticker, date, knowledge_date, and all fundamental metrics.

        The 'date' is the fiscal quarter end.
        The 'knowledge_date' is date + REPORTING_LAG_DAYS (when the
        data would have been available to an investor).
        """
        all_records = []

        for ticker in self.tickers:
            try:
                records = self._load_single_ticker(ticker)
                all_records.extend(records)
            except Exception as e:
                logger.warning(f"Failed to load fundamentals for {ticker}: {e}")

        if not all_records:
            logger.error("No fundamental data loaded")
            return pd.DataFrame()

        df = pd.DataFrame(all_records)
        logger.info(f"Loaded {len(df)} quarterly fundamental records "
                     f"for {df['ticker'].nunique()} tickers")
        return df

    def _load_single_ticker(self, ticker: str) -> list:
        """Load quarterly financials for a single ticker."""
        stock = yf.Ticker(ticker)
        records = []

        # Get quarterly financials
        try:
            quarterly_income = stock.quarterly_income_stmt
            quarterly_balance = stock.quarterly_balance_sheet
            quarterly_cashflow = stock.quarterly_cashflow
        except Exception:
            # Fallback: use current snapshot with reporting lag
            return self._load_snapshot_fallback(ticker)

        if quarterly_income is None or quarterly_income.empty:
            return self._load_snapshot_fallback(ticker)

        # Process each quarter
        for col_date in quarterly_income.columns:
            record = {'ticker': ticker}
            record['date'] = pd.Timestamp(col_date)
            record['knowledge_date'] = (
                record['date'] + pd.Timedelta(days=REPORTING_LAG_DAYS)
            )

            # Income statement fields
            record['total_revenue'] = self._safe_get(
                quarterly_income, 'Total Revenue', col_date)
            record['ebitda'] = self._safe_get(
                quarterly_income, 'EBITDA', col_date)
            record['net_income'] = self._safe_get(
                quarterly_income, 'Net Income', col_date)
            record['operating_income'] = self._safe_get(
                quarterly_income, 'Operating Income', col_date)

            # Balance sheet fields
            if quarterly_balance is not None and col_date in quarterly_balance.columns:
                record['total_assets'] = self._safe_get(
                    quarterly_balance, 'Total Assets', col_date)
                record['total_debt'] = self._safe_get(
                    quarterly_balance, 'Total Debt', col_date)
                record['total_equity'] = self._safe_get(
                    quarterly_balance, 'Stockholders Equity', col_date)
                record['total_cash'] = self._safe_get(
                    quarterly_balance, 'Cash And Cash Equivalents', col_date)
                record['current_assets'] = self._safe_get(
                    quarterly_balance, 'Current Assets', col_date)
                record['current_liabilities'] = self._safe_get(
                    quarterly_balance, 'Current Liabilities', col_date)

            # Cash flow fields
            if quarterly_cashflow is not None and col_date in quarterly_cashflow.columns:
                record['free_cash_flow'] = self._safe_get(
                    quarterly_cashflow, 'Free Cash Flow', col_date)
                record['capex'] = self._safe_get(
                    quarterly_cashflow, 'Capital Expenditure', col_date)
                record['operating_cash_flow'] = self._safe_get(
                    quarterly_cashflow, 'Operating Cash Flow', col_date)

            records.append(record)

        return records

    def _load_snapshot_fallback(self, ticker: str) -> list:
        """
        Fallback: load current snapshot and backdate with reporting lag.
        Less ideal but ensures data availability.
        """
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            record = {
                'ticker': ticker,
                'date': pd.Timestamp.today() - pd.Timedelta(days=REPORTING_LAG_DAYS),
                'knowledge_date': pd.Timestamp.today(),
                'trailing_pe': info.get('trailingPE'),
                'forward_pe': info.get('forwardPE'),
                'roe': info.get('returnOnEquity'),
                'roa': info.get('returnOnAssets'),
                'dividend_yield': info.get('dividendYield', 0),
                'profit_margin': info.get('profitMargins'),
                'operating_margin': info.get('operatingMargins'),
                'debt_to_equity': info.get('debtToEquity'),
                'market_cap': info.get('marketCap'),
                'enterprise_value': info.get('enterpriseValue'),
                'revenue_growth': info.get('revenueGrowth'),
                'earnings_growth': info.get('earningsGrowth'),
                'free_cash_flow': info.get('freeCashflow'),
                'total_revenue': info.get('totalRevenue'),
                'ebitda': info.get('ebitda'),
                'total_debt': info.get('totalDebt'),
                'total_cash': info.get('totalCash'),
                'book_value': info.get('bookValue'),
            }
            return [record]
        except Exception as e:
            logger.warning(f"Snapshot fallback failed for {ticker}: {e}")
            return []

    @staticmethod
    def _safe_get(df: pd.DataFrame, field: str, col) -> float:
        """Safely extract a value from a financial statement."""
        try:
            val = df.loc[field, col]
            if pd.isna(val):
                return np.nan
            return float(val)
        except (KeyError, TypeError):
            return np.nan

    def load_current_snapshot(self) -> pd.DataFrame:
        """
        Load current fundamental snapshot for all tickers.
        Used for live (non-backtest) analysis.
        """
        records = []
        for ticker in self.tickers:
            try:
                stock = yf.Ticker(ticker)
                info = stock.info
                records.append({
                    'ticker': ticker,
                    'trailing_pe': info.get('trailingPE'),
                    'forward_pe': info.get('forwardPE'),
                    'roe': info.get('returnOnEquity'),
                    'roa': info.get('returnOnAssets'),
                    'dividend_yield': info.get('dividendYield', 0),
                    'profit_margin': info.get('profitMargins'),
                    'operating_margin': info.get('operatingMargins'),
                    'debt_to_equity': info.get('debtToEquity'),
                    'market_cap': info.get('marketCap'),
                    'enterprise_value': info.get('enterpriseValue'),
                    'revenue_growth': info.get('revenueGrowth'),
                    'earnings_growth': info.get('earningsGrowth'),
                    'free_cash_flow': info.get('freeCashflow'),
                    'total_revenue': info.get('totalRevenue'),
                    'ebitda': info.get('ebitda'),
                    'total_debt': info.get('totalDebt'),
                    'total_cash': info.get('totalCash'),
                    'book_value': info.get('bookValue'),
                    'sector': info.get('sector', 'Unknown'),
                    'industry': info.get('industry', 'Unknown'),
                })
            except Exception as e:
                logger.warning(f"Failed snapshot for {ticker}: {e}")
        return pd.DataFrame(records)
