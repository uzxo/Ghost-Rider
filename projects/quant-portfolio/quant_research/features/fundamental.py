"""
Fundamental Feature Engineering
================================
Computes derived fundamental features from raw financial data.
Includes Piotroski F-Score, quality metrics, and valuation ratios.
"""

import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class FundamentalFeatures:
    """Compute derived fundamental features."""

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all fundamental features from raw financial data.

        Parameters
        ----------
        df : pd.DataFrame
            Raw fundamental data with columns from FundamentalLoader.

        Returns
        -------
        pd.DataFrame
            Enhanced with derived features.
        """
        result = df.copy()

        # ── Valuation Ratios ──
        result['fcf_yield'] = self._safe_div(
            result.get('free_cash_flow'),
            result.get('enterprise_value')
        )
        result['ebitda_margin'] = self._safe_div(
            result.get('ebitda'),
            result.get('total_revenue')
        )
        result['earnings_yield'] = self._safe_div(
            1.0, result.get('trailing_pe')
        )

        # ── Profitability ──
        result['roic'] = self._compute_roic(result)
        result['gross_margin'] = self._safe_div(
            result.get('operating_income'),
            result.get('total_revenue')
        )

        # ── Leverage ──
        result['net_debt_ebitda'] = self._safe_div(
            (result.get('total_debt', 0) - result.get('total_cash', 0)),
            result.get('ebitda')
        )
        result['debt_equity'] = result.get('debt_to_equity', np.nan)

        # ── Quality ──
        result['accruals_ratio'] = self._compute_accruals(result)
        result['piotroski_f'] = self._piotroski_score(result)

        # ── Growth ──
        result['revenue_growth'] = result.get('revenue_growth', np.nan)
        result['earnings_growth'] = result.get('earnings_growth', np.nan)

        return result

    def compute_cross_sectional(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute cross-sectional features (ranks, z-scores).

        Parameters
        ----------
        df : pd.DataFrame
            Fundamental data with tickers as rows.

        Returns
        -------
        pd.DataFrame
            With additional cross-sectional features.
        """
        result = df.copy()
        numeric_cols = result.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            if col in result.columns and result[col].notna().sum() > 5:
                # Winsorize at 1st/99th percentile
                lo = result[col].quantile(0.01)
                hi = result[col].quantile(0.99)
                clipped = result[col].clip(lo, hi)

                # Z-score
                result[f'{col}_zscore'] = (
                    (clipped - clipped.mean()) / clipped.std()
                )
                # Rank (percentile)
                result[f'{col}_rank'] = clipped.rank(pct=True)

        return result

    def _compute_roic(self, df: pd.DataFrame) -> pd.Series:
        """
        Compute Return on Invested Capital.
        ROIC = Operating Income / (Total Equity + Total Debt - Cash)
        """
        operating = df.get('operating_income', pd.Series(dtype=float))
        equity = df.get('total_equity', pd.Series(dtype=float))
        debt = df.get('total_debt', pd.Series(dtype=float))
        cash = df.get('total_cash', pd.Series(dtype=float))

        invested_capital = equity.fillna(0) + debt.fillna(0) - cash.fillna(0)
        return self._safe_div(operating, invested_capital)

    def _compute_accruals(self, df: pd.DataFrame) -> pd.Series:
        """
        Compute accruals ratio.
        Accruals = (Net Income - Operating Cash Flow) / Total Assets
        High accruals = low earnings quality.
        """
        net_income = df.get('net_income', pd.Series(dtype=float))
        ocf = df.get('operating_cash_flow', pd.Series(dtype=float))
        total_assets = df.get('total_assets', pd.Series(dtype=float))

        return self._safe_div(
            net_income.fillna(0) - ocf.fillna(0),
            total_assets
        )

    def _piotroski_score(self, df: pd.DataFrame) -> pd.Series:
        """
        Compute Piotroski F-Score (0-9).

        Simplified version using available fields:
        1. ROA > 0
        2. Operating Cash Flow > 0
        3. ROA increasing (proxy: ROA > median)
        4. Accruals < 0 (Cash flow > Net income)
        5. Leverage decreasing (proxy: debt/equity < median)
        6. Current ratio improving (proxy: current ratio > 1)
        7. No dilution (proxy: always 1 for simplicity)
        8. Gross margin improving (proxy: > median)
        9. Asset turnover improving (proxy: > median)
        """
        score = pd.Series(0, index=df.index, dtype=float)

        # 1. Positive ROA
        roa = df.get('roa', pd.Series(dtype=float))
        if roa is not None:
            score += (roa > 0).astype(float).fillna(0)

        # 2. Positive OCF
        ocf = df.get('operating_cash_flow', pd.Series(dtype=float))
        if ocf is not None:
            score += (ocf > 0).astype(float).fillna(0)

        # 3. Positive ROE (proxy for ROA improvement)
        roe = df.get('roe', pd.Series(dtype=float))
        if roe is not None:
            score += (roe > 0).astype(float).fillna(0)

        # 4. Cash flow quality (OCF > Net Income)
        ni = df.get('net_income', pd.Series(dtype=float))
        if ocf is not None and ni is not None:
            score += (ocf > ni).astype(float).fillna(0)

        # 5. Lower leverage
        de = df.get('debt_to_equity', pd.Series(dtype=float))
        if de is not None and de.notna().sum() > 0:
            score += (de < de.median()).astype(float).fillna(0)

        # 6. Positive current ratio
        ca = df.get('current_assets', pd.Series(dtype=float))
        cl = df.get('current_liabilities', pd.Series(dtype=float))
        if ca is not None and cl is not None:
            cr = self._safe_div(ca, cl)
            score += (cr > 1).astype(float).fillna(0)

        # 7-9: Simplified (give 1 point each for positive margins)
        pm = df.get('profit_margin', pd.Series(dtype=float))
        if pm is not None:
            score += (pm > 0).astype(float).fillna(0)

        om = df.get('operating_margin', pd.Series(dtype=float))
        if om is not None:
            score += (om > 0).astype(float).fillna(0)

        rg = df.get('revenue_growth', pd.Series(dtype=float))
        if rg is not None:
            score += (rg > 0).astype(float).fillna(0)

        return score.clip(0, 9)

    @staticmethod
    def _safe_div(numerator, denominator) -> pd.Series:
        """Safe division handling zeros and NaN."""
        if numerator is None or denominator is None:
            return pd.Series(dtype=float)
        num = pd.Series(numerator) if not isinstance(numerator, pd.Series) else numerator
        den = pd.Series(denominator) if not isinstance(denominator, pd.Series) else denominator
        den = den.replace(0, np.nan)
        return num / den
