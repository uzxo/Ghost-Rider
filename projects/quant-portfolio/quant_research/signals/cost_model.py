"""
Transaction Cost Model
======================
Estimates realistic execution costs including commissions,
spread, and market impact.
"""

import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class TransactionCostModel:
    """
    Multi-component transaction cost estimator.

    Components:
    1. Commission (fixed bps)
    2. Spread cost (estimated from price/volume)
    3. Market impact (square-root model)
    """

    def __init__(self, config: dict):
        costs = config.get('costs', {})
        self.commission_bps = costs.get('commission_bps', 1.0)
        self.spread_bps = costs.get('spread_bps', 2.0)
        self.impact_eta = costs.get('market_impact_eta', 0.2)

    def estimate_cost(self, trade_value: float,
                      daily_volume: float,
                      volatility: float,
                      price: float) -> dict:
        """
        Estimate total transaction cost for a single trade.

        Parameters
        ----------
        trade_value : float
            Dollar value of the trade.
        daily_volume : float
            Average daily dollar volume.
        volatility : float
            Annualized volatility of the security.
        price : float
            Current price.

        Returns
        -------
        dict
            Breakdown of cost components in basis points.
        """
        # Commission
        commission = self.commission_bps

        # Spread
        spread = self.spread_bps

        # Market impact (square-root model)
        # Impact = sigma_daily * sqrt(Q / ADV) * eta
        daily_vol = volatility / np.sqrt(252)
        participation = trade_value / max(daily_volume, 1)
        impact_bps = (
            daily_vol * np.sqrt(participation) * self.impact_eta * 10000
        )

        total = commission + spread + impact_bps

        return {
            'commission_bps': commission,
            'spread_bps': spread,
            'impact_bps': impact_bps,
            'total_bps': total,
            'total_pct': total / 10000,
        }

    def estimate_portfolio_cost(self, old_weights: pd.Series,
                                 new_weights: pd.Series,
                                 portfolio_value: float,
                                 daily_volumes: pd.Series,
                                 volatilities: pd.Series,
                                 prices: pd.Series) -> dict:
        """
        Estimate total rebalancing cost for a portfolio.

        Parameters
        ----------
        old_weights, new_weights : pd.Series
            Portfolio weights before and after rebalancing.
        portfolio_value : float
            Total portfolio value.
        daily_volumes, volatilities, prices : pd.Series
            Security-level data.

        Returns
        -------
        dict
            Total cost and per-security breakdown.
        """
        all_tickers = (
            set(old_weights.index) | set(new_weights.index)
        )

        total_cost = 0
        details = {}

        for ticker in all_tickers:
            old_w = old_weights.get(ticker, 0)
            new_w = new_weights.get(ticker, 0)
            trade_w = abs(new_w - old_w)

            if trade_w < 1e-6:
                continue

            trade_value = trade_w * portfolio_value
            adv = daily_volumes.get(ticker, portfolio_value * 0.01)
            vol = volatilities.get(ticker, 0.3)
            px = prices.get(ticker, 100)

            cost = self.estimate_cost(trade_value, adv, vol, px)
            cost_dollars = cost['total_pct'] * trade_value
            total_cost += cost_dollars

            details[ticker] = {
                'trade_weight': trade_w,
                'trade_value': trade_value,
                **cost,
                'cost_dollars': cost_dollars,
            }

        turnover = sum(
            abs(new_weights.get(t, 0) - old_weights.get(t, 0))
            for t in all_tickers
        ) / 2  # One-way turnover

        return {
            'total_cost_dollars': total_cost,
            'total_cost_bps': (
                total_cost / portfolio_value * 10000
                if portfolio_value > 0 else 0
            ),
            'one_way_turnover': turnover,
            'details': details,
        }
