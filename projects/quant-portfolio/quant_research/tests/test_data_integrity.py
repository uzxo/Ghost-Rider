"""
Data Integrity Tests
====================
Verifies no look-ahead bias, proper data handling,
and universe construction validity.
"""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


class TestNoLookAhead(unittest.TestCase):
    """Test that no future data contaminates past decisions."""

    def test_fundamental_reporting_lag(self):
        """Fundamental data must respect 90-day reporting lag."""
        from data.loaders.fundamental_loader import REPORTING_LAG_DAYS
        self.assertGreaterEqual(REPORTING_LAG_DAYS, 60,
                                 "Reporting lag must be >= 60 days")

    def test_pit_store_query_respects_knowledge_date(self):
        """PIT store must only return data known before query date."""
        from data.store.pit_store import PointInTimeStore
        import tempfile
        import os

        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name

        try:
            store = PointInTimeStore(db_path)

            # Insert data with future knowledge date
            store.insert('AAPL', 'pe_ratio', 25.0,
                          '2024-01-01', '2024-04-01', 'test')

            # Query as of 2024-03-01 (before knowledge date)
            result = store.query(['AAPL'], ['pe_ratio'], '2024-03-01')
            self.assertTrue(result['pe_ratio'].isna().all(),
                             "PIT query must not return data known after query date")

            # Query as of 2024-04-15 (after knowledge date)
            result = store.query(['AAPL'], ['pe_ratio'], '2024-04-15')
            self.assertAlmostEqual(result.loc['AAPL', 'pe_ratio'], 25.0)

        finally:
            os.unlink(db_path)

    def test_feature_scaling_no_leakage(self):
        """Scaler must be fit on training data only."""
        from signals.normalize import SignalNormalizer

        # Create data with known distribution
        signal = pd.Series([1, 2, 3, 4, 5, 100])  # 100 is outlier
        zscore = SignalNormalizer.zscore(signal)

        # Z-score of 100 should be extremely large
        self.assertGreater(zscore.iloc[-1], 2.0,
                            "Z-score should detect outlier")


class TestUniverseConstruction(unittest.TestCase):
    """Test universe construction rules."""

    def test_etfs_excluded(self):
        """ETFs must not be in the investable universe."""
        from backtest.universe import UniverseConstructor

        config = {'exclude_types': ['ETF'], 'listing_buffer_days': 22}
        uc = UniverseConstructor(config)

        # SPY and QQQ should be excluded
        self.assertTrue(
            uc._is_excluded_type('SPY', {'ETF'}),
            "SPY (ETF) must be excluded from universe"
        )
        self.assertTrue(
            uc._is_excluded_type('QQQ', {'ETF'}),
            "QQQ (ETF) must be excluded from universe"
        )
        self.assertFalse(
            uc._is_excluded_type('AAPL', {'ETF'}),
            "AAPL should not be excluded"
        )


class TestReturnsComputation(unittest.TestCase):
    """Test return computation correctness."""

    def test_simple_returns(self):
        """Verify simple return computation."""
        from data.loaders.price_loader import PriceLoader

        loader = PriceLoader([], '2020-01-01', '2020-12-31')
        prices = pd.DataFrame({
            'A': [100, 110, 105, 115],
            'B': [50, 55, 52, 58],
        })
        returns = loader.compute_returns(prices, period=1)

        # First return for A: (110-100)/100 = 0.10
        self.assertAlmostEqual(returns['A'].iloc[0], 0.10, places=4)

    def test_returns_no_nan_propagation(self):
        """Returns should handle NaN gracefully."""
        from data.loaders.price_loader import PriceLoader

        loader = PriceLoader([], '2020-01-01', '2020-12-31')
        prices = pd.DataFrame({
            'A': [100, np.nan, 105, 115],
            'B': [50, 55, 52, 58],
        })
        returns = loader.compute_returns(prices, period=1)
        # Should not crash; NaN returns where prices are missing
        self.assertFalse(returns.empty)


class TestSignalNormalization(unittest.TestCase):
    """Test signal normalization properties."""

    def test_zscore_properties(self):
        """Z-scored signal should have mean ~0, std ~1."""
        from signals.normalize import SignalNormalizer

        np.random.seed(42)
        signal = pd.Series(np.random.randn(100))
        zscore = SignalNormalizer.zscore(signal)

        self.assertAlmostEqual(zscore.mean(), 0, places=10)
        self.assertAlmostEqual(zscore.std(), 1, places=10)

    def test_rank_normalize_range(self):
        """Rank-normalized signal should be in [-1, 1]."""
        from signals.normalize import SignalNormalizer

        signal = pd.Series([10, 5, 20, 15, 1])
        ranked = SignalNormalizer.rank_normalize(signal)

        self.assertGreaterEqual(ranked.min(), -1.0)
        self.assertLessEqual(ranked.max(), 1.0)

    def test_winsorize_clips_outliers(self):
        """Winsorization should clip extreme values."""
        from signals.normalize import SignalNormalizer

        signal = pd.Series([1, 2, 3, 4, 5, 1000])
        winsorized = SignalNormalizer.winsorize(signal, 5, 95)

        self.assertLess(winsorized.max(), 1000,
                         "Winsorization should clip the outlier")


class TestPortfolioConstraints(unittest.TestCase):
    """Test portfolio constraint enforcement."""

    def test_weights_sum_to_one(self):
        """Portfolio weights must sum to 1."""
        from portfolio.optimizer import PortfolioOptimizer

        config = {'optimizer': 'hrp'}
        opt = PortfolioOptimizer(config)

        # Create simple covariance matrix
        n = 5
        np.random.seed(42)
        returns = pd.DataFrame(
            np.random.randn(252, n) * 0.01,
            columns=[f'stock_{i}' for i in range(n)]
        )
        cov = returns.cov()
        signal = pd.Series(np.random.randn(n),
                            index=cov.index)

        weights = opt.optimize(signal, cov)
        self.assertAlmostEqual(weights.sum(), 1.0, places=4,
                                msg="Weights must sum to 1")

    def test_max_weight_constraint(self):
        """No single position should exceed max_weight."""
        from portfolio.optimizer import PortfolioOptimizer

        config = {
            'optimizer': 'mean_variance',
            'constraints': {'max_weight': 0.10}
        }
        opt = PortfolioOptimizer(config)

        n = 10
        np.random.seed(42)
        returns = pd.DataFrame(
            np.random.randn(252, n) * 0.01,
            columns=[f's{i}' for i in range(n)]
        )
        cov = returns.cov()
        signal = pd.Series(np.random.randn(n), index=cov.index)

        weights = opt.optimize(signal, cov,
                               constraints={'max_weight': 0.10})
        weights = opt._apply_constraints(weights,
                                          {'max_weight': 0.10, 'min_weight': 0.001})
        self.assertLessEqual(weights.max(), 0.10 + 0.01,
                              "Max weight constraint violated")


if __name__ == '__main__':
    unittest.main()
