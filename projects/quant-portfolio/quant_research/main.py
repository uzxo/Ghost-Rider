#!/usr/bin/env python3
"""
Quant Research Framework — Main Entry Point
=============================================
Orchestrates the full research pipeline:
  Data -> Features -> Models -> Signals -> Portfolio -> Evaluation

Usage:
    python main.py                        # Full backtest
    python main.py --mode signal_only     # Signal generation only
    python main.py --mode evaluate        # Evaluate existing results
"""

import os
import sys
import time
import logging
import argparse
import random
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# ── Setup paths ──
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── Import framework modules ──
from data.loaders.price_loader import PriceLoader
from data.loaders.fundamental_loader import FundamentalLoader
from data.loaders.macro_loader import MacroLoader
from features.technical import TechnicalFeatures
from features.fundamental import FundamentalFeatures
from features.macro import MacroFeatures
from features.factor_exposures import FactorExposures
from models.xgboost_model import CrossSectionalGBM
from models.linear_model import LinearAlphaModel
from models.hmm_model import RegimeHMM
from models.ensemble import EnsembleModel
from signals.normalize import SignalNormalizer
from signals.combine import SignalCombiner
from signals.cost_model import TransactionCostModel
from portfolio.optimizer import PortfolioOptimizer
from portfolio.constraints import PortfolioConstraints
from backtest.engine import BacktestEngine
from backtest.universe import UniverseConstructor
from evaluation.metrics import PerformanceMetrics
from evaluation.attribution import FactorAttribution
from evaluation.robustness import RobustnessTests
from evaluation import visualization as viz


def setup_logging(level: str = "INFO"):
    """Configure structured logging."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s | %(name)-25s | %(levelname)-7s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(
                PROJECT_ROOT / 'research.log', mode='a'
            ),
        ]
    )


def set_seeds(seed: int = 42):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def load_config() -> dict:
    """Load all configuration files."""
    config = {}
    config_dir = PROJECT_ROOT / 'config'
    for f in config_dir.glob('*.yaml'):
        with open(f) as fh:
            config.update(yaml.safe_load(fh))
    return config


def run_pipeline(config: dict):
    """
    Execute the full research pipeline.

    Pipeline Steps:
    ===============
    1. Load data (prices, fundamentals, macro)
    2. Compute features (technical, fundamental, macro, factor)
    3. Train models (XGBoost, ElasticNet, HMM)
    4. Generate and combine signals
    5. Construct portfolio (HRP / Black-Litterman)
    6. Run walk-forward backtest
    7. Evaluate performance and robustness
    """
    logger = logging.getLogger('pipeline')
    logger.info("=" * 60)
    logger.info("QUANT RESEARCH PIPELINE — START")
    logger.info("=" * 60)
    start_time = time.time()

    # ── Configuration ──
    tickers = config['universe']['fallback_tickers']
    start_date = config['dates']['start']
    end_date = config['dates']['end']
    holdout_start = config['dates']['holdout_start']
    seed = config.get('seed', 42)
    set_seeds(seed)

    # ══════════════════════════════════════════════════════
    # STEP 1: LOAD DATA
    # ══════════════════════════════════════════════════════
    logger.info("STEP 1: Loading data...")

    # Price data
    price_loader = PriceLoader(tickers, start_date, end_date)
    close_prices = price_loader.load_close_prices()
    daily_returns = price_loader.compute_returns(close_prices)
    logger.info(f"  Prices: {close_prices.shape}")

    # Remove tickers with insufficient data
    min_obs = 252 * 2  # 2 years minimum
    valid_tickers = [
        t for t in close_prices.columns
        if close_prices[t].dropna().shape[0] >= min_obs
    ]
    close_prices = close_prices[valid_tickers]
    daily_returns = daily_returns[valid_tickers]
    tickers = valid_tickers
    logger.info(f"  Valid tickers: {len(tickers)}")

    # Fundamental data (current snapshot for signal generation)
    fund_loader = FundamentalLoader(tickers, start_date, end_date)
    fundamentals = fund_loader.load_current_snapshot()
    logger.info(f"  Fundamentals: {fundamentals.shape}")

    # Macro data
    macro_loader = MacroLoader(start_date, end_date)
    macro_data = macro_loader.load_all()
    logger.info(f"  Macro: {macro_data.shape}")

    # ══════════════════════════════════════════════════════
    # STEP 2: FEATURE ENGINEERING
    # ══════════════════════════════════════════════════════
    logger.info("STEP 2: Computing features...")

    # Technical features (per ticker)
    tech_engine = TechnicalFeatures()
    tech_features = {}
    for ticker in tickers:
        ohlcv = price_loader.load_ohlcv(ticker)
        if not ohlcv.empty:
            tech_features[ticker] = tech_engine.compute(ohlcv)

    # Fundamental features
    fund_engine = FundamentalFeatures()
    if not fundamentals.empty:
        fundamentals = fund_engine.compute(fundamentals)
        fundamentals = fund_engine.compute_cross_sectional(fundamentals)

    # Macro features
    macro_engine = MacroFeatures()
    macro_features = macro_engine.compute(macro_data)

    # Factor exposures
    factor_engine = FactorExposures(rolling_window=60)
    factors = factor_engine.load_factors(start_date, end_date)

    logger.info(f"  Technical features: {len(tech_features)} tickers")
    logger.info(f"  Macro features: {macro_features.shape[1]} columns")

    # ══════════════════════════════════════════════════════
    # STEP 3: BUILD CROSS-SECTIONAL FEATURE MATRIX
    # ══════════════════════════════════════════════════════
    logger.info("STEP 3: Building cross-sectional feature matrix...")

    # For the backtest, we need to construct features at each rebalance date
    # Here we build the latest cross-section for model training

    feature_matrix = _build_feature_matrix(
        tickers, close_prices, daily_returns,
        tech_features, fundamentals, macro_features
    )
    logger.info(f"  Feature matrix: {feature_matrix.shape}")

    # ══════════════════════════════════════════════════════
    # STEP 4: TRAIN MODELS
    # ══════════════════════════════════════════════════════
    logger.info("STEP 4: Training models...")

    # Compute forward return target
    fwd_returns = daily_returns.shift(-21).rolling(21).mean()  # 1-month fwd
    latest_fwd = fwd_returns.iloc[-22]  # Most recent complete target

    # Cross-sectional target: rank of forward returns
    target = latest_fwd.reindex(tickers).rank(pct=True)
    target = target.dropna()

    # Align features and target
    common_tickers = list(
        set(feature_matrix.index) & set(target.index)
    )
    if len(common_tickers) < 10:
        logger.warning("Insufficient tickers for model training; "
                        "using available data")
        common_tickers = list(feature_matrix.index)
        target = target.reindex(common_tickers).fillna(0.5)

    X = feature_matrix.loc[common_tickers]
    y = target.loc[common_tickers]

    # Model 1: XGBoost
    xgb_model = CrossSectionalGBM(config)
    xgb_model.build_model()
    xgb_model.train(X, y)
    xgb_preds = xgb_model.predict(X)
    logger.info(f"  XGBoost trained. Top features: "
                 f"{list(xgb_model.feature_importance.head(5).index)}")

    # Model 2: Elastic Net
    enet_model = LinearAlphaModel(config)
    enet_model.build_model()
    enet_model.train(X, y)
    enet_preds = enet_model.predict(X)
    logger.info(f"  ElasticNet trained. Non-zero coeffs: "
                 f"{(enet_model.get_coefficients().abs() > 1e-6).sum()}")

    # Model 3: HMM Regime Detection
    hmm_model = RegimeHMM(config)
    hmm_model.build_model()
    hmm_model.train(macro_features)
    current_regime = hmm_model.predict_regime(macro_features)
    logger.info(f"  HMM regime detection complete. "
                 f"Current regime: {current_regime.iloc[-1] if len(current_regime) > 0 else 'N/A'}")

    # ══════════════════════════════════════════════════════
    # STEP 5: COMBINE SIGNALS
    # ══════════════════════════════════════════════════════
    logger.info("STEP 5: Combining signals...")

    # Normalize predictions
    normalizer = SignalNormalizer()
    signals = pd.DataFrame({
        'xgboost': normalizer.rank_normalize(xgb_preds),
        'elastic_net': normalizer.rank_normalize(enet_preds),
    })

    # Add fundamental signal
    if 'fcf_yield_rank' in fundamentals.columns:
        fund_signal = fundamentals.set_index('ticker')['fcf_yield_rank']
        fund_signal = fund_signal.reindex(common_tickers)
        signals['fundamental'] = normalizer.rank_normalize(
            fund_signal.fillna(0.5)
        )

    # Ensemble
    ensemble = EnsembleModel(config)
    combiner = SignalCombiner(ic_lookback=12)

    # IC-weighted combination (use equal weights initially)
    composite_signal = signals.mean(axis=1)
    composite_signal.name = 'composite'
    logger.info(f"  Composite signal range: "
                 f"[{composite_signal.min():.3f}, {composite_signal.max():.3f}]")

    # ══════════════════════════════════════════════════════
    # STEP 6: PORTFOLIO CONSTRUCTION
    # ══════════════════════════════════════════════════════
    logger.info("STEP 6: Constructing portfolio...")

    # Covariance estimation
    port_optimizer = PortfolioOptimizer(config)
    returns_for_cov = daily_returns[common_tickers].dropna()
    cov_matrix = port_optimizer.estimate_covariance(
        returns_for_cov, method='ledoit_wolf'
    )

    # Optimize
    weights = port_optimizer.optimize(
        expected_returns=composite_signal,
        cov_matrix=cov_matrix,
        constraints=config.get('constraints', config.get('portfolio', {})),
    )

    # Apply constraints
    constraint_mgr = PortfolioConstraints(config)
    weights = constraint_mgr.apply_volatility_targeting(
        weights, cov_matrix,
        target_vol=config.get('portfolio', {}).get('target_volatility', 0.12)
    )

    # Normalize to long-only
    weights = weights.clip(lower=0)
    if weights.sum() > 0:
        weights = weights / weights.sum()

    # Select top N holdings
    num_holdings = config.get('portfolio', {}).get('num_holdings', 40)
    top_weights = weights.nlargest(num_holdings)
    top_weights = top_weights / top_weights.sum()  # Renormalize

    logger.info(f"  Portfolio: {len(top_weights)} holdings")
    logger.info(f"  Top 5: {top_weights.head(5).to_dict()}")

    # ══════════════════════════════════════════════════════
    # STEP 7: BACKTEST
    # ══════════════════════════════════════════════════════
    logger.info("STEP 7: Running backtest...")

    engine = BacktestEngine(config)

    # Define signal function for backtest
    def signal_func(date, universe, prices):
        """Generate signals at each rebalance date."""
        rets = prices[universe].pct_change().dropna()
        if rets.empty:
            return pd.Series(0, index=universe)

        # Simple momentum + mean-reversion blend
        mom_12_1 = prices[universe].iloc[-252:].iloc[-1] / prices[universe].iloc[-252:].iloc[0] - 1
        mom_1m = prices[universe].iloc[-21:].iloc[-1] / prices[universe].iloc[-21:].iloc[0] - 1
        momentum = mom_12_1 - mom_1m  # 12-1 momentum

        # Volatility (inverse)
        vol = rets[universe].tail(63).std() * np.sqrt(252)
        inv_vol = 1 / vol.replace(0, np.nan)

        # Combine
        signal = normalizer.rank_normalize(momentum.fillna(0)) * 0.6 + \
                 normalizer.rank_normalize(inv_vol.fillna(0)) * 0.4

        return signal

    def portfolio_func(signals, returns, date):
        """Construct portfolio at each rebalance."""
        avail = [t for t in signals.index if t in returns.columns]
        if len(avail) < 5:
            return pd.Series(dtype=float)

        sig = signals[avail]
        ret = returns[avail].tail(252)

        cov = port_optimizer.estimate_covariance(ret, method='ledoit_wolf')
        w = port_optimizer.optimize(
            expected_returns=sig,
            cov_matrix=cov,
            constraints=config.get('constraints', config.get('portfolio', {})),
        )

        # Select top N
        w = w.nlargest(num_holdings)
        w = w / w.sum() if w.sum() > 0 else w
        return w

    # Cost function
    cost_model = TransactionCostModel(config)
    def cost_func(old_w, new_w):
        all_t = set(old_w.index) | set(new_w.index)
        turnover = sum(
            abs(new_w.get(t, 0) - old_w.get(t, 0)) for t in all_t
        ) / 2
        return turnover * (config.get('costs', {}).get('commission_bps', 1) +
                           config.get('costs', {}).get('spread_bps', 2))

    # Run backtest
    bt_start = config['dates'].get('start', '2018-01-01')
    # Ensure at least 2 years of warmup data
    bt_start_date = max(
        pd.Timestamp(bt_start),
        close_prices.index[0] + pd.Timedelta(days=504)
    ).strftime('%Y-%m-%d')

    backtest_results = engine.run(
        close_prices=close_prices,
        signal_func=signal_func,
        portfolio_func=portfolio_func,
        cost_func=cost_func,
        start=bt_start_date,
        end=end_date,
    )

    portfolio_returns = backtest_results['returns']
    logger.info(f"  Backtest: {len(portfolio_returns)} days of returns")

    # ══════════════════════════════════════════════════════
    # STEP 8: EVALUATION
    # ══════════════════════════════════════════════════════
    logger.info("STEP 8: Evaluating performance...")

    # Benchmark
    universe_constructor = UniverseConstructor(config.get('universe', {}))
    benchmark = universe_constructor.get_benchmark_prices(
        '^GSPC', start=bt_start_date, end=end_date
    )
    benchmark_returns = benchmark.pct_change().dropna()

    # Performance metrics
    perf = PerformanceMetrics(
        risk_free_rate=config.get('risk_free_rate', 0.04)
    )
    metrics = perf.compute_all(
        portfolio_returns,
        benchmark_returns=benchmark_returns
    )
    print("\n" + perf.generate_report(metrics))

    # Rolling metrics
    rolling = perf.compute_rolling(portfolio_returns)

    # Factor attribution
    attr = FactorAttribution(
        risk_free_rate=config.get('risk_free_rate', 0.04)
    )
    factor_results = attr.attribute(portfolio_returns, factors)
    if factor_results:
        print("\n" + attr.generate_report(factor_results))

    # Robustness
    robust = RobustnessTests(
        risk_free_rate=config.get('risk_free_rate', 0.04)
    )

    # Bootstrap
    bootstrap = robust.monte_carlo_bootstrap(
        portfolio_returns,
        benchmark_returns=benchmark_returns,
        n_bootstrap=5000,
    )

    # Stress testing
    stress = robust.regime_stress_test(
        portfolio_returns,
        benchmark_returns=benchmark_returns,
    )

    # Holdout test
    holdout_ret = portfolio_returns.loc[holdout_start:]
    is_sharpe = metrics['sharpe_ratio']
    holdout = robust.holdout_test(is_sharpe, holdout_ret) if len(holdout_ret) > 20 else None

    print("\n" + robust.generate_report(bootstrap, stress, holdout))

    # ══════════════════════════════════════════════════════
    # STEP 9: VISUALIZATION
    # ══════════════════════════════════════════════════════
    logger.info("STEP 9: Generating visualizations...")

    output_dir = PROJECT_ROOT / 'output'
    output_dir.mkdir(exist_ok=True)

    viz.plot_equity_curve(
        portfolio_returns,
        benchmark_returns=benchmark_returns,
        title="Portfolio vs S&P 500",
        save_path=str(output_dir / 'equity_curve.png'),
    )

    viz.plot_rolling_metrics(
        rolling,
        title="Rolling Performance (252-day)",
        save_path=str(output_dir / 'rolling_metrics.png'),
    )

    if factor_results:
        viz.plot_factor_attribution(
            factor_results,
            save_path=str(output_dir / 'factor_attribution.png'),
        )

    viz.plot_weights(
        top_weights,
        title="Current Portfolio Weights",
        save_path=str(output_dir / 'portfolio_weights.png'),
    )

    viz.plot_bootstrap_distribution(
        bootstrap,
        save_path=str(output_dir / 'bootstrap_sharpe.png'),
    )

    # ══════════════════════════════════════════════════════
    # SUMMARY
    # ══════════════════════════════════════════════════════
    elapsed = time.time() - start_time
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info(f"  Elapsed: {elapsed:.1f} seconds")
    logger.info(f"  CAGR: {metrics.get('cagr', 0):.2%}")
    logger.info(f"  Sharpe: {metrics.get('sharpe_ratio', 0):.3f}")
    logger.info(f"  Max DD: {metrics.get('max_drawdown', 0):.2%}")
    logger.info(f"  Alpha (CAPM): {metrics.get('alpha_capm', 'N/A')}")
    logger.info("=" * 60)

    return {
        'metrics': metrics,
        'backtest': backtest_results,
        'weights': top_weights,
        'factor_attribution': factor_results,
        'bootstrap': bootstrap,
        'stress': stress,
        'holdout': holdout,
    }


def _build_feature_matrix(tickers, close_prices, daily_returns,
                           tech_features, fundamentals, macro_features):
    """Build a cross-sectional feature matrix for the latest date."""
    records = []

    for ticker in tickers:
        row = {'ticker': ticker}

        # Technical features (latest values)
        if ticker in tech_features:
            tf = tech_features[ticker]
            if not tf.empty:
                latest = tf.iloc[-1]
                for col in ['ret_5d', 'ret_21d', 'ret_63d', 'ret_252d',
                             'momentum_12_1', 'rsi_14', 'macd_hist',
                             'bb_position', 'vol_21d', 'vol_63d',
                             'vol_ratio', 'atr_pct', 'volume_ratio',
                             'amihud', 'drawdown', 'stoch_k',
                             'ma_5_dist', 'ma_20_dist', 'ma_50_dist',
                             'ma_200_dist', 'ma_cross_50_200']:
                    if col in latest.index:
                        row[col] = latest[col]

        # Fundamental features
        if not fundamentals.empty:
            fund_row = fundamentals[fundamentals['ticker'] == ticker]
            if not fund_row.empty:
                fr = fund_row.iloc[0]
                for col in ['trailing_pe', 'forward_pe', 'roe', 'roa',
                             'dividend_yield', 'profit_margin',
                             'operating_margin', 'debt_to_equity',
                             'revenue_growth', 'earnings_growth',
                             'fcf_yield', 'ebitda_margin', 'roic',
                             'net_debt_ebitda', 'accruals_ratio',
                             'piotroski_f']:
                    if col in fr.index:
                        row[col] = fr[col]

        records.append(row)

    df = pd.DataFrame(records).set_index('ticker')

    # Add macro features (same for all tickers — cross-sectional broadcast)
    if not macro_features.empty:
        latest_macro = macro_features.iloc[-1]
        for col in ['vix', 'vix_zscore_1y', 'sp500_ret_21d',
                     'sp500_ret_63d', 'sp500_above_200ma']:
            if col in latest_macro.index:
                df[col] = latest_macro[col]

    # Drop columns that are all NaN
    df = df.dropna(axis=1, how='all')

    # Fill remaining NaN with median (cross-sectional)
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].fillna(df[col].median())

    return df


# ══════════════════════════════════════════════════════
# CLI ENTRY POINT
# ══════════════════════════════════════════════════════
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Quant Research Framework'
    )
    parser.add_argument('--mode', type=str, default='full',
                        choices=['full', 'signal_only', 'evaluate'],
                        help='Pipeline mode')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to custom config file')
    args = parser.parse_args()

    # Load config
    config = load_config()

    # Setup
    setup_logging(config.get('log_level', 'INFO'))
    set_seeds(config.get('seed', 42))

    # Run
    results = run_pipeline(config)
