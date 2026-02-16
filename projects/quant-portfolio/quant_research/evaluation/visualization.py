"""
Standardized Visualization
============================
Publication-quality charts for quant research.
"""

import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# Style configuration
COLORS = {
    'portfolio': '#1a5276',
    'benchmark': '#7f8c8d',
    'drawdown': '#c0392b',
    'positive': '#27ae60',
    'negative': '#e74c3c',
    'neutral': '#95a5a6',
    'accent': '#2980b9',
}

plt.rcParams.update({
    'figure.figsize': (14, 7),
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'figure.dpi': 100,
})


def plot_equity_curve(portfolio_returns: pd.Series,
                      benchmark_returns: pd.Series = None,
                      title: str = "Portfolio Equity Curve",
                      save_path: str = None):
    """Plot cumulative returns (equity curve)."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 9),
                              gridspec_kw={'height_ratios': [3, 1]},
                              sharex=True)

    # Equity curve
    cum_port = (1 + portfolio_returns).cumprod()
    axes[0].plot(cum_port.index, cum_port.values,
                 color=COLORS['portfolio'], linewidth=1.5,
                 label='Portfolio')

    if benchmark_returns is not None:
        common = portfolio_returns.index.intersection(benchmark_returns.index)
        cum_bench = (1 + benchmark_returns[common]).cumprod()
        axes[0].plot(cum_bench.index, cum_bench.values,
                     color=COLORS['benchmark'], linewidth=1.2,
                     linestyle='--', label='Benchmark')

    axes[0].set_title(title, fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Cumulative Return')
    axes[0].legend(loc='upper left')

    # Drawdown
    rolling_max = cum_port.expanding().max()
    drawdown = cum_port / rolling_max - 1
    axes[1].fill_between(drawdown.index, drawdown.values, 0,
                         color=COLORS['drawdown'], alpha=0.4)
    axes[1].set_ylabel('Drawdown')
    axes[1].set_xlabel('Date')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()


def plot_rolling_metrics(rolling_df: pd.DataFrame,
                         title: str = "Rolling Performance Metrics",
                         save_path: str = None):
    """Plot rolling Sharpe, volatility, and drawdown."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    # Rolling Sharpe
    if 'sharpe' in rolling_df.columns:
        axes[0].plot(rolling_df.index, rolling_df['sharpe'],
                     color=COLORS['portfolio'])
        axes[0].axhline(y=0, color='gray', linestyle='-', alpha=0.5)
        axes[0].axhline(y=1, color=COLORS['positive'],
                         linestyle='--', alpha=0.5, label='Sharpe=1')
        axes[0].set_ylabel('Rolling Sharpe')
        axes[0].set_title(title, fontsize=14, fontweight='bold')
        axes[0].legend()

    # Rolling Volatility
    if 'volatility' in rolling_df.columns:
        axes[1].plot(rolling_df.index, rolling_df['volatility'],
                     color=COLORS['accent'])
        axes[1].set_ylabel('Rolling Volatility')

    # Rolling Drawdown
    if 'drawdown' in rolling_df.columns:
        axes[2].fill_between(rolling_df.index, rolling_df['drawdown'], 0,
                             color=COLORS['drawdown'], alpha=0.4)
        axes[2].set_ylabel('Drawdown')
        axes[2].set_xlabel('Date')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()


def plot_factor_attribution(attribution: Dict,
                             save_path: str = None):
    """Plot factor exposures as a bar chart."""
    if 'factor_betas' not in attribution:
        return

    betas = attribution['factor_betas']
    t_stats = attribution.get('factor_t_stats', {})

    fig, ax = plt.subplots(figsize=(10, 6))

    names = list(betas.keys())
    values = [betas[n] for n in names]
    colors = [COLORS['positive'] if v > 0 else COLORS['negative']
              for v in values]

    bars = ax.barh(names, values, color=colors, alpha=0.8)

    # Add t-stat annotations
    for i, (name, val) in enumerate(zip(names, values)):
        t = t_stats.get(name, 0)
        sig = "*" if abs(t) > 2 else ""
        ax.text(val + 0.005 * np.sign(val), i,
                f't={t:.1f}{sig}', va='center', fontsize=10)

    ax.axvline(x=0, color='gray', linestyle='-', alpha=0.5)
    ax.set_xlabel('Factor Beta')
    ax.set_title(f"Factor Attribution  |  Alpha: "
                 f"{attribution.get('alpha_annualized', 0):.2%} "
                 f"(t={attribution.get('alpha_t_stat', 0):.2f})",
                 fontsize=13, fontweight='bold')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()


def plot_weights(weights: pd.Series,
                 sector_map: Dict[str, str] = None,
                 title: str = "Portfolio Weights",
                 save_path: str = None):
    """Plot portfolio weight distribution."""
    weights = weights[weights > 0].sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(10, max(6, len(weights) * 0.3)))

    if sector_map:
        sectors = [sector_map.get(t, 'Unknown') for t in weights.index]
        unique_sectors = list(set(sectors))
        sector_colors = plt.cm.Set3(np.linspace(0, 1, len(unique_sectors)))
        color_map = dict(zip(unique_sectors, sector_colors))
        colors = [color_map[s] for s in sectors]
    else:
        colors = COLORS['portfolio']

    ax.barh(weights.index, weights.values, color=colors, alpha=0.8)
    ax.axvline(x=1/len(weights), color='gray', linestyle='--',
               alpha=0.5, label='Equal Weight')
    ax.set_xlabel('Weight')
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()


def plot_bootstrap_distribution(bootstrap_results: Dict,
                                 save_path: str = None):
    """Plot bootstrap Sharpe ratio distribution."""
    fig, ax = plt.subplots(figsize=(10, 6))

    sharpe_data = bootstrap_results.get('sharpe', {})
    mean = sharpe_data.get('mean', 0)
    ci_lo = sharpe_data.get('ci_lower', 0)
    ci_hi = sharpe_data.get('ci_upper', 0)

    # Simulate distribution for plotting
    n = bootstrap_results.get('n_bootstrap', 10000)
    std = sharpe_data.get('std', 0.5)
    samples = np.random.normal(mean, std, n)

    ax.hist(samples, bins=80, density=True,
            color=COLORS['portfolio'], alpha=0.6, edgecolor='white')
    ax.axvline(mean, color=COLORS['accent'], linewidth=2,
               label=f'Mean: {mean:.3f}')
    ax.axvline(ci_lo, color=COLORS['negative'], linestyle='--',
               label=f'95% CI: [{ci_lo:.3f}, {ci_hi:.3f}]')
    ax.axvline(ci_hi, color=COLORS['negative'], linestyle='--')
    ax.axvline(0, color='gray', linewidth=1.5)

    ax.set_xlabel('Sharpe Ratio')
    ax.set_ylabel('Density')
    ax.set_title('Bootstrap Distribution of Sharpe Ratio',
                 fontsize=13, fontweight='bold')
    ax.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()
