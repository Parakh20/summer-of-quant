"""Performance metrics and plots computed on top of BackTester's output,
without modifying backtester.py itself.

backtester.py's own get_statistics() reports a per-trade Sharpe and leaves
Sortino as None. This module instead works off the day-by-day equity curve
(bt.data['capital'], built by bt.calc_capital()) to compute standard
annualized Sharpe/Sortino/CAGR/drawdown-duration, plus trade-level stats,
and saves everything to results/.

Caller: scripts/run_backtest.py, scripts/optimise.py, scripts/walk_forward.py.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from backtest.backtester import BackTester

TRADING_DAYS_PER_YEAR = 365


def equity_curve(bt: BackTester) -> pd.Series:
    bt.calc_capital()
    return bt.data["capital"]


def daily_returns(capital: pd.Series) -> pd.Series:
    return capital.pct_change().dropna()


def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    if returns.std() == 0 or len(returns) == 0:
        return 0.0
    excess = returns - risk_free_rate / TRADING_DAYS_PER_YEAR
    return float(excess.mean() / returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR))


def sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    downside = returns[returns < 0]
    downside_std = downside.std()
    if downside_std == 0 or pd.isna(downside_std) or len(returns) == 0:
        return 0.0
    excess = returns - risk_free_rate / TRADING_DAYS_PER_YEAR
    return float(excess.mean() / downside_std * np.sqrt(TRADING_DAYS_PER_YEAR))


def max_drawdown(capital: pd.Series) -> tuple[float, int]:
    """Returns (max drawdown %, max drawdown duration in days)."""
    running_max = capital.cummax()
    drawdown = (capital - running_max) / running_max
    max_dd = float(abs(drawdown.min())) * 100

    underwater = drawdown < 0
    duration = 0
    max_duration = 0
    for is_under in underwater:
        duration = duration + 1 if is_under else 0
        max_duration = max(max_duration, duration)

    return max_dd, max_duration


def cagr(capital: pd.Series) -> float:
    if len(capital) < 2 or capital.iloc[0] <= 0:
        return 0.0
    days = (capital.index[-1] - capital.index[0]).days
    if days <= 0:
        return 0.0
    total_return = capital.iloc[-1] / capital.iloc[0]
    years = days / 365.25
    return float(total_return ** (1 / years) - 1) if total_return > 0 else -1.0


def trade_stats(bt: BackTester) -> dict:
    trades = bt.trades
    if not trades:
        return {
            "total_trades": 0,
            "win_rate_pct": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "avg_holding_period": pd.Timedelta(0),
        }

    wins = [t.pnl() for t in trades if t.is_win()]
    losses = [t.pnl() for t in trades if not t.is_win()]
    holding_periods = [t.holding_time() for t in trades]

    return {
        "total_trades": len(trades),
        "win_rate_pct": len(wins) / len(trades) * 100,
        "avg_win": float(np.mean(wins)) if wins else 0.0,
        "avg_loss": float(np.mean(losses)) if losses else 0.0,
        "avg_holding_period": sum(holding_periods, pd.Timedelta(0)) / len(holding_periods),
    }


def compute_all_metrics(bt: BackTester) -> dict:
    capital = equity_curve(bt)
    returns = daily_returns(capital)
    max_dd, max_dd_duration = max_drawdown(capital)

    metrics = {
        "sharpe_ratio": sharpe_ratio(returns),
        "sortino_ratio": sortino_ratio(returns),
        "max_drawdown_pct": max_dd,
        "max_drawdown_duration_days": max_dd_duration,
        "cagr_pct": cagr(capital) * 100,
        "cumulative_return_pct": (capital.iloc[-1] / capital.iloc[0] - 1) * 100 if len(capital) else 0.0,
        "final_capital": float(capital.iloc[-1]) if len(capital) else 0.0,
    }
    metrics.update(trade_stats(bt))
    return metrics


def save_metrics_csv(metrics: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([metrics]).to_csv(path, index=False)


def plot_equity_curve(bt: BackTester, path: str | Path) -> None:
    capital = equity_curve(bt)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(capital.index, capital.values, color="tab:blue", label="Equity ($)")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Equity ($)", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.plot(bt.data.index, bt.data["close"], color="gray", alpha=0.5, label="BTC Close")
    ax2.set_ylabel("BTC Close Price", color="gray")
    ax2.tick_params(axis="y", labelcolor="gray")

    fig.suptitle("Equity Curve vs. BTC Price")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_monthly_returns_heatmap(bt: BackTester, path: str | Path) -> None:
    capital = equity_curve(bt)
    monthly = capital.resample("ME").last().pct_change().dropna() * 100
    if monthly.empty:
        return

    table = monthly.to_frame("return_pct")
    table["year"] = table.index.year
    table["month"] = table.index.month
    pivot = table.pivot(index="year", columns="month", values="return_pct")
    pivot = pivot.reindex(columns=range(1, 13))

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, max(3, 0.5 * len(pivot))))
    im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto", vmin=-20, vmax=20)
    ax.set_xticks(range(12))
    ax.set_xticklabels(
        ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    )
    ax.set_yticks(range(len(pivot)))
    ax.set_yticklabels(pivot.index)
    for r in range(pivot.shape[0]):
        for c in range(pivot.shape[1]):
            val = pivot.values[r, c]
            if not np.isnan(val):
                ax.text(c, r, f"{val:.1f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="Monthly Return (%)")
    ax.set_title("Monthly Returns Heatmap (%)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
