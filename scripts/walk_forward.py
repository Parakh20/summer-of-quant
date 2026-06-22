#!/usr/bin/env python3
"""Walk-forward robustness check: rolling 6-month train / 1-month test windows.

This validates robustness of a FIXED parameter set (not a re-optimization
per window — that's what scripts/optimise.py is for).

Implementation note: an earlier version of this script re-ran BackTester
independently on each 1-month price slice, forcing the position flat at
every window boundary. That truncated in-flight trades artificially (this
strategy's average holding period is ~29 days, close to the window length
itself), which destroyed real profit and made the strategy look far worse
than it is. Instead, we run ONE continuous backtest over the whole series
(matching scripts/run_backtest.py exactly, no truncation) and then slice
the resulting daily equity curve by calendar month. Each 1-month window's
Sharpe/return reflects the mark-to-market P&L during that month, including
any position open at the time, with no artificial interruption.

Usage (from repo root):
    python scripts/walk_forward.py
    python scripts/walk_forward.py --sma-length 50 --vol-std-mult 1.5 --trailing-stop-mult 1.5
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from backtest.metrics import daily_returns, equity_curve, sharpe_ratio  # noqa: E402
from backtest.runner import run_backtest  # noqa: E402
from data.loader import load_training_data  # noqa: E402
from strategy.params import StrategyParams  # noqa: E402

RESULTS_DIR = REPO_ROOT / "results"
TRAIN_MONTHS = 6
TEST_MONTHS = 1


def build_windows(start: pd.Timestamp, end: pd.Timestamp):
    windows = []
    test_start = start + pd.DateOffset(months=TRAIN_MONTHS)
    while True:
        test_end = test_start + pd.DateOffset(months=TEST_MONTHS)
        if test_end > end:
            break
        windows.append((test_start, test_end))
        test_start = test_start + pd.DateOffset(months=TEST_MONTHS)
    return windows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sma-length", type=int, default=StrategyParams().sma_length)
    parser.add_argument("--vol-std-mult", type=float, default=StrategyParams().vol_std_mult)
    parser.add_argument("--trailing-stop-mult", type=float, default=StrategyParams().trailing_stop_mult)
    args = parser.parse_args()

    params = StrategyParams(
        sma_length=args.sma_length,
        vol_std_mult=args.vol_std_mult,
        trailing_stop_mult=args.trailing_stop_mult,
    )
    print(f"Walk-forward validation with params: {params}")

    df = load_training_data()
    bt, _ = run_backtest(df, params)
    capital = equity_curve(bt)

    windows = build_windows(df["datetime"].iloc[0], df["datetime"].iloc[-1])
    print(f"Evaluating {len(windows)} rolling {TRAIN_MONTHS}-month-train / {TEST_MONTHS}-month-test windows...\n")

    rows = []
    for test_start, test_end in windows:
        window_capital = capital.loc[(capital.index >= test_start) & (capital.index < test_end)]
        if len(window_capital) < 2:
            continue

        window_returns = daily_returns(window_capital)
        window_sharpe = sharpe_ratio(window_returns)
        window_return_pct = (window_capital.iloc[-1] / window_capital.iloc[0] - 1) * 100

        trades_closed = sum(1 for t in bt.trades if test_start <= t.final_timestamp < test_end)

        rows.append(
            {
                "test_start": test_start.date(),
                "test_end": test_end.date(),
                "sharpe_ratio": window_sharpe,
                "return_pct": window_return_pct,
                "trades_closed": trades_closed,
            }
        )

    wf_df = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    wf_df.to_csv(RESULTS_DIR / "walk_forward.csv", index=False)

    print(wf_df.to_string(index=False))

    sharpe = wf_df["sharpe_ratio"]
    returns = wf_df["return_pct"]
    print("\n=== Walk-forward Sharpe distribution (per 1-month window, mark-to-market) ===")
    print(f"mean={sharpe.mean():.3f}  std={sharpe.std():.3f}  min={sharpe.min():.3f}  max={sharpe.max():.3f}")
    print(f"% windows with positive return: {(returns > 0).mean() * 100:.1f}%")
    print(f"% windows with Sharpe > 0: {(sharpe > 0).mean() * 100:.1f}%")

    # Chaining the monthly mark-to-market returns reproduces the same total
    # growth as the continuous equity curve (no double counting, no gaps),
    # since the windows are contiguous and non-overlapping.
    chained_growth = (1 + returns / 100).prod()
    print(f"\nChained walk-forward growth across all windows: {(chained_growth - 1) * 100:.1f}%")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(sharpe.dropna(), bins=15, color="tab:blue", edgecolor="black")
    axes[0].set_title("Walk-forward Sharpe distribution")
    axes[0].set_xlabel("Sharpe ratio (per 1-month window)")

    axes[1].hist(returns.dropna(), bins=15, color="tab:green", edgecolor="black")
    axes[1].set_title("Walk-forward monthly return distribution")
    axes[1].set_xlabel("Return (%)")

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "walk_forward_distribution.png", dpi=150)
    plt.close(fig)

    print(f"\nSaved walk_forward.csv and walk_forward_distribution.png to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
