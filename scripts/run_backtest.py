#!/usr/bin/env python3
"""Run the strategy on the official training data and save all results.

Usage (from repo root):
    python scripts/run_backtest.py
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from backtest.metrics import (  # noqa: E402
    compute_all_metrics,
    plot_equity_curve,
    plot_monthly_returns_heatmap,
    save_metrics_csv,
)
from backtest.runner import run_backtest  # noqa: E402
from data.loader import load_training_data  # noqa: E402
from strategy.params import StrategyParams  # noqa: E402

RESULTS_DIR = REPO_ROOT / "results"


def main() -> None:
    df = load_training_data()
    params = StrategyParams()

    print(f"Loaded {len(df)} rows of training data ({df['datetime'].min()} to {df['datetime'].max()})")
    print(f"Running strategy with params: {params}")

    bt, signals_df = run_backtest(df, params)
    print(f"Total trades: {len(bt.trades)}")

    signals_df.to_csv(RESULTS_DIR / "signals.csv", index=False)

    stats = bt.get_statistics()
    if stats is None:
        print("No trades were executed. Nothing to report.")
        return

    print("\n--- BackTester native statistics ---")
    for key, val in stats.items():
        print(f"{key:.<35}: {val}")

    metrics = compute_all_metrics(bt)
    print("\n--- Equity-curve based metrics ---")
    for key, val in metrics.items():
        print(f"{key:.<35}: {val}")

    save_metrics_csv(metrics, RESULTS_DIR / "metrics.csv")
    plot_equity_curve(bt, RESULTS_DIR / "equity_curve.png")
    plot_monthly_returns_heatmap(bt, RESULTS_DIR / "monthly_returns_heatmap.png")

    print(f"\nSaved metrics.csv, equity_curve.png, monthly_returns_heatmap.png, signals.csv to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
