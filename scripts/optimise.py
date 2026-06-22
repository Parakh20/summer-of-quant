#!/usr/bin/env python3
"""Grid search over the strategy's most impactful parameters.

In-sample = first 80% of the official training data (chronological).
Out-of-sample = last 20%, held out and only evaluated once, for the single
best in-sample combination found.

Signals are generated once on the FULL series per param combo (so every
indicator has proper historical warmup with no lookahead - see
tests/test_signals.py::test_no_lookahead_bias_truncated_prefix_matches_full_run)
and only then split into in-sample / out-of-sample slices for independent
backtesting.

This script only REPORTS results. It does not modify strategy/params.py.
Usage (from repo root):
    python scripts/optimise.py
"""

import itertools
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd  # noqa: E402

from backtest.metrics import compute_all_metrics  # noqa: E402
from backtest.runner import run_backtest_from_signals  # noqa: E402
from data.loader import load_training_data  # noqa: E402
from strategy.params import StrategyParams  # noqa: E402
from strategy.signals import generate_signals, trim_to_flat_start  # noqa: E402

RESULTS_DIR = REPO_ROOT / "results"

SMA_LENGTHS = [30, 50, 75, 100]
VOL_STD_MULTS = [1.0, 1.5, 2.0, 2.5]
TRAILING_STOP_MULTS = [1.5, 2.0, 2.5, 3.0]

OVERFIT_RATIO_THRESHOLD = 1.5  # warn if in-sample Sharpe is this many times the OOS Sharpe


def evaluate_combo(df: pd.DataFrame, params: StrategyParams, split_idx: int) -> dict:
    signals_df = generate_signals(df, params)

    in_sample = trim_to_flat_start(signals_df.iloc[:split_idx].copy())
    out_of_sample = trim_to_flat_start(signals_df.iloc[split_idx:].copy())

    is_bt = run_backtest_from_signals(in_sample)
    oos_bt = run_backtest_from_signals(out_of_sample)

    is_metrics = compute_all_metrics(is_bt)
    oos_metrics = compute_all_metrics(oos_bt)

    return {
        "sma_length": params.sma_length,
        "vol_std_mult": params.vol_std_mult,
        "trailing_stop_mult": params.trailing_stop_mult,
        "is_sharpe": is_metrics["sharpe_ratio"],
        "is_cagr_pct": is_metrics["cagr_pct"],
        "is_max_dd_pct": is_metrics["max_drawdown_pct"],
        "is_trades": is_metrics["total_trades"],
        "oos_sharpe": oos_metrics["sharpe_ratio"],
        "oos_cagr_pct": oos_metrics["cagr_pct"],
        "oos_max_dd_pct": oos_metrics["max_drawdown_pct"],
        "oos_trades": oos_metrics["total_trades"],
    }


def main() -> None:
    df = load_training_data()
    split_idx = int(len(df) * 0.8)
    print(f"In-sample: rows 0..{split_idx} ({df['datetime'].iloc[0]} to {df['datetime'].iloc[split_idx - 1]})")
    print(f"Out-of-sample: rows {split_idx}..{len(df)} ({df['datetime'].iloc[split_idx]} to {df['datetime'].iloc[-1]})")

    base = StrategyParams()
    combos = list(itertools.product(SMA_LENGTHS, VOL_STD_MULTS, TRAILING_STOP_MULTS))
    print(f"\nSweeping {len(combos)} combinations of sma_length x vol_std_mult x trailing_stop_mult...")

    rows = []
    for sma_length, vol_std_mult, trailing_stop_mult in combos:
        params = StrategyParams(
            atr_length=base.atr_length,
            sma_length=sma_length,
            rsi_length=base.rsi_length,
            rsi_overbought=base.rsi_overbought,
            rsi_oversold=base.rsi_oversold,
            vol_window=base.vol_window,
            vol_std_mult=vol_std_mult,
            trailing_stop_mult=trailing_stop_mult,
        )
        rows.append(evaluate_combo(df, params, split_idx))

    sweep_df = pd.DataFrame(rows).sort_values("is_sharpe", ascending=False).reset_index(drop=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    sweep_df.to_csv(RESULTS_DIR / "param_sweep.csv", index=False)

    print("\nTop 10 combinations by in-sample Sharpe:")
    print(sweep_df.head(10).to_string(index=False))

    best = sweep_df.iloc[0]
    print("\n=== Best in-sample combination ===")
    print(
        f"sma_length={int(best['sma_length'])}, vol_std_mult={best['vol_std_mult']}, "
        f"trailing_stop_mult={best['trailing_stop_mult']}"
    )
    print(f"In-sample Sharpe: {best['is_sharpe']:.3f} | Out-of-sample Sharpe: {best['oos_sharpe']:.3f}")
    print(f"In-sample CAGR: {best['is_cagr_pct']:.1f}% | Out-of-sample CAGR: {best['oos_cagr_pct']:.1f}%")
    print(f"In-sample trades: {int(best['is_trades'])} | Out-of-sample trades: {int(best['oos_trades'])}")

    if best["oos_sharpe"] <= 0 < best["is_sharpe"]:
        print(
            "\n⚠️  OVERFITTING WARNING: out-of-sample Sharpe is non-positive while "
            "in-sample Sharpe is positive. This combination likely overfits the "
            "in-sample period and should not be trusted as-is."
        )
    elif best["is_sharpe"] > OVERFIT_RATIO_THRESHOLD * max(best["oos_sharpe"], 1e-9):
        print(
            f"\n⚠️  OVERFITTING WARNING: in-sample Sharpe ({best['is_sharpe']:.2f}) is more than "
            f"{OVERFIT_RATIO_THRESHOLD}x the out-of-sample Sharpe ({best['oos_sharpe']:.2f}). "
            "Treat the in-sample result with skepticism."
        )
    else:
        print("\nNo strong overfitting signal: out-of-sample Sharpe is reasonably close to in-sample Sharpe.")

    print(
        "\nThis script only reports results — strategy/params.py defaults have NOT been changed. "
        "Confirm with the user before adopting these parameters permanently."
    )


if __name__ == "__main__":
    main()
