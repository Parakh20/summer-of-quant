"""Integration test: strategy signals -> BackTester -> metrics/plots,
exercising backtest/runner.py and backtest/metrics.py end-to-end against
the real (unmodified) BackTester.
"""

import numpy as np
import pandas as pd

from backtest.metrics import compute_all_metrics, plot_equity_curve, plot_monthly_returns_heatmap, save_metrics_csv
from backtest.runner import run_backtest
from strategy.params import StrategyParams


def _make_trending_ohlcv(n=400, seed=7):
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.normal(0.05, 1, n))
    high = close + rng.uniform(0, 1, n)
    low = close - rng.uniform(0, 1, n)
    open_ = close + rng.normal(0, 0.5, n)
    volume = rng.uniform(100, 200, n)
    spike_idx = rng.choice(np.arange(60, n), size=40, replace=False)
    volume[spike_idx] *= 5
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    return pd.DataFrame(
        {"datetime": dates, "open": open_, "high": high, "low": low, "close": close, "volume": volume}
    )


def test_run_backtest_executes_trades_and_returns_signals():
    df = _make_trending_ohlcv()
    bt, signals_df = run_backtest(df, StrategyParams())
    assert len(signals_df) == len(df)
    assert len(bt.trades) > 0


def test_compute_all_metrics_returns_finite_values():
    df = _make_trending_ohlcv()
    bt, _ = run_backtest(df, StrategyParams())
    metrics = compute_all_metrics(bt)

    for key in ["sharpe_ratio", "sortino_ratio", "max_drawdown_pct", "cagr_pct", "cumulative_return_pct"]:
        assert np.isfinite(metrics[key]), f"{key} is not finite: {metrics[key]}"
    assert metrics["total_trades"] == len(bt.trades)


def test_save_metrics_csv_writes_readable_file(tmp_path):
    df = _make_trending_ohlcv()
    bt, _ = run_backtest(df, StrategyParams())
    metrics = compute_all_metrics(bt)

    out_path = tmp_path / "metrics.csv"
    save_metrics_csv(metrics, out_path)

    assert out_path.exists()
    loaded = pd.read_csv(out_path)
    assert loaded.shape[0] == 1
    assert "sharpe_ratio" in loaded.columns


def test_plot_functions_write_png_files(tmp_path):
    df = _make_trending_ohlcv()
    bt, _ = run_backtest(df, StrategyParams())

    equity_path = tmp_path / "equity.png"
    heatmap_path = tmp_path / "heatmap.png"

    plot_equity_curve(bt, equity_path)
    plot_monthly_returns_heatmap(bt, heatmap_path)

    assert equity_path.exists()
    assert equity_path.stat().st_size > 0
    # heatmap needs >=2 months of data; with 400 days it should exist too
    assert heatmap_path.exists()


def test_run_backtest_is_deterministic_for_same_inputs():
    df = _make_trending_ohlcv()
    params = StrategyParams()
    bt1, signals1 = run_backtest(df, params)
    bt2, signals2 = run_backtest(df, params)
    pd.testing.assert_series_equal(signals1["signals"], signals2["signals"], check_names=False)
    assert len(bt1.trades) == len(bt2.trades)
