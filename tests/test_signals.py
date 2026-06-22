import numpy as np
import pandas as pd
import pytest

from strategy.params import StrategyParams
from strategy.signals import generate_signals, trim_to_flat_start


def _make_ohlcv(n=200, seed=0):
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    high = close + rng.uniform(0, 1, n)
    low = close - rng.uniform(0, 1, n)
    open_ = close + rng.normal(0, 0.5, n)
    volume = rng.uniform(100, 200, n)
    # inject a few volume spikes so the strategy actually trades
    spike_idx = rng.choice(np.arange(60, n), size=10, replace=False)
    volume[spike_idx] *= 5
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    return pd.DataFrame(
        {"datetime": dates, "open": open_, "high": high, "low": low, "close": close, "volume": volume}
    )


def test_generate_signals_returns_same_length_and_required_columns():
    df = _make_ohlcv()
    params = StrategyParams()
    result = generate_signals(df, params)
    assert len(result) == len(df)
    assert "signals" in result.columns
    assert "trade_type" in result.columns


def test_warmup_period_has_no_signals():
    df = _make_ohlcv()
    params = StrategyParams()
    result = generate_signals(df, params)
    assert (result["signals"].iloc[: params.warmup] == 0).all()


def test_signals_are_valid_values_only():
    df = _make_ohlcv()
    params = StrategyParams()
    result = generate_signals(df, params)
    assert set(result["signals"].unique()).issubset({-2, -1, 0, 1, 2})


def test_no_lookahead_bias_truncated_prefix_matches_full_run():
    """Signals up to index i must depend only on data up to and including i."""
    df = _make_ohlcv(n=150)
    params = StrategyParams()

    full_result = generate_signals(df, params)

    cutoff = 120
    truncated_result = generate_signals(df.iloc[: cutoff + 1].copy(), params)

    pd.testing.assert_series_equal(
        full_result["signals"].iloc[: cutoff + 1],
        truncated_result["signals"],
        check_names=False,
    )


def test_no_trades_before_warmup_even_with_extreme_volume():
    df = _make_ohlcv(n=80)
    df.loc[5, "volume"] = 1_000_000
    params = StrategyParams(sma_length=50)
    result = generate_signals(df, params)
    assert (result["signals"].iloc[:50] == 0).all()


def test_trim_to_flat_start_drops_leading_open_position_rows():
    df = _make_ohlcv(n=300, seed=3)
    params = StrategyParams()
    full = generate_signals(df, params)

    # pick a split point that lands mid-position somewhere in the series
    split_idx = 150
    sliced = full.iloc[split_idx:].copy()
    trimmed = trim_to_flat_start(sliced)

    assert trimmed["position_before"].iloc[0] == 0
    assert len(trimmed) <= len(sliced)


def test_trim_to_flat_start_is_noop_when_already_flat():
    df = _make_ohlcv(n=120, seed=9)
    params = StrategyParams()
    full = generate_signals(df, params)

    trimmed = trim_to_flat_start(full)
    assert trimmed["position_before"].iloc[0] == 0
    assert len(trimmed) == len(full)
