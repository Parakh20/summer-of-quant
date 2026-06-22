import numpy as np
import pandas as pd

from strategy.indicators import atr, rsi, sma


def test_sma_matches_manual_rolling_mean():
    series = pd.Series([1, 2, 3, 4, 5, 6])
    result = sma(series, length=3)
    expected = pd.Series([np.nan, np.nan, 2.0, 3.0, 4.0, 5.0])
    pd.testing.assert_series_equal(result, expected, check_names=False)


def test_sma_first_n_minus_1_values_are_nan():
    series = pd.Series(range(10), dtype=float)
    result = sma(series, length=4)
    assert result.iloc[:3].isna().all()
    assert not result.iloc[3:].isna().any()


def test_rsi_is_100_for_strictly_increasing_series():
    series = pd.Series(np.arange(1, 30, dtype=float))
    result = rsi(series, length=14)
    assert result.iloc[-1] == 100.0


def test_rsi_is_0_for_strictly_decreasing_series():
    series = pd.Series(np.arange(30, 1, -1, dtype=float))
    result = rsi(series, length=14)
    assert result.iloc[-1] == 0.0


def test_rsi_stays_within_0_100_bounds():
    rng = np.random.default_rng(42)
    series = pd.Series(100 + np.cumsum(rng.normal(0, 1, 200)))
    result = rsi(series, length=14).dropna()
    assert (result >= 0).all()
    assert (result <= 100).all()


def test_atr_is_nonnegative():
    rng = np.random.default_rng(0)
    close = pd.Series(100 + np.cumsum(rng.normal(0, 1, 100)))
    high = close + rng.uniform(0, 2, 100)
    low = close - rng.uniform(0, 2, 100)
    result = atr(high, low, close, length=14).dropna()
    assert (result >= 0).all()


def test_atr_zero_for_constant_price_series():
    close = pd.Series([100.0] * 30)
    high = close.copy()
    low = close.copy()
    result = atr(high, low, close, length=14).dropna()
    assert np.allclose(result, 0.0)
