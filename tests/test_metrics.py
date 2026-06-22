import numpy as np
import pandas as pd
import pytest

from backtest.metrics import cagr, max_drawdown, sharpe_ratio, sortino_ratio, trade_stats


def _dt_index(n, freq="D"):
    return pd.date_range("2020-01-01", periods=n, freq=freq)


def test_sharpe_ratio_zero_for_constant_returns():
    returns = pd.Series([0.0] * 10)
    assert sharpe_ratio(returns) == 0.0


def test_sharpe_ratio_positive_for_consistently_positive_returns():
    rng = np.random.default_rng(1)
    returns = pd.Series(0.01 + rng.normal(0, 0.001, 200))
    assert sharpe_ratio(returns) > 0


def test_sharpe_ratio_negative_for_consistently_negative_returns():
    rng = np.random.default_rng(1)
    returns = pd.Series(-0.01 + rng.normal(0, 0.001, 200))
    assert sharpe_ratio(returns) < 0


def test_sortino_ratio_zero_when_no_downside():
    returns = pd.Series([0.01, 0.02, 0.03, 0.01])
    assert sortino_ratio(returns) == 0.0


def test_sortino_ratio_finite_with_mixed_returns():
    returns = pd.Series([0.02, -0.01, 0.03, -0.02, 0.01])
    result = sortino_ratio(returns)
    assert np.isfinite(result)


def test_max_drawdown_zero_for_monotonically_increasing_equity():
    capital = pd.Series([1000, 1100, 1200, 1300], index=_dt_index(4))
    dd, duration = max_drawdown(capital)
    assert dd == 0.0
    assert duration == 0


def test_max_drawdown_detects_known_drop():
    capital = pd.Series([1000, 1200, 600, 900, 1300], index=_dt_index(5))
    dd, duration = max_drawdown(capital)
    assert dd == pytest.approx(50.0)
    assert duration == 2


def test_max_drawdown_duration_counts_consecutive_underwater_days():
    capital = pd.Series([1000, 800, 700, 900, 1100, 1000], index=_dt_index(6))
    dd, duration = max_drawdown(capital)
    assert duration == 3


def test_cagr_doubles_in_one_year():
    capital = pd.Series([1000, 2000], index=pd.to_datetime(["2020-01-01", "2021-01-01"]))
    result = cagr(capital)
    assert result == pytest.approx(1.0, rel=0.02)


def test_cagr_zero_for_flat_equity():
    capital = pd.Series([1000, 1000], index=pd.to_datetime(["2020-01-01", "2021-01-01"]))
    result = cagr(capital)
    assert result == pytest.approx(0.0, abs=1e-9)


class _FakeTrade:
    def __init__(self, pnl_value, holding_days):
        self._pnl = pnl_value
        self._holding = pd.Timedelta(days=holding_days)

    def pnl(self):
        return self._pnl

    def is_win(self):
        return self._pnl > 0

    def holding_time(self):
        return self._holding


class _FakeBackTester:
    def __init__(self, trades):
        self.trades = trades


def test_trade_stats_with_no_trades():
    stats = trade_stats(_FakeBackTester([]))
    assert stats["total_trades"] == 0
    assert stats["win_rate_pct"] == 0.0


def test_trade_stats_computes_win_rate_and_avg_win_loss():
    trades = [_FakeTrade(100, 1), _FakeTrade(-50, 2), _FakeTrade(200, 3), _FakeTrade(-25, 1)]
    stats = trade_stats(_FakeBackTester(trades))
    assert stats["total_trades"] == 4
    assert stats["win_rate_pct"] == pytest.approx(50.0)
    assert stats["avg_win"] == pytest.approx(150.0)
    assert stats["avg_loss"] == pytest.approx(-37.5)
    assert stats["avg_holding_period"] == pd.Timedelta(days=1.75)
