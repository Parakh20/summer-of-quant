"""Volume-spike breakout strategy: signal generation.

Ported from notebooks/main.ipynb's process_data()/strat(). Logic is
unchanged (same entries, exits, trailing stop, reversal rules); the
pandas_ta calls are replaced with strategy/indicators.py, the inner loop
reads numpy arrays instead of per-cell `.loc` lookups for speed, and all
magic numbers are pulled from StrategyParams so the strategy is tunable.

Caller: backtest/runner.py calls generate_signals(df, params) to produce
the 'signals' column consumed by backtest/backtester.py.
"""

import numpy as np
import pandas as pd

from strategy.indicators import atr as calc_atr
from strategy.indicators import rsi as calc_rsi
from strategy.indicators import sma as calc_sma
from strategy.params import StrategyParams


def process_data(data: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
    """Add ATR/SMA/RSI indicator columns. Does not drop NaNs or reset the
    index, so the leading `params.warmup` rows stay aligned with raw data."""
    data = data.copy()
    data["ATR"] = calc_atr(data["high"], data["low"], data["close"], length=params.atr_length)
    data["SMA"] = calc_sma(data["close"], length=params.sma_length)
    data["RSI"] = calc_rsi(data["close"], length=params.rsi_length)
    return data


def generate_signals(data: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
    """Implements the volume-spike breakout strategy with an SMA trend
    filter, an RSI overbought/oversold filter, and an ATR trailing stop.

    signals: 1 = open/close long, -1 = open/close short,
             2 = reverse short->long, -2 = reverse long->short.
    """
    data = process_data(data, params)

    n = len(data)
    close = data["close"].to_numpy()
    open_ = data["open"].to_numpy()
    volume = data["volume"].to_numpy()
    atr = data["ATR"].to_numpy()
    sma = data["SMA"].to_numpy()
    rsi = data["RSI"].to_numpy()

    signals = np.zeros(n, dtype=int)
    trade_type = np.full(n, "HOLD", dtype=object)
    position_before = np.zeros(n, dtype=int)

    position = 0
    trailing_stop = 0.0
    warmup = params.warmup

    for i in range(warmup, n):
        position_before[i] = position
        c = close[i]
        a = atr[i]

        window = volume[max(0, i - params.vol_window + 1) : i + 1]
        vol_spike_threshold = np.mean(window) + params.vol_std_mult * np.std(window)

        if position == 1:
            trailing_stop = max(trailing_stop, c - a * params.trailing_stop_mult)
            if c < trailing_stop:
                signals[i] = -1
                position = 0
                trailing_stop = 0.0
                trade_type[i] = "CLOSE_TSL"
                continue
        elif position == -1:
            trailing_stop = min(trailing_stop, c + a * params.trailing_stop_mult)
            if c > trailing_stop:
                signals[i] = 1
                position = 0
                trailing_stop = 0.0
                trade_type[i] = "CLOSE_TSL"
                continue

        is_bullish_spike = volume[i] > vol_spike_threshold and c > open_[i]
        is_bearish_spike = volume[i] > vol_spike_threshold and c < open_[i]

        is_uptrend = c > sma[i]
        is_downtrend = c < sma[i]
        is_not_overbought = rsi[i] < params.rsi_overbought
        is_not_oversold = rsi[i] > params.rsi_oversold

        if position == 0 and is_bullish_spike and is_uptrend and is_not_overbought:
            signals[i] = 1
            position = 1
            trade_type[i] = "LONG"
            trailing_stop = c - a * params.trailing_stop_mult
        elif position == 0 and is_bearish_spike and is_downtrend and is_not_oversold:
            signals[i] = -1
            position = -1
            trade_type[i] = "SHORT"
            trailing_stop = c + a * params.trailing_stop_mult
        elif position == 1 and is_bearish_spike and is_downtrend and is_not_oversold:
            signals[i] = -2
            position = -1
            trade_type[i] = "REVERSE_LONG_TO_SHORT"
            trailing_stop = c + a * params.trailing_stop_mult
        elif position == -1 and is_bullish_spike and is_uptrend and is_not_overbought:
            signals[i] = 2
            position = 1
            trade_type[i] = "REVERSE_SHORT_TO_LONG"
            trailing_stop = c - a * params.trailing_stop_mult

    data["signals"] = signals
    data["trade_type"] = trade_type
    data["position_before"] = position_before
    return data


def trim_to_flat_start(signals_df: pd.DataFrame) -> pd.DataFrame:
    """Drop leading rows until the strategy is flat (position_before == 0).

    Needed before backtesting an arbitrary slice of a signals dataframe
    (e.g. an in-sample/out-of-sample or walk-forward split): BackTester
    always starts a fresh slice with no open position, so any leading rows
    that assumed a position carried over from before the slice would
    produce an "invalid signal for current position" error.
    """
    flat_mask = signals_df["position_before"] == 0
    if not flat_mask.any():
        return signals_df.iloc[0:0]
    first_flat_idx = flat_mask.idxmax()
    return signals_df.loc[first_flat_idx:]
