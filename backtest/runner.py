"""Glue layer between strategy signal generation and the (unmodified)
BackTester from backtest/backtester.py.

BackTester only accepts a CSV file path (it calls pd.read_csv internally),
so this module writes the signal dataframe to a temporary CSV and points
BackTester at it, then cleans up. backtester.py itself is never edited.

Caller: scripts/run_backtest.py, scripts/optimise.py, scripts/walk_forward.py.
"""

import os
import tempfile

import pandas as pd

from backtest.backtester import BackTester
from strategy.params import StrategyParams
from strategy.signals import generate_signals


def run_backtest_from_signals(
    signals_df: pd.DataFrame,
    initial_capital: float = 1000.0,
    compound: bool = True,
) -> BackTester:
    """Run BackTester on an already-generated signals dataframe.

    Useful for in-sample/out-of-sample and walk-forward splits: generate
    signals once on the full series (so indicators have proper warmup with
    no lookahead), then slice the resulting dataframe and backtest each
    slice independently with this function.
    """
    fd, tmp_path = tempfile.mkstemp(suffix=".csv")
    os.close(fd)
    try:
        signals_df.to_csv(tmp_path, index=False)
        bt = BackTester(
            "BTC",
            signal_data_path=tmp_path,
            master_file_path=tmp_path,
            compound_flag=1 if compound else 0,
        )
        bt.get_trades(initial_capital)
    finally:
        os.remove(tmp_path)

    return bt


def run_backtest(
    df: pd.DataFrame,
    params: StrategyParams,
    initial_capital: float = 1000.0,
    compound: bool = True,
) -> tuple[BackTester, pd.DataFrame]:
    """Generate signals for `df` and run them through BackTester.

    Returns (BackTester instance with trades populated, the signals dataframe).
    """
    signals_df = generate_signals(df, params)
    bt = run_backtest_from_signals(signals_df, initial_capital, compound)
    return bt, signals_df
