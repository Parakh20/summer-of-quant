"""Tunable parameters for the volume-spike breakout strategy.

Caller: strategy/signals.py reads these fields to parameterize indicator
lengths, filters, and trailing-stop sizing. No data files touched.
Created per user Step 5 instruction: 'Identify all tuneable parameters in
the strategy (thresholds, lookback windows, position sizing coefficients)'.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class StrategyParams:
    atr_length: int = 14
    sma_length: int = 50
    rsi_length: int = 14
    rsi_overbought: float = 70.0
    rsi_oversold: float = 30.0
    vol_window: int = 6
    vol_std_mult: float = 1.5
    trailing_stop_mult: float = 2.0

    @property
    def warmup(self) -> int:
        """Number of leading bars needed before all indicators are valid."""
        return max(self.sma_length, self.atr_length, self.rsi_length, self.vol_window)
