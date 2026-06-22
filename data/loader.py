"""Load and normalize the OHLCV CSVs shipped with this project.

The two provided CSVs have different schemas (one has a stray index column
and no `Adj Close`, the other has `Adj Close`). This module normalizes both
to a single canonical schema: datetime, open, high, low, close, volume.
"""

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent

REQUIRED_COLUMNS = ["datetime", "open", "high", "low", "close", "volume"]


def load_ohlcv(path: str | Path) -> pd.DataFrame:
    """Load an OHLCV CSV and return a clean dataframe with REQUIRED_COLUMNS.

    Drops any extra columns (e.g. a stray index column, 'Adj Close'),
    sorts ascending by datetime, drops duplicate timestamps, and resets
    the index to a plain integer range.
    """
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    column_map = {c.lower(): c for c in df.columns}
    if "datetime" not in column_map:
        raise ValueError(f"{path}: missing 'datetime' column")

    df = df.rename(columns={column_map[c]: c for c in column_map if c in REQUIRED_COLUMNS})
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing required columns {missing}")

    df = df[REQUIRED_COLUMNS].copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").drop_duplicates(subset="datetime")
    df = df.reset_index(drop=True)

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)

    return df


def load_training_data() -> pd.DataFrame:
    """The official competition training set (2019-09-08 to 2023)."""
    return load_ohlcv(DATA_DIR / "BTC_2019_2023_1d.csv")


def load_extended_history() -> pd.DataFrame:
    """Earlier BTC history (2014-2019), useful as an extra robustness check
    but not part of the competition's official training/test data."""
    return load_ohlcv(DATA_DIR / "btc_2014_2019_1d.csv")
