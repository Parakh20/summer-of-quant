import pandas as pd
import pytest

from data.loader import REQUIRED_COLUMNS, load_ohlcv


def test_load_ohlcv_normalizes_schema_with_index_column(tmp_path):
    csv_path = tmp_path / "indexed.csv"
    csv_path.write_text(
        ",datetime,open,high,low,close,volume\n"
        "0,2020-01-01,1,2,0,1.5,100\n"
        "1,2020-01-02,1.5,2.5,1,2,150\n"
    )
    df = load_ohlcv(csv_path)
    assert list(df.columns) == REQUIRED_COLUMNS
    assert len(df) == 2


def test_load_ohlcv_normalizes_schema_with_adj_close(tmp_path):
    csv_path = tmp_path / "adjclose.csv"
    csv_path.write_text(
        "datetime,open,high,low,close,Adj Close,volume\n"
        "2020-01-01,1,2,0,1.5,1.5,100\n"
        "2020-01-02,1.5,2.5,1,2,2,150\n"
    )
    df = load_ohlcv(csv_path)
    assert list(df.columns) == REQUIRED_COLUMNS
    assert "Adj Close" not in df.columns


def test_load_ohlcv_sorts_and_dedupes(tmp_path):
    csv_path = tmp_path / "unsorted.csv"
    csv_path.write_text(
        "datetime,open,high,low,close,volume\n"
        "2020-01-02,2,2,2,2,2\n"
        "2020-01-01,1,1,1,1,1\n"
        "2020-01-01,9,9,9,9,9\n"
    )
    df = load_ohlcv(csv_path)
    assert len(df) == 2
    assert df["datetime"].is_monotonic_increasing


def test_load_ohlcv_raises_on_missing_columns(tmp_path):
    csv_path = tmp_path / "broken.csv"
    csv_path.write_text("datetime,open,high\n2020-01-01,1,2\n")
    with pytest.raises(ValueError):
        load_ohlcv(csv_path)
