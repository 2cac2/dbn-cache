"""Tests for CachedData with a DataReader source and Datetime ts_event filtering."""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import polars as pl

from dbn_cache.models import CachedData

UTC = ZoneInfo("UTC")


class _FrameReader:
    """A minimal DataReader backed by an in-memory frame (no files)."""

    def __init__(self, df: pl.DataFrame) -> None:
        self._df = df

    def scan(self) -> pl.LazyFrame:
        return self._df.lazy()

    @property
    def paths(self) -> list[Path]:
        return []


class TestReaderSource:
    def test_reader_backed_cached_data(self) -> None:
        df = pl.DataFrame({"a": [1, 2, 3], "close": [10.0, 20.0, 30.0]})
        data = CachedData(_FrameReader(df))
        assert data.paths == []
        assert data.to_polars().collect().height == 3
        assert data.to_pandas().shape == (3, 2)

    def test_reader_date_filter_datetime_ts_event(self) -> None:
        rows = [
            datetime(2024, 1, 14, 12, tzinfo=UTC),
            datetime(2024, 1, 15, 12, tzinfo=UTC),
            datetime(2024, 1, 16, 12, tzinfo=UTC),
        ]
        df = pl.DataFrame(
            {
                "ts_event": pl.Series(rows).cast(pl.Datetime("ns", "UTC")),
                "close": [1.0, 2.0, 3.0],
            }
        )
        data = CachedData(
            _FrameReader(df), start=date(2024, 1, 15), end=date(2024, 1, 15)
        )
        out = data.to_polars().collect()
        assert out.height == 1
        assert out["close"][0] == 2.0


class TestDatetimeDtypeFilter:
    def test_naive_datetime_column_filter(self, tmp_path: Path) -> None:
        # Datetime (no tz) ts_event, written to parquet, filtered by date.
        rows = [datetime(2024, 3, d, 10, 0) for d in (10, 11, 12)]
        df = pl.DataFrame(
            {
                "ts_event": pl.Series(rows).cast(pl.Datetime("ns")),
                "v": [1, 2, 3],
            }
        )
        path = tmp_path / "d.parquet"
        df.write_parquet(path)
        data = CachedData([path], start=date(2024, 3, 11), end=date(2024, 3, 11))
        out = data.to_polars().collect()
        assert out.height == 1
        assert out["v"][0] == 2

    def test_int_ns_column_still_works(self, tmp_path: Path) -> None:
        # Backward compatibility: integer-nanosecond ts_event.
        ns = [int(datetime(2024, 3, d).timestamp() * 1e9) for d in (10, 11, 12)]
        df = pl.DataFrame({"ts_event": ns, "v": [1, 2, 3]})
        path = tmp_path / "i.parquet"
        df.write_parquet(path)
        data = CachedData([path], start=date(2024, 3, 11), end=date(2024, 3, 11))
        out = data.to_polars().collect()
        assert out.height == 1
        assert out["v"][0] == 2
