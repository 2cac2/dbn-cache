"""Behavioral parity between the filesystem and SQL (SQLite) storage backends."""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import polars as pl
import pytest

from dbn_cache import DataCache
from dbn_cache.exceptions import EmptyDataError
from dbn_cache.models import CacheStatus

UTC = ZoneInfo("UTC")


def _ohlcv(days: list[int]) -> pl.DataFrame:
    """A databento-shaped OHLCV frame (tz-aware UTC Datetime ts_event)."""
    rows = [datetime(2024, 1, d, 14, 30, tzinfo=UTC) for d in days]
    return pl.DataFrame(
        {
            "ts_event": pl.Series(rows).cast(pl.Datetime("ns", "UTC")),
            "open": [100.0 + d for d in days],
            "close": [101.0 + d for d in days],
            "volume": [1000 + d for d in days],
            "symbol": ["ESH5" for _ in days],
        }
    )


@pytest.fixture(params=["fs", "sqlite"])
def cache(request: pytest.FixtureRequest, tmp_path: Path) -> DataCache:
    if request.param == "fs":
        return DataCache(cache_dir=tmp_path)
    return DataCache(url=f"sqlite:///{tmp_path / 'cache.db'}", cache_dir=tmp_path)


def _mock_ohlcv(cache: DataCache, days: list[int]) -> list[tuple[date, date]]:
    calls: list[tuple[date, date]] = []

    def mock(
        symbol: str,
        schema: str,
        start: date,
        end: date,
        dataset: str,
        dest: Path,
        stype: str | None = None,
    ) -> None:
        calls.append((start, end))
        _ohlcv(days).write_parquet(dest)

    cache._download_partition = mock  # type: ignore[method-assign]
    return calls


class TestBackendParity:
    def test_download_and_read(self, cache: DataCache) -> None:
        _mock_ohlcv(cache, [2, 15, 28])
        data = cache.download("ES.c.0", "ohlcv-1m", date(2024, 1, 1), date(2024, 1, 31))
        df = data.to_polars().collect()
        assert df.height == 3
        assert set(df["close"].to_list()) == {103.0, 116.0, 129.0}

    def test_cache_hit_no_redownload(self, cache: DataCache) -> None:
        calls = _mock_ohlcv(cache, [2, 15, 28])
        cache.download("ES.c.0", "ohlcv-1m", date(2024, 1, 1), date(2024, 1, 31))
        assert len(calls) == 1
        # Second get() must not download.
        got = cache.get("ES.c.0", "ohlcv-1m", date(2024, 1, 1), date(2024, 1, 31))
        assert got.to_polars().collect().height == 3
        assert len(calls) == 1

    def test_date_filtering(self, cache: DataCache) -> None:
        _mock_ohlcv(cache, [2, 15, 28])
        cache.download("ES.c.0", "ohlcv-1m", date(2024, 1, 1), date(2024, 1, 31))
        only = cache.get("ES.c.0", "ohlcv-1m", date(2024, 1, 15), date(2024, 1, 15))
        df = only.to_polars().collect()
        assert df.height == 1
        assert df["close"][0] == 116.0

    def test_check_cache_status(self, cache: DataCache) -> None:
        _mock_ohlcv(cache, [2, 15, 28])
        empty = cache.check_cache(
            "ES.c.0", "ohlcv-1m", date(2024, 1, 1), date(2024, 1, 31)
        )
        assert empty.status == CacheStatus.EMPTY
        cache.download("ES.c.0", "ohlcv-1m", date(2024, 1, 1), date(2024, 1, 31))
        full = cache.check_cache(
            "ES.c.0", "ohlcv-1m", date(2024, 1, 1), date(2024, 1, 31)
        )
        assert full.status == CacheStatus.COMPLETE

    def test_info_and_list(self, cache: DataCache) -> None:
        _mock_ohlcv(cache, [2, 15, 28])
        cache.download("ES.c.0", "ohlcv-1m", date(2024, 1, 1), date(2024, 1, 31))
        info = cache.info("ES.c.0", "ohlcv-1m")
        assert info is not None
        assert info.symbol == "ES.c.0"
        assert info.ranges[0].start == date(2024, 1, 1)
        assert info.size_bytes > 0
        listed = cache.list_cached()
        assert [(i.symbol, i.schema_) for i in listed] == [("ES.c.0", "ohlcv-1m")]

    def test_clear_cache(self, cache: DataCache) -> None:
        _mock_ohlcv(cache, [2, 15, 28])
        cache.download("ES.c.0", "ohlcv-1m", date(2024, 1, 1), date(2024, 1, 31))
        deleted = cache.clear_cache(
            "ES.c.0", "ohlcv-1m", date(2024, 1, 1), date(2024, 1, 31)
        )
        assert deleted == 1
        assert cache.info("ES.c.0", "ohlcv-1m") is None
        assert cache.list_cached() == []

    def test_empty_download_raises(self, cache: DataCache) -> None:
        def mock(
            symbol: str,
            schema: str,
            start: date,
            end: date,
            dataset: str,
            dest: Path,
            stype: str | None = None,
        ) -> None:
            _ohlcv([]).write_parquet(dest)

        cache._download_partition = mock  # type: ignore[method-assign]
        with pytest.raises(EmptyDataError):
            cache.download("ZZZ.c.0", "ohlcv-1m", date(2024, 1, 1), date(2024, 1, 31))

    def test_tick_schema_daily_partitions(self, cache: DataCache) -> None:
        def mock(
            symbol: str,
            schema: str,
            start: date,
            end: date,
            dataset: str,
            dest: Path,
            stype: str | None = None,
        ) -> None:
            _ohlcv([start.day]).select(["ts_event", "close", "symbol"]).write_parquet(
                dest
            )

        cache._download_partition = mock  # type: ignore[method-assign]
        data = cache.download("ES.c.0", "trades", date(2024, 1, 8), date(2024, 1, 9))
        assert data.to_polars().collect().height == 2
        chk = cache.check_cache("ES.c.0", "trades", date(2024, 1, 8), date(2024, 1, 9))
        assert chk.status == CacheStatus.COMPLETE

    def test_repair_metadata(self, cache: DataCache) -> None:
        _mock_ohlcv(cache, [2, 15, 28])
        cache.download("ES.c.0", "ohlcv-1m", date(2024, 1, 1), date(2024, 1, 31))
        # Drop metadata but keep the data -> repair should rebuild it.
        cache.backend.delete_meta("GLBX.MDP3", "ES.c.0", "ohlcv-1m")
        assert cache.info("ES.c.0", "ohlcv-1m") is None
        repaired = cache.repair_metadata()
        assert ("GLBX.MDP3", "ES.c.0", "ohlcv-1m") in repaired
        info = cache.info("ES.c.0", "ohlcv-1m")
        assert info is not None
        assert info.ranges[0].start == date(2024, 1, 2)
        assert info.ranges[0].end == date(2024, 1, 28)

    def test_validate_metadata_no_false_positive(self, cache: DataCache) -> None:
        # Data spans the full requested month, so metadata (partition range)
        # matches the actual data range and validate reports no mismatch.
        _mock_ohlcv(cache, [1, 31])
        cache.download("ES.c.0", "ohlcv-1m", date(2024, 1, 1), date(2024, 1, 31))
        assert cache.validate_metadata() == []
