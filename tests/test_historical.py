"""Tests for the databento drop-in `Historical` client."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import date, datetime
from pathlib import Path
from unittest.mock import MagicMock
from zoneinfo import ZoneInfo

import polars as pl
import pytest

from dbn_cache import Historical
from dbn_cache.exceptions import MissingAPIKeyError

UTC = ZoneInfo("UTC")
KEY = "db-testkey0000000000000000000000000000"

DownloadCall = tuple[str, date, date, "str | None"]


def _ohlcv(days: list[int]) -> pl.DataFrame:
    rows = [datetime(2024, 1, d, 14, 30, tzinfo=UTC) for d in days]
    return pl.DataFrame(
        {
            "ts_event": pl.Series(rows).cast(pl.Datetime("ns", "UTC")),
            "close": [100.0 + d for d in days],
            "symbol": ["ESH5" for _ in days],
        }
    )


def _make_client(
    tmp_path: Path, days: list[int], monkeypatch: pytest.MonkeyPatch
) -> tuple[Historical, list[DownloadCall]]:
    client = Historical(key=KEY, cache_dir=tmp_path)
    calls: list[DownloadCall] = []

    def mock(
        symbol: str,
        schema: str,
        start: date,
        end: date,
        dataset: str,
        dest: Path,
        stype: str | None = None,
    ) -> None:
        calls.append((symbol, start, end, stype))
        _ohlcv(days).write_parquet(dest)

    monkeypatch.setattr(client.cache, "_download_partition", mock)
    return client, calls


def _noop(_record: object) -> None:
    return None


class TestGetRange:
    def test_returns_cachestore_with_data(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        client, _ = _make_client(tmp_path, [2, 4, 10], monkeypatch)
        store = client.timeseries.get_range(
            dataset="GLBX.MDP3",
            symbols="ES.c.0",
            schema="ohlcv-1m",
            start="2024-01-01",
            end="2024-02-01",
        )
        df = store.to_df()
        assert type(store).__name__ == "CacheStore"
        assert len(df) == 3

    def test_threads_stype_in(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        client, calls = _make_client(tmp_path, [2], monkeypatch)
        client.timeseries.get_range(
            dataset="GLBX.MDP3",
            symbols="ES.c.0",
            schema="ohlcv-1m",
            start="2024-01-01",
            end="2024-02-01",
            stype_in="continuous",
        )
        assert calls[0][3] == "continuous"

    def test_second_call_is_cache_hit(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        client, calls = _make_client(tmp_path, [2, 4, 10], monkeypatch)
        for _ in range(2):
            client.timeseries.get_range(
                dataset="GLBX.MDP3",
                symbols="ES.c.0",
                schema="ohlcv-1m",
                start="2024-01-01",
                end="2024-02-01",
            ).to_df()
        assert len(calls) == 1

    def test_exclusive_end(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        client, _ = _make_client(tmp_path, [2, 4, 31], monkeypatch)
        # end=2024-01-31 is exclusive -> Jan 31 excluded.
        store = client.timeseries.get_range(
            dataset="GLBX.MDP3",
            symbols="ES.c.0",
            schema="ohlcv-1m",
            start="2024-01-01",
            end="2024-01-31",
        )
        days = sorted({ts.day for ts in store.to_polars()["ts_event"].to_list()})
        assert days == [2, 4]

    def test_intraday_exclusive_end(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        client = Historical(key=KEY, cache_dir=tmp_path)

        def mock(
            symbol: str,
            schema: str,
            start: date,
            end: date,
            dataset: str,
            dest: Path,
            stype: str | None = None,
        ) -> None:
            rows = [
                datetime(2024, 1, 5, 9, 0, tzinfo=UTC),
                datetime(2024, 1, 5, 15, 0, tzinfo=UTC),
            ]
            pl.DataFrame(
                {"ts_event": pl.Series(rows).cast(pl.Datetime("ns", "UTC"))}
            ).write_parquet(dest)

        monkeypatch.setattr(client.cache, "_download_partition", mock)
        store = client.timeseries.get_range(
            dataset="GLBX.MDP3",
            symbols="ES.c.0",
            schema="ohlcv-1m",
            start="2024-01-05",
            end="2024-01-05T12:00:00",  # exclusive -> only the 09:00 row
        )
        df = store.to_polars()
        assert df.height == 1
        assert df["ts_event"][0].hour == 9

    def test_multiple_symbols(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        client, calls = _make_client(tmp_path, [2, 4], monkeypatch)
        store = client.timeseries.get_range(
            dataset="GLBX.MDP3",
            symbols=["ES.c.0", "NQ.c.0"],
            schema="ohlcv-1m",
            start="2024-01-01",
            end="2024-02-01",
        )
        assert {c[0] for c in calls} == {"ES.c.0", "NQ.c.0"}
        assert store.to_polars().height == 4

    def test_multiple_symbols_time_sorted(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # databento returns records ordered by ts_event; a multi-symbol union
        # must be re-sorted (not ES-block-then-NQ-block).
        client, _ = _make_client(tmp_path, [2, 4], monkeypatch)
        store = client.timeseries.get_range(
            dataset="GLBX.MDP3",
            symbols=["ES.c.0", "NQ.c.0"],
            schema="ohlcv-1m",
            start="2024-01-01",
            end="2024-02-01",
        )
        ts = [t.day for t in store.to_polars()["ts_event"].to_list()]
        assert ts == sorted(ts)
        assert ts == [2, 2, 4, 4]

    def test_generator_symbols_not_exhausted_on_passthrough(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        client = Historical(key=KEY, cache_dir=tmp_path)
        mock_real = MagicMock()
        monkeypatch.setattr(client, "_real", mock_real)

        def gen() -> Iterator[int]:
            yield 12345  # instrument id -> uncacheable, forces passthrough

        client.timeseries.get_range(
            dataset="GLBX.MDP3",
            symbols=gen(),
            schema="ohlcv-1m",
            start="2024-01-01",
            end="2024-02-01",
        )
        _, kwargs = mock_real.timeseries.get_range.call_args
        assert list(kwargs["symbols"]) == [12345]

    def test_to_parquet_and_csv(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        client, _ = _make_client(tmp_path, [2, 4, 10], monkeypatch)
        store = client.timeseries.get_range(
            dataset="GLBX.MDP3",
            symbols="ES.c.0",
            schema="ohlcv-1m",
            start="2024-01-01",
            end="2024-02-01",
        )
        pq = tmp_path / "out.parquet"
        csv = tmp_path / "out.csv"
        store.to_parquet(pq)
        store.to_csv(csv)
        assert pl.read_parquet(pq).height == 3
        assert csv.read_text().count("\n") >= 3


class TestPassthrough:
    def test_metadata_forwards_to_real_client(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        client = Historical(key=KEY, cache_dir=tmp_path)
        mock_real = MagicMock()
        mock_real.metadata.get_cost.return_value = 4.2
        monkeypatch.setattr(client, "_real", mock_real)
        assert client.metadata.get_cost(dataset="GLBX.MDP3") == 4.2

    def test_all_symbols_bypasses_cache(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        client = Historical(key=KEY, cache_dir=tmp_path)
        mock_real = MagicMock()
        sentinel = object()
        mock_real.timeseries.get_range.return_value = sentinel
        monkeypatch.setattr(client, "_real", mock_real)
        result = client.timeseries.get_range(
            dataset="GLBX.MDP3",
            symbols="ALL_SYMBOLS",
            schema="ohlcv-1m",
            start="2024-01-01",
            end="2024-02-01",
        )
        assert result is sentinel
        mock_real.timeseries.get_range.assert_called_once()


class TestConstruction:
    def test_missing_key_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("DATABENTO_API_KEY", raising=False)
        with pytest.raises(MissingAPIKeyError):
            Historical(cache_dir=tmp_path)

    def test_key_from_env(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("DATABENTO_API_KEY", KEY)
        client = Historical(cache_dir=tmp_path)
        assert client.cache is not None

    def test_defaults_to_sqlite_backend(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from dbn_cache.storage.sql import SqlBackend

        monkeypatch.delenv("DBN_CACHE_URL", raising=False)
        monkeypatch.delenv("DATABENTO_CACHE_URL", raising=False)
        client = Historical(key=KEY, cache_dir=tmp_path)
        assert isinstance(client.cache.backend, SqlBackend)
        assert (tmp_path / "cache.db").exists()

    def test_file_url_selects_filesystem(self, tmp_path: Path) -> None:
        from dbn_cache.storage.filesystem import FilesystemBackend

        client = Historical(key=KEY, url=f"file://{tmp_path}/fs")
        assert isinstance(client.cache.backend, FilesystemBackend)


class TestUnsupported:
    def test_raw_dbn_ops_raise(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        client, _ = _make_client(tmp_path, [2], monkeypatch)
        store = client.timeseries.get_range(
            dataset="GLBX.MDP3",
            symbols="ES.c.0",
            schema="ohlcv-1m",
            start="2024-01-01",
            end="2024-02-01",
        )
        with pytest.raises(NotImplementedError):
            store.to_file("x.dbn")
        with pytest.raises(NotImplementedError):
            store.replay(_noop)
