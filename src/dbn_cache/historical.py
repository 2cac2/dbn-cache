"""A caching drop-in for :class:`databento.Historical`.

``dbn_cache.Historical`` mirrors ``databento.Historical`` so existing databento
code works unchanged, but transparently caches ``timeseries.get_range`` results::

    import dbn_cache as db

    client = db.Historical("YOUR_KEY")                       # or DATABENTO_API_KEY
    data = client.timeseries.get_range(
        dataset="GLBX.MDP3", symbols="ES.c.0", schema="ohlcv-1m",
        start="2024-01-01", end="2024-02-01",
    )
    df = data.to_df()                                        # served from cache

``metadata``, ``symbology``, ``batch`` (and any other attribute) pass through to a
real ``databento.Historical``. Only ``timeseries.get_range`` is cached.

The result is a :class:`CacheStore`, a DBNStore-compatible object backed by the
columnar cache: it supports ``to_df`` / ``to_ndarray`` / ``to_parquet`` /
``to_csv`` / ``to_json`` / iteration. Operations that require the raw DBN binary
(``to_file``, ``replay``, ``request_symbology`` ...) raise ``NotImplementedError``
— use the underlying ``databento.Historical`` for those.
"""

from __future__ import annotations

import os
from datetime import UTC, date, datetime, timedelta
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import polars as pl

from .cache import DataCache
from .client import DatabentoClient
from .exceptions import MissingAPIKeyError
from .models import CachedData

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator
    from pathlib import Path

    import pandas as pd

    from .storage.base import StorageBackend

_NANOS_PER_SECOND = 1_000_000_000


def _coerce_datetime(value: object) -> datetime:
    """Coerce a databento-style time value into a naive/aware datetime."""
    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day)
    if isinstance(value, int):
        # Databento treats bare ints as UNIX nanoseconds.
        return datetime.fromtimestamp(value / 1e9, tz=UTC)
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return datetime.combine(date.fromisoformat(value), datetime.min.time())
    # pandas.Timestamp or anything with to_pydatetime()
    to_pydatetime = getattr(value, "to_pydatetime", None)
    if callable(to_pydatetime):
        result = to_pydatetime()
        if isinstance(result, datetime):
            return result
    msg = f"Unsupported time value: {value!r}"
    raise TypeError(msg)


def _epoch_ns(dt: datetime) -> int:
    """Nanoseconds since the UNIX epoch (naive datetimes are treated as UTC)."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return int(dt.timestamp() * _NANOS_PER_SECOND)


def _resolve_start(start: object) -> tuple[int, date]:
    """Return (inclusive start ns, start date). Bare ints are UNIX nanoseconds."""
    if isinstance(start, int):
        return start, datetime.fromtimestamp(start / 1e9, tz=UTC).date()
    start_dt = _coerce_datetime(start)
    return _epoch_ns(start_dt), start_dt.date()


def _resolve_end(end: object) -> tuple[int, date]:
    """Return (exclusive end ns, inclusive end date). Bare ints are UNIX ns.

    databento's end is exclusive; the inclusive end date is that of the last
    instant actually included (end - 1 ns).
    """
    if isinstance(end, int):
        return end, datetime.fromtimestamp((end - 1) / 1e9, tz=UTC).date()
    end_dt = _coerce_datetime(end)
    return _epoch_ns(end_dt), (end_dt - timedelta(microseconds=1)).date()


class CacheStore:
    """DBNStore-compatible view over cached rows for a get_range request.

    Backed by the columnar cache. Timestamps are trimmed to the exact requested
    ``[start, end)`` window (databento's end is exclusive), so intraday bounds are
    honoured precisely.
    """

    def __init__(
        self,
        cached: list[CachedData],
        start_ns: int | None,
        end_ns: int | None,
        *,
        dataset: str,
        schema: str,
        symbols: list[str],
        stype_in: str,
        stype_out: str,
    ) -> None:
        self._cached = cached
        self._start_ns = start_ns
        self._end_ns = end_ns
        self._dataset = dataset
        self._schema = schema
        self._symbols = symbols
        self._stype_in = stype_in
        self._stype_out = stype_out

    # -- data access --------------------------------------------------------
    def _frame(self) -> pl.LazyFrame:
        frames = [cd.to_polars() for cd in self._cached]
        if not frames:
            return pl.LazyFrame()
        if len(frames) == 1:
            lf = frames[0]
        else:
            lf = pl.concat(frames, how="vertical_relaxed")

        schema = lf.collect_schema()
        if "ts_event" in schema:
            dtype = schema["ts_event"]
            is_datetime = dtype == pl.Datetime or str(dtype).startswith("Datetime")
            ts_ns = (
                pl.col("ts_event").dt.epoch("ns") if is_datetime else pl.col("ts_event")
            )
            if self._start_ns is not None:
                lf = lf.filter(ts_ns >= self._start_ns)
            if self._end_ns is not None:
                lf = lf.filter(ts_ns < self._end_ns)  # databento end is exclusive
            if len(self._cached) > 1:
                # databento returns records ordered by ts_event; a multi-symbol
                # union must be re-sorted to preserve that contract.
                lf = lf.sort("ts_event")
        return lf

    def to_polars(self) -> pl.DataFrame:
        """Return the cached rows as a Polars DataFrame (dbn-cache extension)."""
        return self._frame().collect()

    def to_df(self, *args: object, **kwargs: object) -> pd.DataFrame:
        """Return the cached rows as a pandas DataFrame (like ``DBNStore.to_df``)."""
        return self._frame().collect().to_pandas()

    def to_ndarray(self, schema: str | None = None, count: int | None = None) -> Any:
        """Return the cached rows as a NumPy structured array."""
        return self._frame().collect().to_pandas().to_records(index=False)

    def to_parquet(self, path: str | Path, *args: object, **kwargs: object) -> None:
        """Write the cached rows to a Parquet file."""
        self._frame().collect().write_parquet(path)

    def to_csv(self, path: str | Path, *args: object, **kwargs: object) -> None:
        """Write the cached rows to a CSV file."""
        self._frame().collect().write_csv(path)

    def to_json(self, path: str | Path, *args: object, **kwargs: object) -> None:
        """Write the cached rows to a newline-delimited JSON file."""
        self._frame().collect().write_ndjson(path)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        yield from self._frame().collect().iter_rows(named=True)

    # -- metadata-ish properties -------------------------------------------
    @property
    def metadata(self) -> SimpleNamespace:
        """A lightweight metadata view (subset of ``DBNStore.metadata``)."""
        return SimpleNamespace(
            dataset=self._dataset,
            schema=self._schema,
            symbols=self._symbols,
            stype_in=self._stype_in,
            stype_out=self._stype_out,
            start=self._start_ns,
            end=self._end_ns,
        )

    @property
    def dataset(self) -> str:
        return self._dataset

    @property
    def schema(self) -> str:
        return self._schema

    @property
    def symbols(self) -> list[str]:
        return self._symbols

    @property
    def stype_in(self) -> str:
        return self._stype_in

    @property
    def stype_out(self) -> str:
        return self._stype_out

    # -- unsupported (raw DBN binary) --------------------------------------
    def _unsupported(self, name: str) -> NotImplementedError:
        return NotImplementedError(
            f"{name} is not supported on a cached CacheStore (the cache stores "
            "columnar rows, not raw DBN bytes). Use databento.Historical directly "
            "for raw DBN access."
        )

    def to_file(self, *args: object, **kwargs: object) -> None:
        raise self._unsupported("to_file")

    def replay(self, *args: object, **kwargs: object) -> None:
        raise self._unsupported("replay")

    def request_symbology(self, *args: object, **kwargs: object) -> None:
        raise self._unsupported("request_symbology")

    def request_full_definitions(self, *args: object, **kwargs: object) -> None:
        raise self._unsupported("request_full_definitions")

    def __repr__(self) -> str:
        return (
            f"CacheStore(dataset={self._dataset!r}, schema={self._schema!r}, "
            f"symbols={self._symbols!r})"
        )


class _CachedTimeseries:
    """Cached mirror of ``databento.Historical.timeseries``."""

    def __init__(self, cache: DataCache, real_getter: Callable[[], Any]) -> None:
        self._cache = cache
        self._get_real = real_getter

    def get_range(
        self,
        dataset: str,
        start: object,
        end: object = None,
        symbols: Iterable[str | int] | str | int | None = None,
        schema: str = "trades",
        stype_in: str = "raw_symbol",
        stype_out: str = "instrument_id",
        limit: int | None = None,
        path: str | Path | None = None,
    ) -> Any:
        """Cached ``timeseries.get_range`` (see :mod:`dbn_cache.historical`)."""
        # Materialize an iterable of symbols once so that, on the uncacheable
        # passthrough path, we don't forward an already-exhausted iterator.
        if symbols is not None and not isinstance(symbols, (str, int)):
            symbols = list(symbols)
        symbol_list = _normalize_symbols(symbols)
        if symbol_list is None:
            # Uncacheable (ALL_SYMBOLS / instrument ids / None): pass through.
            return self._get_real().timeseries.get_range(
                dataset=dataset,
                start=start,
                end=end,
                symbols=symbols,
                schema=schema,
                stype_in=stype_in,
                stype_out=stype_out,
                limit=limit,
                path=path,
            )

        start_ns, start_date = _resolve_start(start)

        if end is None:
            inclusive_end = self._cache.available_end(dataset)
            if inclusive_end is None:
                inclusive_end = datetime.now(UTC).date() - timedelta(days=1)
            end_ns = None
        else:
            end_ns, inclusive_end = _resolve_end(end)

        cached: list[CachedData] = []
        for symbol in symbol_list:
            cd = self._cache.ensure(
                symbol, schema, start_date, inclusive_end, dataset, stype=stype_in
            )
            cached.append(cd)

        return CacheStore(
            cached,
            start_ns,
            end_ns,
            dataset=dataset,
            schema=schema,
            symbols=symbol_list,
            stype_in=stype_in,
            stype_out=stype_out,
        )


def _normalize_symbols(
    symbols: Iterable[str | int] | str | int | None,
) -> list[str] | None:
    """Return a list of string symbols, or None if the request can't be cached."""
    if symbols is None:
        return None
    if isinstance(symbols, str):
        if symbols.strip().upper() in ("ALL_SYMBOLS", "*"):
            return None
        return [symbols]
    if isinstance(symbols, int):
        return None  # instrument ids aren't cache-keyed
    result: list[str] = []
    for s in symbols:
        if not isinstance(s, str):
            return None
        if s.strip().upper() in ("ALL_SYMBOLS", "*"):
            return None
        result.append(s)
    return result or None


class Historical:
    """A caching drop-in for :class:`databento.Historical`."""

    def __init__(
        self,
        key: str | None = None,
        gateway: str | None = None,
        *,
        cache_dir: Path | None = None,
        storage: StorageBackend | None = None,
        url: str | None = None,
    ) -> None:
        """Create a caching historical client.

        Args:
            key: Databento API key. Falls back to the ``DATABENTO_API_KEY`` env var.
            gateway: Optional databento gateway override.
            cache_dir: Filesystem cache directory (filesystem backend).
            storage: An explicit storage backend (overrides ``url``).
            url: A SQL connection URL (e.g. ``sqlite:///cache.db``) to cache into a
                database instead of local files. Falls back to ``DBN_CACHE_URL``.
        """
        import databento as databento_mod

        resolved = key or os.environ.get("DATABENTO_API_KEY")
        if not resolved:
            raise MissingAPIKeyError
        self._key = resolved
        self._gateway = gateway

        if gateway is not None:
            self._real: Any = databento_mod.Historical(resolved, gateway)
        else:
            self._real = databento_mod.Historical(resolved)

        self._cache = DataCache(
            cache_dir=cache_dir,
            client=DatabentoClient(api_key=resolved),
            storage=storage,
            url=url,
        )
        self.timeseries = _CachedTimeseries(self._cache, lambda: self._real)

    @property
    def cache(self) -> DataCache:
        """The underlying :class:`~dbn_cache.cache.DataCache`."""
        return self._cache

    def __getattr__(self, name: str) -> Any:
        # Forward any non-cached attribute (metadata, symbology, batch, key,
        # gateway, ...) to the real databento client.
        real = self.__dict__.get("_real")
        if real is None:
            raise AttributeError(name)
        return getattr(real, name)
