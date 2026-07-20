"""Storage backend abstraction for the data cache.

A :class:`StorageBackend` encapsulates *where* and *how* cached partitions and
metadata live. The default :class:`~dbn_cache.storage.filesystem.FilesystemBackend`
stores partitioned Parquet files on disk (the historical behaviour); the
:class:`~dbn_cache.storage.sql.SqlBackend` stores columnar rows in a SQL database
so the cache can be centralized and shared.

``DataCache`` keeps all calendar/date logic and talks to a backend purely through
this interface, so the two never need to know about each other's storage details.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from ..utils import normalize_symbol

if TYPE_CHECKING:
    from datetime import date
    from pathlib import Path

    import polars as pl

    from ..models import SymbolMeta


@dataclass(frozen=True)
class PartitionKey:
    """Identifies a single cache partition, independent of storage layout.

    ``day`` is ``None`` for month-granularity partitions (OHLCV/statistics) and
    set for day-granularity partitions (tick schemas), mirroring the on-disk
    partition scheme.
    """

    dataset: str
    symbol: str
    schema: str
    year: int
    month: int
    day: int | None = None

    @property
    def symbol_normalized(self) -> str:
        """Symbol normalized the same way filesystem paths are (``.`` -> ``_``)."""
        return normalize_symbol(self.symbol)

    @property
    def granularity(self) -> str:
        """``"daily"`` for tick partitions, ``"monthly"`` otherwise."""
        return "daily" if self.day is not None else "monthly"


@runtime_checkable
class DataReader(Protocol):
    """A lazy handle to cached rows for a requested date range.

    Backends return a reader from :meth:`StorageBackend.read_range`; ``CachedData``
    consumes it. ``scan()`` yields a Polars ``LazyFrame`` of the (coarsely
    range-filtered) rows; exact inclusive-date trimming is applied by ``CachedData``.
    """

    def scan(self) -> pl.LazyFrame:
        """Return a LazyFrame over the cached rows (may be empty)."""
        ...

    @property
    def paths(self) -> list[Path]:
        """Backing file paths, or an empty list for non-file backends."""
        ...


class StorageBackend(ABC):
    """Abstract storage for cached partitions and per-symbol metadata."""

    @property
    @abstractmethod
    def cache_dir(self) -> Path:
        """Filesystem location associated with this backend.

        For the filesystem backend this is the cache root; for SQL backends it is
        the directory used for sidecar lock files (and, for SQLite, the database
        file's parent).
        """

    # -- partition data -----------------------------------------------------
    @abstractmethod
    def commit_partition(self, key: PartitionKey, src_parquet: Path) -> None:
        """Persist a materialized temp Parquet file as ``key``'s partition.

        Overwrites any existing data for that partition (idempotent re-download).
        """

    @abstractmethod
    def partition_exists(self, key: PartitionKey) -> bool:
        """Return whether the partition has been fetched and stored."""

    @abstractmethod
    def delete_partition(self, key: PartitionKey) -> bool:
        """Delete a partition's data. Return ``True`` if anything was removed."""

    @abstractmethod
    def read_range(
        self, dataset: str, symbol: str, schema: str, start: date, end: date
    ) -> DataReader:
        """Return a :class:`DataReader` for the stored rows within ``[start, end]``."""

    # -- metadata -----------------------------------------------------------
    @abstractmethod
    def load_meta(self, dataset: str, symbol: str, schema: str) -> SymbolMeta | None:
        """Load a symbol/schema's metadata, or ``None`` if absent."""

    @abstractmethod
    def save_meta(self, meta: SymbolMeta) -> None:
        """Persist a symbol/schema's metadata."""

    @abstractmethod
    def delete_meta(self, dataset: str, symbol: str, schema: str) -> None:
        """Delete a symbol/schema's metadata if present."""

    # -- introspection / repair --------------------------------------------
    @abstractmethod
    def list_keys(self, dataset: str | None = None) -> list[tuple[str, str, str]]:
        """List ``(dataset, symbol_key, schema)`` triples that have metadata.

        ``symbol_key`` is accepted as-is by :meth:`load_meta` (normalized form).
        """

    @abstractmethod
    def list_orphans(self, dataset: str | None = None) -> list[tuple[str, str, str]]:
        """List ``(dataset, symbol_key, schema)`` triples with data but no metadata."""

    @abstractmethod
    def size_bytes(self, dataset: str, symbol: str, schema: str) -> int:
        """Approximate stored size in bytes for a symbol/schema."""

    @abstractmethod
    def actual_data_range(
        self, dataset: str, symbol: str, schema: str
    ) -> tuple[date, date] | None:
        """Return the min/max data date across all stored partitions, or ``None``."""

    # -- housekeeping -------------------------------------------------------
    @abstractmethod
    def lock(
        self, dataset: str, symbol: str, schema: str, timeout: float = 300
    ) -> AbstractContextManager[None]:
        """Return a context manager serializing writers for a symbol/schema."""

    @abstractmethod
    def cleanup(self, dataset: str, symbol: str, schema: str) -> None:
        """Remove empty/stale artifacts for a symbol/schema (best effort)."""
