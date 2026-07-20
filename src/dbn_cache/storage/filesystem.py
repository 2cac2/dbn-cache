"""Filesystem storage backend: partitioned Parquet files on disk.

This is the historical (default) storage layout and is a behaviour-preserving
extraction of the logic that previously lived directly in ``DataCache``.
"""

from __future__ import annotations

import json
import shutil
from contextlib import contextmanager
from datetime import UTC, date, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl
from filelock import FileLock

from ..models import SymbolMeta
from ..utils import (
    get_partition_path,
    is_tick_schema,
    iter_days,
    iter_months,
    normalize_symbol,
)
from .base import DataReader, PartitionKey, StorageBackend

if TYPE_CHECKING:
    from collections.abc import Iterator


def parquet_date_range(parquet_path: Path) -> tuple[date, date] | None:
    """Get the actual (min, max) data date from timestamps in a parquet file.

    Returns ``None`` if the file has no recognizable timestamp column or is empty.
    """
    df = pl.scan_parquet(parquet_path)
    schema = df.collect_schema()

    ts_col: str | None = None
    for col in ["ts_event", "ts"]:
        if col in schema:
            ts_col = col
            break

    if ts_col is None:
        return None

    col_type = schema[ts_col]
    is_datetime = col_type == pl.Datetime or str(col_type).startswith("Datetime")
    is_int = col_type == pl.Int64 or col_type == pl.UInt64

    if not is_datetime and not is_int:
        return None

    result = df.select(
        pl.col(ts_col).min().alias("min_ts"),
        pl.col(ts_col).max().alias("max_ts"),
    ).collect()

    if result.is_empty():
        return None

    min_ts = result["min_ts"][0]
    max_ts = result["max_ts"][0]

    if min_ts is None or max_ts is None:
        return None

    if is_datetime:
        return min_ts.date(), max_ts.date()
    # Int64/UInt64: nanoseconds since UNIX epoch (Databento format)
    start_date = datetime.fromtimestamp(min_ts / 1e9, tz=UTC).date()
    end_date = datetime.fromtimestamp(max_ts / 1e9, tz=UTC).date()
    return start_date, end_date


class ParquetReader:
    """A :class:`~dbn_cache.storage.base.DataReader` backed by parquet files."""

    def __init__(self, paths: list[Path]) -> None:
        self._paths = sorted(paths)

    def scan(self) -> pl.LazyFrame:
        if not self._paths:
            return pl.LazyFrame()
        return pl.scan_parquet(self._paths)

    @property
    def paths(self) -> list[Path]:
        return self._paths


class FilesystemBackend(StorageBackend):
    """Stores each partition as a Parquet file with a JSON metadata sidecar."""

    def __init__(self, cache_dir: Path) -> None:
        self._cache_dir = cache_dir

    @property
    def cache_dir(self) -> Path:
        return self._cache_dir

    # -- path helpers -------------------------------------------------------
    def _symbol_path(self, dataset: str, symbol: str, schema: str) -> Path:
        return self._cache_dir / dataset / normalize_symbol(symbol) / schema

    def _meta_path(self, dataset: str, symbol: str, schema: str) -> Path:
        return self._symbol_path(dataset, symbol, schema) / "meta.json"

    def _lock_path(self, dataset: str, symbol: str, schema: str) -> Path:
        return self._symbol_path(dataset, symbol, schema) / ".lock"

    def _partition_path(self, key: PartitionKey) -> Path:
        base = self._symbol_path(key.dataset, key.symbol, key.schema)
        return get_partition_path(base, key.schema, key.year, key.month, key.day)

    # -- partition data -----------------------------------------------------
    def commit_partition(self, key: PartitionKey, src_parquet: Path) -> None:
        dest = self._partition_path(key)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(src_parquet, dest)

    def partition_exists(self, key: PartitionKey) -> bool:
        return self._partition_path(key).exists()

    def delete_partition(self, key: PartitionKey) -> bool:
        path = self._partition_path(key)
        if path.exists():
            path.unlink()
            return True
        return False

    def read_range(
        self, dataset: str, symbol: str, schema: str, start: date, end: date
    ) -> DataReader:
        base_path = self._symbol_path(dataset, symbol, schema)
        files: list[Path] = []
        if is_tick_schema(schema):
            for d in iter_days(start, end):
                path = get_partition_path(base_path, schema, d.year, d.month, d.day)
                if path.exists():
                    files.append(path)
        else:
            for year, month in iter_months(start, end):
                path = get_partition_path(base_path, schema, year, month)
                if path.exists():
                    files.append(path)
        return ParquetReader(files)

    # -- metadata -----------------------------------------------------------
    def load_meta(self, dataset: str, symbol: str, schema: str) -> SymbolMeta | None:
        meta_path = self._meta_path(dataset, symbol, schema)
        if not meta_path.exists():
            return None
        with meta_path.open() as f:
            data = json.load(f)
        return SymbolMeta.model_validate(data)

    def save_meta(self, meta: SymbolMeta) -> None:
        meta_path = self._meta_path(meta.dataset, meta.symbol, meta.schema_)
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        with meta_path.open("w") as f:
            json.dump(meta.model_dump(by_alias=True), f, indent=2, default=str)

    def delete_meta(self, dataset: str, symbol: str, schema: str) -> None:
        meta_path = self._meta_path(dataset, symbol, schema)
        if meta_path.exists():
            meta_path.unlink()

    # -- introspection / repair --------------------------------------------
    def _iter_symbol_schema_dirs(
        self, dataset: str | None
    ) -> Iterator[tuple[str, Path]]:
        if dataset:
            datasets = [dataset]
        else:
            if not self._cache_dir.exists():
                return
            datasets = [d.name for d in self._cache_dir.iterdir() if d.is_dir()]

        for ds in datasets:
            ds_path = self._cache_dir / ds
            if not ds_path.exists():
                continue
            for symbol_dir in ds_path.iterdir():
                if not symbol_dir.is_dir():
                    continue
                for schema_dir in symbol_dir.iterdir():
                    if schema_dir.is_dir():
                        yield ds, schema_dir

    def list_keys(self, dataset: str | None = None) -> list[tuple[str, str, str]]:
        keys: list[tuple[str, str, str]] = []
        for ds, schema_dir in self._iter_symbol_schema_dirs(dataset):
            if (schema_dir / "meta.json").exists():
                keys.append((ds, schema_dir.parent.name, schema_dir.name))
        return keys

    def list_orphans(self, dataset: str | None = None) -> list[tuple[str, str, str]]:
        orphans: list[tuple[str, str, str]] = []
        for ds, schema_dir in self._iter_symbol_schema_dirs(dataset):
            has_parquet = any(schema_dir.rglob("*.parquet"))
            if has_parquet and not (schema_dir / "meta.json").exists():
                orphans.append((ds, schema_dir.parent.name, schema_dir.name))
        return orphans

    def size_bytes(self, dataset: str, symbol: str, schema: str) -> int:
        base_path = self._symbol_path(dataset, symbol, schema)
        if not base_path.exists():
            return 0
        return sum(
            f.stat().st_size for f in base_path.rglob("*.parquet") if f.is_file()
        )

    def actual_data_range(
        self, dataset: str, symbol: str, schema: str
    ) -> tuple[date, date] | None:
        base_path = self._symbol_path(dataset, symbol, schema)
        parquet_files = list(base_path.glob("**/*.parquet"))
        if not parquet_files:
            return None

        all_min: date | None = None
        all_max: date | None = None
        for pf in parquet_files:
            actual_range = parquet_date_range(pf)
            if actual_range:
                file_min, file_max = actual_range
                if all_min is None or file_min < all_min:
                    all_min = file_min
                if all_max is None or file_max > all_max:
                    all_max = file_max

        if all_min is None or all_max is None:
            return None
        return all_min, all_max

    # -- housekeeping -------------------------------------------------------
    @contextmanager
    def lock(
        self, dataset: str, symbol: str, schema: str, timeout: float = 300
    ) -> Iterator[None]:
        lock_path = self._lock_path(dataset, symbol, schema)
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        file_lock = FileLock(lock_path, timeout=timeout)
        with file_lock:
            yield

    def _cleanup_empty_dirs(self, start_path: Path, stop_at: Path) -> None:
        current = start_path
        while current >= stop_at:
            try:
                if current.is_dir() and not any(current.iterdir()):
                    current.rmdir()
                else:
                    break
            except OSError:
                break
            current = current.parent

    def cleanup(self, dataset: str, symbol: str, schema: str) -> None:
        lock_path = self._lock_path(dataset, symbol, schema)
        lock_path.unlink(missing_ok=True)
        symbol_path = self._symbol_path(dataset, symbol, schema)
        dataset_path = self._cache_dir / dataset
        self._cleanup_empty_dirs(symbol_path, dataset_path)
