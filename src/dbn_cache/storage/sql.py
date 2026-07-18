"""SQL storage backend (SQLModel / SQLAlchemy).

Stores cached market data as columnar rows in a SQL database so the cache can be
centralized and shared. SQLite (``sqlite:///path.db``) is the simple, zero-config
default; the same code path works with PostgreSQL/MySQL (``postgresql://...`` /
``mysql://...``) for a shared, remotely-accessible cache.

Layout:

- ``cache_meta`` (SQLModel): one row per (dataset, symbol, schema) holding the
  JSON-encoded :class:`~dbn_cache.models.SymbolMeta`.
- ``cache_partitions`` (SQLModel): a registry of every fetched partition. This is
  the source of truth for "fetched vs. genuinely-empty", which file existence
  provides for the filesystem backend.
- ``data_<schema>``: one table per schema holding the actual rows (columns vary by
  schema, so these are created dynamically), prefixed with helper columns
  (``_dataset``, ``_symbol_normalized``, ``_part_year``, ``_part_month``,
  ``_part_day``) used for partition-scoped reads and deletes.
"""

from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from datetime import UTC, date, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl
from filelock import FileLock

try:
    from sqlalchemy import Engine, inspect, text
    from sqlmodel import Field, Session, SQLModel, col, create_engine, select
except ImportError as exc:  # pragma: no cover - import guard
    _MSG = (
        "The SQL cache backend requires SQLModel. "
        "Install it with: pip install 'dbn-cache[sql]'"
    )
    raise ImportError(_MSG) from exc

from ..models import SymbolMeta
from ..utils import (
    detect_stype,
    get_default_cache_dir,
    is_tick_schema,
    normalize_symbol,
)
from .base import DataReader, PartitionKey, StorageBackend

if TYPE_CHECKING:
    from collections.abc import Iterator

_HELPER_COLUMNS = (
    "_dataset",
    "_symbol_normalized",
    "_part_year",
    "_part_month",
    "_part_day",
)
_TS_COLUMNS = ("ts_event", "ts_recv")


class CacheMeta(SQLModel, table=True):
    """Per symbol/schema metadata (SymbolMeta serialized as JSON)."""

    __tablename__ = "cache_meta"  # pyright: ignore[reportAssignmentType]

    dataset: str = Field(primary_key=True)
    symbol_normalized: str = Field(primary_key=True)
    schema_name: str = Field(primary_key=True)
    symbol: str = ""
    meta_json: str = ""


class CachePartition(SQLModel, table=True):
    """Registry of fetched partitions (fetched-vs-empty source of truth)."""

    __tablename__ = "cache_partitions"  # pyright: ignore[reportAssignmentType]

    dataset: str = Field(primary_key=True)
    symbol_normalized: str = Field(primary_key=True)
    schema_name: str = Field(primary_key=True)
    part_year: int = Field(primary_key=True)
    part_month: int = Field(primary_key=True)
    part_day: int = Field(primary_key=True)  # 0 for monthly partitions
    stype: str = ""
    granularity: str = ""
    start_date: str | None = None
    end_date: str | None = None
    row_count: int = 0
    fetched_at: str = ""


def _data_table_name(schema: str) -> str:
    """Table name for a schema's rows (e.g. ``ohlcv-1m`` -> ``data_ohlcv_1m``)."""
    safe = "".join(c if c.isalnum() else "_" for c in schema)
    return f"data_{safe}"


def _part_ordinal_sql() -> str:
    """SQL expression turning partition columns into a comparable ordinal."""
    return "(_part_year * 10000 + _part_month * 100 + _part_day)"


def _df_date_range(df: pl.DataFrame) -> tuple[date, date] | None:
    """Compute the (min, max) data date from a dataframe's timestamp column."""
    ts_col: str | None = None
    for candidate in ("ts_event", "ts"):
        if candidate in df.columns:
            ts_col = candidate
            break
    if ts_col is None or df.height == 0:
        return None

    dtype = df.schema[ts_col]
    is_datetime = dtype == pl.Datetime or str(dtype).startswith("Datetime")
    is_int = dtype == pl.Int64 or dtype == pl.UInt64
    if not is_datetime and not is_int:
        return None

    min_v = df[ts_col].min()
    max_v = df[ts_col].max()
    if min_v is None or max_v is None:
        return None

    if isinstance(min_v, datetime) and isinstance(max_v, datetime):
        return min_v.date(), max_v.date()
    return (
        datetime.fromtimestamp(int(min_v) / 1e9, tz=UTC).date(),  # pyright: ignore[reportArgumentType]
        datetime.fromtimestamp(int(max_v) / 1e9, tz=UTC).date(),  # pyright: ignore[reportArgumentType]
    )


class DatabaseReader:
    """A :class:`~dbn_cache.storage.base.DataReader` backed by a SQL query."""

    def __init__(
        self,
        engine: Engine,
        table: str | None,
        dataset: str,
        symbol_normalized: str,
        lo_ordinal: int | None,
        hi_ordinal: int | None,
    ) -> None:
        self._engine = engine
        self._table = table
        self._dataset = dataset
        self._symbol_normalized = symbol_normalized
        self._lo = lo_ordinal
        self._hi = hi_ordinal

    def scan(self) -> pl.LazyFrame:
        if self._table is None or self._lo is None or self._hi is None:
            return pl.LazyFrame()

        stmt = text(
            f"SELECT * FROM {self._table} "  # noqa: S608 - internal table name
            "WHERE _dataset = :d AND _symbol_normalized = :s "
            f"AND {_part_ordinal_sql()} BETWEEN :lo AND :hi"
        ).bindparams(
            d=self._dataset, s=self._symbol_normalized, lo=self._lo, hi=self._hi
        )
        with self._engine.connect() as conn:
            df = pl.read_database(stmt, connection=conn)

        drop = [c for c in _HELPER_COLUMNS if c in df.columns]
        if drop:
            df = df.drop(drop)
        # SQLite has no native datetime type: timestamps round-trip as strings.
        casts = [
            pl.col(c).str.to_datetime(strict=False)
            for c in _TS_COLUMNS
            if c in df.columns and df.schema[c] == pl.String
        ]
        if casts:
            df = df.with_columns(casts)
        return df.lazy()

    @property
    def paths(self) -> list[Path]:
        return []


class SqlBackend(StorageBackend):
    """SQLModel/SQLAlchemy-backed columnar cache (SQLite by default)."""

    def __init__(self, url: str, cache_dir: Path | None = None) -> None:
        self._url = url
        self._engine: Engine = create_engine(url)
        self._dialect = self._engine.dialect.name
        self._cache_dir = Path(cache_dir) if cache_dir else get_default_cache_dir()
        self._lock_dir = self._cache_dir / ".sqllocks"
        SQLModel.metadata.create_all(self._engine)

    @property
    def cache_dir(self) -> Path:
        return self._cache_dir

    def _table_exists(self, table: str) -> bool:
        return inspect(self._engine).has_table(table)

    # -- partition data -----------------------------------------------------
    def commit_partition(self, key: PartitionKey, src_parquet: Path) -> None:
        df = pl.read_parquet(src_parquet)
        data_range = _df_date_range(df)

        # Normalize tz-aware timestamps to naive UTC for portable SQL storage.
        tz_casts = [
            pl.col(name).dt.replace_time_zone(None)
            for name, dtype in df.schema.items()
            if isinstance(dtype, pl.Datetime) and dtype.time_zone is not None
        ]
        if tz_casts:
            df = df.with_columns(tz_casts)

        part_day = key.day if key.day is not None else 0
        df = df.with_columns(
            pl.lit(key.dataset).alias("_dataset"),
            pl.lit(key.symbol_normalized).alias("_symbol_normalized"),
            pl.lit(key.year).alias("_part_year"),
            pl.lit(key.month).alias("_part_month"),
            pl.lit(part_day).alias("_part_day"),
        )

        table = _data_table_name(key.schema)
        with self._engine.begin() as conn:
            if self._table_exists(table):
                conn.execute(
                    text(
                        f"DELETE FROM {table} "  # noqa: S608 - internal table name
                        "WHERE _dataset = :d AND _symbol_normalized = :s "
                        "AND _part_year = :y AND _part_month = :m "
                        "AND _part_day = :day"
                    ),
                    {
                        "d": key.dataset,
                        "s": key.symbol_normalized,
                        "y": key.year,
                        "m": key.month,
                        "day": part_day,
                    },
                )

        df.write_database(table, self._engine, if_table_exists="append")

        self._upsert_partition(
            key,
            part_day,
            df.height,
            data_range[0].isoformat() if data_range else None,
            data_range[1].isoformat() if data_range else None,
        )

    def _upsert_partition(
        self,
        key: PartitionKey,
        part_day: int,
        row_count: int,
        start_date: str | None,
        end_date: str | None,
    ) -> None:
        with Session(self._engine) as session:
            obj = session.get(
                CachePartition,
                (
                    key.dataset,
                    key.symbol_normalized,
                    key.schema,
                    key.year,
                    key.month,
                    part_day,
                ),
            )
            if obj is None:
                obj = CachePartition(
                    dataset=key.dataset,
                    symbol_normalized=key.symbol_normalized,
                    schema_name=key.schema,
                    part_year=key.year,
                    part_month=key.month,
                    part_day=part_day,
                )
            obj.stype = detect_stype(key.symbol)
            obj.granularity = key.granularity
            obj.start_date = start_date
            obj.end_date = end_date
            obj.row_count = row_count
            obj.fetched_at = datetime.now(UTC).isoformat()
            session.add(obj)
            session.commit()

    def partition_exists(self, key: PartitionKey) -> bool:
        part_day = key.day if key.day is not None else 0
        with Session(self._engine) as session:
            obj = session.get(
                CachePartition,
                (
                    key.dataset,
                    key.symbol_normalized,
                    key.schema,
                    key.year,
                    key.month,
                    part_day,
                ),
            )
            return obj is not None

    def delete_partition(self, key: PartitionKey) -> bool:
        part_day = key.day if key.day is not None else 0
        table = _data_table_name(key.schema)
        with self._engine.begin() as conn:
            if self._table_exists(table):
                conn.execute(
                    text(
                        f"DELETE FROM {table} "  # noqa: S608 - internal table name
                        "WHERE _dataset = :d AND _symbol_normalized = :s "
                        "AND _part_year = :y AND _part_month = :m "
                        "AND _part_day = :day"
                    ),
                    {
                        "d": key.dataset,
                        "s": key.symbol_normalized,
                        "y": key.year,
                        "m": key.month,
                        "day": part_day,
                    },
                )
        with Session(self._engine) as session:
            obj = session.get(
                CachePartition,
                (
                    key.dataset,
                    key.symbol_normalized,
                    key.schema,
                    key.year,
                    key.month,
                    part_day,
                ),
            )
            if obj is None:
                return False
            session.delete(obj)
            session.commit()
            return True

    def read_range(
        self, dataset: str, symbol: str, schema: str, start: date, end: date
    ) -> DataReader:
        symbol_norm = normalize_symbol(symbol)
        table = _data_table_name(schema)
        if not self._table_exists(table):
            return DatabaseReader(self._engine, None, dataset, symbol_norm, None, None)

        if is_tick_schema(schema):
            lo = start.year * 10000 + start.month * 100 + start.day
            hi = end.year * 10000 + end.month * 100 + end.day
        else:
            lo = start.year * 10000 + start.month * 100
            hi = end.year * 10000 + end.month * 100
        return DatabaseReader(self._engine, table, dataset, symbol_norm, lo, hi)

    # -- metadata -----------------------------------------------------------
    def load_meta(self, dataset: str, symbol: str, schema: str) -> SymbolMeta | None:
        with Session(self._engine) as session:
            obj = session.get(CacheMeta, (dataset, normalize_symbol(symbol), schema))
            if obj is None:
                return None
            payload = obj.meta_json
        return SymbolMeta.model_validate(json.loads(payload))

    def save_meta(self, meta: SymbolMeta) -> None:
        symbol_norm = normalize_symbol(meta.symbol)
        payload = json.dumps(meta.model_dump(by_alias=True), default=str)
        with Session(self._engine) as session:
            obj = session.get(CacheMeta, (meta.dataset, symbol_norm, meta.schema_))
            if obj is None:
                obj = CacheMeta(
                    dataset=meta.dataset,
                    symbol_normalized=symbol_norm,
                    schema_name=meta.schema_,
                )
            obj.symbol = meta.symbol
            obj.meta_json = payload
            session.add(obj)
            session.commit()

    def delete_meta(self, dataset: str, symbol: str, schema: str) -> None:
        with Session(self._engine) as session:
            obj = session.get(CacheMeta, (dataset, normalize_symbol(symbol), schema))
            if obj is not None:
                session.delete(obj)
                session.commit()

    # -- introspection / repair --------------------------------------------
    def list_keys(self, dataset: str | None = None) -> list[tuple[str, str, str]]:
        stmt = select(CacheMeta)
        if dataset:
            stmt = stmt.where(col(CacheMeta.dataset) == dataset)
        with Session(self._engine) as session:
            objs = session.exec(stmt).all()
        return [(o.dataset, o.symbol_normalized, o.schema_name) for o in objs]

    def list_orphans(self, dataset: str | None = None) -> list[tuple[str, str, str]]:
        with Session(self._engine) as session:
            parts = session.exec(select(CachePartition)).all()
            metas = session.exec(select(CacheMeta)).all()
        meta_keys = {(m.dataset, m.symbol_normalized, m.schema_name) for m in metas}
        part_keys = {(p.dataset, p.symbol_normalized, p.schema_name) for p in parts}
        orphans = sorted(
            k
            for k in part_keys
            if k not in meta_keys and (dataset is None or k[0] == dataset)
        )
        return list(orphans)

    def _partitions_for(
        self, dataset: str, symbol: str, schema: str
    ) -> list[CachePartition]:
        stmt = select(CachePartition).where(
            col(CachePartition.dataset) == dataset,
            col(CachePartition.symbol_normalized) == normalize_symbol(symbol),
            col(CachePartition.schema_name) == schema,
        )
        with Session(self._engine) as session:
            return list(session.exec(stmt).all())

    def size_bytes(self, dataset: str, symbol: str, schema: str) -> int:
        """Approximate size as the stored row count (SQL has no cheap byte size)."""
        return sum(p.row_count for p in self._partitions_for(dataset, symbol, schema))

    def actual_data_range(
        self, dataset: str, symbol: str, schema: str
    ) -> tuple[date, date] | None:
        parts = self._partitions_for(dataset, symbol, schema)
        starts = [p.start_date for p in parts if p.start_date]
        ends = [p.end_date for p in parts if p.end_date]
        if not starts or not ends:
            return None
        return date.fromisoformat(min(starts)), date.fromisoformat(max(ends))

    # -- housekeeping -------------------------------------------------------
    def _lock_file(self, dataset: str, symbol: str, schema: str) -> Path:
        digest = hashlib.sha1(  # noqa: S324 - non-cryptographic lock naming
            f"{dataset}/{normalize_symbol(symbol)}/{schema}".encode()
        ).hexdigest()
        return self._lock_dir / f"{digest}.lock"

    @contextmanager
    def lock(
        self, dataset: str, symbol: str, schema: str, timeout: float = 300
    ) -> Iterator[None]:
        if self._dialect in ("postgresql", "mysql", "mariadb"):
            with self._advisory_lock(dataset, symbol, schema, timeout):
                yield
        else:
            self._lock_dir.mkdir(parents=True, exist_ok=True)
            file_lock = FileLock(
                self._lock_file(dataset, symbol, schema), timeout=timeout
            )
            with file_lock:
                yield

    @contextmanager
    def _advisory_lock(
        self, dataset: str, symbol: str, schema: str, timeout: float
    ) -> Iterator[None]:
        name = f"{dataset}/{normalize_symbol(symbol)}/{schema}"
        digest = hashlib.sha1(name.encode()).digest()  # noqa: S324 - lock key only
        key = int.from_bytes(digest[:8], "big", signed=True)
        conn = self._engine.connect()
        try:
            if self._dialect == "postgresql":
                conn.execute(text("SELECT pg_advisory_lock(:k)"), {"k": key})
            else:  # mysql / mariadb
                conn.execute(
                    text("SELECT GET_LOCK(:k, :t)"),
                    {"k": digest[:8].hex(), "t": int(timeout)},
                )
            conn.commit()
            yield
        finally:
            if self._dialect == "postgresql":
                conn.execute(text("SELECT pg_advisory_unlock(:k)"), {"k": key})
            else:
                conn.execute(text("SELECT RELEASE_LOCK(:k)"), {"k": digest[:8].hex()})
            conn.commit()
            conn.close()

    def cleanup(self, dataset: str, symbol: str, schema: str) -> None:
        self._lock_file(dataset, symbol, schema).unlink(missing_ok=True)
