from dataclasses import dataclass
from datetime import UTC, date, datetime
from enum import Enum, auto
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import polars as pl
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from .storage.base import DataReader


class DownloadStatus(Enum):
    """Status of a partition download."""

    DOWNLOADING = auto()
    COMPLETED = auto()


class CacheStatus(Enum):
    """Status of cache for a requested date range."""

    EMPTY = auto()
    PARTIAL = auto()
    COMPLETE = auto()


@dataclass
class CacheCheckResult:
    """Result of checking cache status for a date range."""

    status: CacheStatus
    cached_ranges: list["DateRange"]
    missing_ranges: list["DateRange"]
    cached_partitions: int
    missing_partitions: int

    @property
    def total_partitions(self) -> int:
        """Total partitions in the requested range."""
        return self.cached_partitions + self.missing_partitions


@dataclass
class PartitionInfo:
    """Information about a single partition."""

    year: int
    month: int
    day: int | None = None

    @property
    def label(self) -> str:
        """Human-readable label for this partition."""
        if self.day is not None:
            return f"{self.year}-{self.month:02d}-{self.day:02d}"
        return f"{self.year}-{self.month:02d}"


@dataclass
class DownloadProgress:
    """Progress update yielded during download."""

    status: DownloadStatus
    partition: PartitionInfo
    current: int
    total: int
    quality_warnings: int = 0


class DateRange(BaseModel):
    """A date range (inclusive on both ends)."""

    start: date
    end: date


class ContractSpecs(BaseModel):
    """Contract specifications for futures."""

    multiplier: float | None = None
    tick_size: float | None = None
    currency: str | None = None


class DataQualityIssue(BaseModel):
    """A data quality issue for a specific date."""

    date: date
    issue_type: str
    message: str | None = None


class SymbolMeta(BaseModel):
    """Metadata for a cached symbol/schema combination."""

    dataset: str
    symbol: str
    stype: str
    schema_: str = Field(alias="schema")
    ranges: list[DateRange]
    updated_at: datetime
    cache_version: int = 1
    contract_specs: ContractSpecs | None = None
    quality_issues: list[DataQualityIssue] = []

    model_config = {"populate_by_name": True}


class CachedDataInfo(BaseModel):
    """Summary info about cached data."""

    dataset: str
    symbol: str
    schema_: str = Field(alias="schema")
    ranges: list[DateRange]
    size_bytes: int
    quality_issues: list[DataQualityIssue] = []

    model_config = {"populate_by_name": True}


@dataclass
class UpdateAllResult:
    """Result of update_all() operation."""

    updated: list[CachedDataInfo]
    up_to_date: list[CachedDataInfo]
    errors: list[tuple[CachedDataInfo, Exception]]

    @property
    def updated_count(self) -> int:
        return len(self.updated)

    @property
    def up_to_date_count(self) -> int:
        return len(self.up_to_date)

    @property
    def error_count(self) -> int:
        return len(self.errors)

    @property
    def has_errors(self) -> bool:
        return len(self.errors) > 0


class CachedData:
    """Wrapper for cached data with multi-library access.

    Accepts either a list of parquet file paths (filesystem cache) or a
    :class:`~dbn_cache.storage.base.DataReader` (e.g. a SQL-backed cache). Both
    are read lazily and trimmed to the requested inclusive date range.
    """

    def __init__(
        self,
        source: "list[Path] | DataReader",
        start: date | None = None,
        end: date | None = None,
    ) -> None:
        if isinstance(source, list):
            self._reader: DataReader | None = None
            self._paths: list[Path] = sorted(source)
        else:
            self._reader = source
            self._paths = sorted(source.paths)
        self._start = start
        self._end = end

    @property
    def paths(self) -> list[Path]:
        """Get backing parquet file paths (empty for non-file backends)."""
        return self._paths

    def _scan(self) -> pl.LazyFrame:
        """Scan the underlying source into a LazyFrame (unfiltered)."""
        if self._reader is not None:
            return self._reader.scan()
        if not self._paths:
            return pl.LazyFrame()
        return pl.scan_parquet(self._paths)

    def _apply_date_filter(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        """Apply the inclusive date range filter using the ts_event column.

        Handles both integer-nanosecond (Databento raw) and Datetime ts_event
        columns; Datetime columns are compared via their UNIX-epoch nanoseconds.
        """
        if self._start is None and self._end is None:
            return lf

        schema = lf.collect_schema()
        if "ts_event" not in schema:
            return lf

        dtype = schema["ts_event"]
        is_datetime = dtype == pl.Datetime or str(dtype).startswith("Datetime")
        ts_ns = pl.col("ts_event").dt.epoch("ns") if is_datetime else pl.col("ts_event")

        # ts_event is compared in UTC epoch nanoseconds, so build the day bounds
        # in UTC too (not the machine's local timezone).
        # end date is inclusive, so we need to include the entire day
        if self._start is not None:
            start_ns = int(
                datetime.combine(
                    self._start, datetime.min.time(), tzinfo=UTC
                ).timestamp()
                * 1e9
            )
            lf = lf.filter(ts_ns >= start_ns)

        if self._end is not None:
            # End of day (23:59:59.999999999) for inclusive end date
            end_dt = datetime.combine(self._end, datetime.min.time(), tzinfo=UTC)
            end_ns = int((end_dt.timestamp() + 86400) * 1e9) - 1
            lf = lf.filter(ts_ns <= end_ns)

        return lf

    def to_polars(self) -> pl.LazyFrame:
        """Load data as Polars LazyFrame, filtered to requested date range."""
        return self._apply_date_filter(self._scan())

    def to_pandas(self) -> pd.DataFrame:
        """Load data as Pandas DataFrame, filtered to requested date range."""
        return self.to_polars().collect().to_pandas()

    def __repr__(self) -> str:
        return f"CachedData(paths={len(self._paths)} files)"
