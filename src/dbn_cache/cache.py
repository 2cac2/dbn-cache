import logging
import os
import tempfile
import warnings
from contextlib import AbstractContextManager
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl

from .calendar import iter_trading_days
from .client import DatabentoClient
from .exceptions import CacheMissError, DownloadCancelledError, EmptyDataError
from .futures import (
    get_contract_dates,
    get_expiration_date,
    is_supported_contract,
    parse_contract_symbol,
    to_databento_symbol,
)
from .models import (
    CacheCheckResult,
    CachedData,
    CachedDataInfo,
    CacheStatus,
    DataQualityIssue,
    DateRange,
    DownloadProgress,
    DownloadStatus,
    PartitionInfo,
    SymbolMeta,
    UpdateAllResult,
)
from .storage.base import PartitionKey, StorageBackend
from .storage.factory import create_backend
from .storage.filesystem import FilesystemBackend
from .utils import (
    detect_stype,
    find_missing_date_ranges,
    get_default_cache_dir,
    has_lookahead_bias,
    is_tick_schema,
    iter_days,
    iter_months,
    merge_date_ranges,
    month_start_end,
    utc_today,
)

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)


def _parse_quality_warnings(
    caught_warnings: list[warnings.WarningMessage],
) -> list[DataQualityIssue]:
    """Parse databento warnings into DataQualityIssue objects."""
    issues: list[DataQualityIssue] = []
    seen_dates: set[date] = set()

    for w in caught_warnings:
        msg = str(w.message)
        if "reduced quality:" in msg:
            parts = msg.split("reduced quality:")[1]
            date_section = parts.split(".")[0].strip()
            for entry in date_section.split(","):
                entry = entry.strip()
                if not entry:
                    continue
                date_str = entry.split(" ")[0]
                issue_type = "degraded"
                if "(" in entry and ")" in entry:
                    issue_type = entry.split("(")[1].split(")")[0]
                try:
                    d = date.fromisoformat(date_str)
                    if d not in seen_dates:
                        issues.append(
                            DataQualityIssue(date=d, issue_type=issue_type, message=msg)
                        )
                        seen_dates.add(d)
                except ValueError:
                    pass

    return sorted(issues, key=lambda i: i.date)


class DataCache:
    """Cache for Databento historical market data."""

    def __init__(
        self,
        cache_dir: Path | None = None,
        client: DatabentoClient | None = None,
        *,
        storage: StorageBackend | None = None,
        url: str | None = None,
    ) -> None:
        """Initialize cache.

        Args:
            cache_dir: Cache directory for the filesystem backend. Defaults to
                ~/.databento or DATABENTO_CACHE_DIR. Also used for sidecar lock
                files when a SQL backend is selected.
            client: DatabentoClient instance. Created on demand if not provided.
            storage: An explicit StorageBackend. Takes precedence over url.
            url: A SQLAlchemy connection URL (e.g. 'sqlite:///cache.db',
                'postgresql://user:pw@host/db') selecting a SQL backend. If not
                given, the DBN_CACHE_URL / DATABENTO_CACHE_URL env vars are used.
        """
        if storage is not None:
            self._backend: StorageBackend = storage
        else:
            resolved_url = (
                url
                or os.environ.get("DBN_CACHE_URL")
                or os.environ.get("DATABENTO_CACHE_URL")
            )
            if resolved_url:
                self._backend = create_backend(resolved_url, cache_dir)
            else:
                if cache_dir is None:
                    env_dir = os.environ.get("DATABENTO_CACHE_DIR")
                    cache_dir = Path(env_dir) if env_dir else get_default_cache_dir()
                self._backend = FilesystemBackend(cache_dir)

        self._cache_dir = self._backend.cache_dir
        self._client = client
        self._available_end_cache: dict[str, date] = {}

    @property
    def cache_dir(self) -> Path:
        """Get cache directory (filesystem location associated with the backend)."""
        return self._cache_dir

    @property
    def backend(self) -> StorageBackend:
        """Get the storage backend."""
        return self._backend

    def _get_client(self) -> DatabentoClient:
        """Get or create client."""
        if self._client is None:
            self._client = DatabentoClient()
        return self._client

    def _get_available_end(self, dataset: str) -> date | None:
        """Get the last date with available data for a dataset.

        Returns the inclusive end date, or None if unavailable.
        Results are cached per dataset for the lifetime of this instance.
        """
        if dataset in self._available_end_cache:
            return self._available_end_cache[dataset]
        try:
            client = self._get_client()
            result = client.get_dataset_range(dataset)
            exclusive_end = date.fromisoformat(result["end"][:10])
            inclusive_end = exclusive_end - timedelta(days=1)
            self._available_end_cache[dataset] = inclusive_end
            return inclusive_end
        except Exception:
            logger.warning(
                "Failed to get available data range for %s",
                dataset,
                exc_info=True,
            )
            return None

    def available_end(self, dataset: str) -> date | None:
        """Return the last inclusive date with data available for a dataset."""
        return self._get_available_end(dataset)

    def _lock(
        self, dataset: str, symbol: str, schema: str, timeout: float = 300
    ) -> AbstractContextManager[None]:
        """Acquire the backend lock for a symbol/schema."""
        return self._backend.lock(dataset, symbol, schema, timeout)

    def _validate_and_fix_meta(self, meta: SymbolMeta) -> SymbolMeta | None:
        """Validate metadata against actual stored data and fix if needed.

        Returns:
            Updated SymbolMeta if fixes were needed, None if metadata is correct.
        """
        actual = self._backend.actual_data_range(
            meta.dataset, meta.symbol, meta.schema_
        )
        if actual is None:
            return None
        all_min, all_max = actual

        # Check if metadata matches actual data
        if not meta.ranges:
            # Metadata has no ranges but data exists — needs fix
            return SymbolMeta(
                dataset=meta.dataset,
                symbol=meta.symbol,
                stype=meta.stype,
                schema=meta.schema_,
                ranges=[DateRange(start=all_min, end=all_max)],
                updated_at=datetime.now(UTC),
                cache_version=meta.cache_version,
                contract_specs=meta.contract_specs,
                quality_issues=meta.quality_issues,
            )

        meta_start = meta.ranges[0].start
        meta_end = meta.ranges[-1].end

        # Fix if dates don't match OR if ranges are fragmented (more than 1 range)
        if meta_start == all_min and meta_end == all_max and len(meta.ranges) == 1:
            return None  # Metadata is correct

        # Consolidate to single range with actual dates
        return SymbolMeta(
            dataset=meta.dataset,
            symbol=meta.symbol,
            stype=meta.stype,
            schema=meta.schema_,
            ranges=[DateRange(start=all_min, end=all_max)],
            updated_at=datetime.now(UTC),
            cache_version=meta.cache_version,
            contract_specs=meta.contract_specs,
            quality_issues=meta.quality_issues,
        )

    def _rebuild_meta_from_files(
        self, dataset: str, symbol: str, schema: str
    ) -> SymbolMeta | None:
        """Rebuild metadata from existing stored data.

        Used when metadata is missing but partition data exists.

        Returns:
            New SymbolMeta if data found, None if no data.
        """
        actual = self._backend.actual_data_range(dataset, symbol, schema)
        if actual is None:
            return None
        all_min, all_max = actual

        # Denormalize symbol (ES_c_0 -> ES.c.0)
        original_symbol = symbol.replace("_", ".")

        return SymbolMeta(
            dataset=dataset,
            symbol=original_symbol,
            stype=detect_stype(original_symbol),
            schema=schema,
            ranges=[DateRange(start=all_min, end=all_max)],
            updated_at=datetime.now(UTC),
        )

    def _merge_ranges(self, ranges: list[DateRange]) -> list[DateRange]:
        """Merge overlapping or adjacent date ranges."""
        tuples = [(r.start, r.end) for r in ranges]
        merged = merge_date_ranges(tuples)
        return [DateRange(start=s, end=e) for s, e in merged]

    def _find_missing_ranges(
        self, start: date, end: date, cached_ranges: list[DateRange]
    ) -> list[DateRange]:
        """Find date ranges not covered by cached_ranges."""
        tuples = [(r.start, r.end) for r in cached_ranges]
        missing = find_missing_date_ranges(start, end, tuples)
        return [DateRange(start=s, end=e) for s, e in missing]

    def _download_partition(
        self,
        symbol: str,
        schema: str,
        start: date,
        end: date,
        dataset: str,
        dest_path: Path,
        stype: str | None = None,
    ) -> None:
        """Download data for a partition and save to dest_path (a temp file)."""
        client = self._get_client()
        # Databento API end date is exclusive, so add 1 day
        api_end = end + timedelta(days=1)
        data = client.fetch(
            symbol=symbol,
            schema=schema,
            start=start,
            end=api_end,
            dataset=dataset,
            stype=stype,
        )
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        df = pl.from_pandas(data.to_df().reset_index())
        df.write_parquet(dest_path)

    def _count_partitions_in_range(
        self,
        schema: str,
        start: date,
        end: date,
        dataset: str,
    ) -> int:
        """Count total partitions in a date range.

        For tick schemas, counts only trading days (skips holidays/weekends).
        For OHLCV schemas, counts months.

        Args:
            schema: Data schema
            start: Start date (inclusive)
            end: End date (inclusive)
            dataset: Databento dataset (required for calendar lookup)
        """
        count = 0
        if is_tick_schema(schema):
            for _ in iter_trading_days(start, end, dataset):
                count += 1
        else:
            for _ in iter_months(start, end):
                count += 1
        return count

    def _get_missing_partitions(
        self,
        dataset: str,
        symbol: str,
        schema: str,
        start: date,
        end: date,
    ) -> list[DateRange]:
        """Find partitions that are missing from storage.

        For tick schemas, only considers trading days (skips holidays/weekends).
        Returns date ranges for partitions that have not been stored.
        """
        missing: list[DateRange] = []

        if is_tick_schema(schema):
            for d in iter_trading_days(start, end, dataset):
                key = PartitionKey(dataset, symbol, schema, d.year, d.month, d.day)
                if not self._backend.partition_exists(key):
                    missing.append(DateRange(start=d, end=d))
        else:
            for year, month in iter_months(start, end):
                key = PartitionKey(dataset, symbol, schema, year, month)
                if not self._backend.partition_exists(key):
                    m_start, m_end = month_start_end(year, month)
                    clamped_start = max(m_start, start)
                    clamped_end = min(m_end, end)
                    missing.append(DateRange(start=clamped_start, end=clamped_end))

        return self._merge_ranges(missing) if missing else []

    def check_cache(
        self,
        symbol: str,
        schema: str,
        start: date,
        end: date,
        dataset: str = "GLBX.MDP3",
        verify_files: bool = True,
    ) -> CacheCheckResult:
        """Check cache status for a date range.

        Args:
            symbol: Symbol to check
            schema: Data schema
            start: Start date (inclusive)
            end: End date (inclusive)
            dataset: Databento dataset
            verify_files: If True, verify actual partitions exist (not just metadata)

        Returns:
            CacheCheckResult with status and details about cached/missing data.
        """
        meta = self._backend.load_meta(dataset, symbol, schema)
        cached_ranges = list(meta.ranges) if meta else []
        missing_from_meta = self._find_missing_ranges(start, end, cached_ranges)

        if verify_files:
            missing_files = self._get_missing_partitions(
                dataset, symbol, schema, start, end
            )
            all_missing = missing_from_meta + missing_files
            missing_ranges = self._merge_ranges(all_missing) if all_missing else []
        else:
            missing_ranges = missing_from_meta

        total_partitions = self._count_partitions_in_range(schema, start, end, dataset)

        if not missing_ranges:
            return CacheCheckResult(
                status=CacheStatus.COMPLETE,
                cached_ranges=cached_ranges,
                missing_ranges=[],
                cached_partitions=total_partitions,
                missing_partitions=0,
            )

        missing_partitions = sum(
            self._count_partitions_in_range(schema, r.start, r.end, dataset)
            for r in missing_ranges
        )
        cached_partitions = total_partitions - missing_partitions
        status = CacheStatus.EMPTY if cached_partitions == 0 else CacheStatus.PARTIAL

        return CacheCheckResult(
            status=status,
            cached_ranges=cached_ranges,
            missing_ranges=missing_ranges,
            cached_partitions=cached_partitions,
            missing_partitions=missing_partitions,
        )

    def clear_cache(
        self,
        symbol: str,
        schema: str,
        start: date,
        end: date,
        dataset: str = "GLBX.MDP3",
    ) -> int:
        """Clear cached data for a date range.

        Args:
            symbol: Symbol to clear
            schema: Data schema
            start: Start date (inclusive)
            end: End date (inclusive)
            dataset: Databento dataset

        Returns:
            Number of partitions deleted.
        """
        deleted = 0

        with self._lock(dataset, symbol, schema):
            if is_tick_schema(schema):
                for d in iter_days(start, end):
                    key = PartitionKey(dataset, symbol, schema, d.year, d.month, d.day)
                    if self._backend.delete_partition(key):
                        deleted += 1
            else:
                for year, month in iter_months(start, end):
                    key = PartitionKey(dataset, symbol, schema, year, month)
                    if self._backend.delete_partition(key):
                        deleted += 1

            meta = self._backend.load_meta(dataset, symbol, schema)
            if meta:
                remaining_ranges: list[DateRange] = []
                for r in meta.ranges:
                    if r.end < start or r.start > end:
                        remaining_ranges.append(r)
                    elif r.start < start and r.end > end:
                        remaining_ranges.append(
                            DateRange(start=r.start, end=start - timedelta(days=1))
                        )
                        remaining_ranges.append(
                            DateRange(start=end + timedelta(days=1), end=r.end)
                        )
                    elif r.start < start:
                        remaining_ranges.append(
                            DateRange(start=r.start, end=start - timedelta(days=1))
                        )
                    elif r.end > end:
                        remaining_ranges.append(
                            DateRange(start=end + timedelta(days=1), end=r.end)
                        )

                if remaining_ranges:
                    meta.ranges = self._merge_ranges(remaining_ranges)
                    self._backend.save_meta(meta)
                else:
                    self._backend.delete_meta(dataset, symbol, schema)

        return deleted

    def _count_partitions_to_download(
        self,
        schema: str,
        missing: list[DateRange],
        dataset: str,
        symbol: str,
        request_start: date,
        request_end: date,
        cached_ranges: list[DateRange],
    ) -> tuple[int, list[tuple[PartitionInfo, date, date]]]:
        """Count partitions that need downloading and build download list.

        For tick schemas, only includes trading days (skips holidays/weekends).

        Returns:
            Tuple of (total count, list of (partition info, dl_start, dl_end)).
        """
        partitions: list[tuple[PartitionInfo, date, date]] = []

        for gap in missing:
            if is_tick_schema(schema):
                for d in iter_trading_days(gap.start, gap.end, dataset):
                    key = PartitionKey(dataset, symbol, schema, d.year, d.month, d.day)
                    if not self._backend.partition_exists(key):
                        info = PartitionInfo(year=d.year, month=d.month, day=d.day)
                        partitions.append((info, d, d))
            else:
                # Track seen partitions to avoid duplicates when gaps span months
                seen: set[tuple[int, int]] = set()
                for year, month in iter_months(gap.start, gap.end):
                    if (year, month) in seen:
                        continue
                    seen.add((year, month))

                    m_start, m_end = month_start_end(year, month)
                    # Start with request range clamped to month
                    dl_start = max(m_start, request_start)
                    dl_end = min(m_end, request_end)

                    # For OHLCV (monthly partitions), expand to include any existing
                    # cached data for this month to avoid overwriting when updating
                    for cached in cached_ranges:
                        if cached.end >= m_start and cached.start <= m_end:
                            dl_start = min(dl_start, max(m_start, cached.start))
                            dl_end = max(dl_end, min(m_end, cached.end))

                    info = PartitionInfo(year=year, month=month)
                    # Always include - if there's a gap in this month, we need to
                    # re-download the partition even if it exists
                    partitions.append((info, dl_start, dl_end))

        return len(partitions), partitions

    def _save_incremental_meta(
        self,
        dataset: str,
        symbol: str,
        schema: str,
        cached_ranges: list[DateRange],
    ) -> None:
        """Save metadata with current progress."""
        new_meta = SymbolMeta(
            dataset=dataset,
            symbol=symbol,
            stype=detect_stype(symbol),
            schema=schema,
            ranges=self._merge_ranges(cached_ranges),
            updated_at=datetime.now(UTC),
        )
        self._backend.save_meta(new_meta)

    def download(
        self,
        symbol: str,
        schema: str,
        start: date | None = None,
        end: date | None = None,
        dataset: str = "GLBX.MDP3",
        on_progress: "Callable[[DownloadProgress], None] | None" = None,
        cancelled: "Callable[[], bool] | None" = None,
        rollover_days: int = 14,
        stype: str | None = None,
    ) -> CachedData:
        """Download data and cache it.

        Args:
            symbol: Symbol to download (e.g., 'ES.c.0', 'ESZ24')
            schema: Data schema (e.g., 'ohlcv-1m', 'trades')
            start: Start date (inclusive). If None, auto-detected for supported
                futures contracts.
            end: End date (inclusive). If None, auto-detected for supported
                futures contracts.
            dataset: Databento dataset
            on_progress: Optional callback for progress updates
            cancelled: Optional callable that returns True if download should stop
            rollover_days: Days before front-month to start (for auto-detected dates).
                Default 14.
            stype: Symbol type (stype_in) to pass to the API. Auto-detected from the
                symbol format if not provided.

        Returns:
            CachedData wrapper for the downloaded data.

        Raises:
            DownloadCancelledError: If download was cancelled via the cancelled callback
            ValueError: If dates are not provided and symbol is not a supported
                futures contract for auto-detection.

        Example:
            >>> cache = DataCache()
            >>> # Auto-detect dates for supported futures contracts
            >>> data = cache.download("NQH25", "ohlcv-1m")
            >>> # Custom rollover buffer
            >>> data = cache.download("NQH25", "ohlcv-1m", rollover_days=7)
            >>> # Explicit dates (always works)
            >>> data = cache.download("NQH25", "ohlcv-1m",
            ...                        start=date(2024, 12, 1),
            ...                        end=date(2025, 3, 21))
        """
        # Handle date auto-detection for supported futures contracts
        if start is None or end is None:
            if start is not None or end is not None:
                msg = (
                    "Both start and end must be provided together, "
                    "or omit both for auto-detection."
                )
                raise ValueError(msg)

            if is_supported_contract(symbol):
                start, end = get_contract_dates(symbol, rollover_days=rollover_days)
                # Cap end date at yesterday (Databento has 24h embargo)
                yesterday = utc_today() - timedelta(days=1)
                if end > yesterday:
                    end = yesterday
                # If contract hasn't started yet, raise error
                if start > yesterday:
                    msg = (
                        f"Contract {symbol} data is not yet available. "
                        f"Start date ({start}) is in the future."
                    )
                    raise ValueError(msg)
            else:
                msg = (
                    f"start and end are required for {symbol}. "
                    "Auto-detection only works for supported futures contracts "
                    "(e.g., NQH25, ESZ24)."
                )
                raise ValueError(msg)

        # Determine API symbol (Databento uses single-digit years for futures)
        api_symbol = (
            to_databento_symbol(symbol) if is_supported_contract(symbol) else symbol
        )
        if has_lookahead_bias(symbol):
            logger.warning(
                "Symbol %s uses volume/OI-based rolls which have look-ahead bias. "
                "Use calendar rolls (.c.) for backtesting.",
                symbol,
            )

        with self._lock(dataset, symbol, schema):
            meta = self._backend.load_meta(dataset, symbol, schema)
            cached_ranges = list(meta.ranges) if meta else []
            missing_from_meta = self._find_missing_ranges(start, end, cached_ranges)

            missing_files = self._get_missing_partitions(
                dataset, symbol, schema, start, end
            )
            all_missing = missing_from_meta + missing_files
            missing = self._merge_ranges(all_missing) if all_missing else []

            if not missing:
                reader = self._backend.read_range(dataset, symbol, schema, start, end)
                return CachedData(reader, start=start, end=end)

            total, partitions = self._count_partitions_to_download(
                schema, missing, dataset, symbol, start, end, cached_ranges
            )

            if total == 0:
                reader = self._backend.read_range(dataset, symbol, schema, start, end)
                return CachedData(reader, start=start, end=end)

            completed_ranges: list[DateRange] = list(cached_ranges)

            with warnings.catch_warnings(record=True) as caught_warnings:
                warnings.simplefilter("always")
                warnings.filterwarnings("ignore", category=ResourceWarning)

                def count_quality_warnings() -> int:
                    return sum(
                        1
                        for w in caught_warnings
                        if "reduced quality:" in str(w.message)
                    )

                try:
                    for current, (partition_info, dl_start, dl_end) in enumerate(
                        partitions, start=1
                    ):
                        if on_progress:
                            on_progress(
                                DownloadProgress(
                                    status=DownloadStatus.DOWNLOADING,
                                    partition=partition_info,
                                    current=current,
                                    total=total,
                                    quality_warnings=count_quality_warnings(),
                                )
                            )

                        with tempfile.NamedTemporaryFile(
                            delete=False, suffix=".parquet"
                        ) as tmp:
                            tmp_path = Path(tmp.name)
                        try:
                            if stype:
                                self._download_partition(
                                    api_symbol,
                                    schema,
                                    dl_start,
                                    dl_end,
                                    dataset,
                                    tmp_path,
                                    stype=stype,
                                )
                            else:
                                self._download_partition(
                                    api_symbol,
                                    schema,
                                    dl_start,
                                    dl_end,
                                    dataset,
                                    tmp_path,
                                )
                            # Check the downloaded data has rows before committing
                            row_count = (
                                pl.scan_parquet(tmp_path)
                                .select(pl.len())
                                .collect()
                                .item()
                            )
                            if row_count == 0:
                                # No data for this partition, skip it
                                tmp_path.unlink(missing_ok=True)
                                continue
                            key = PartitionKey(
                                dataset,
                                symbol,
                                schema,
                                partition_info.year,
                                partition_info.month,
                                partition_info.day,
                            )
                            self._backend.commit_partition(key, tmp_path)
                            tmp_path.unlink(missing_ok=True)
                        except BaseException:
                            tmp_path.unlink(missing_ok=True)
                            raise

                        # Use partition dates (calendar-based) for metadata
                        # This ensures contiguous months merge correctly
                        # (e.g., June 30 + 1 = July 1)
                        completed_ranges.append(DateRange(start=dl_start, end=dl_end))
                        self._save_incremental_meta(
                            dataset, symbol, schema, completed_ranges
                        )

                        if on_progress:
                            on_progress(
                                DownloadProgress(
                                    status=DownloadStatus.COMPLETED,
                                    partition=partition_info,
                                    current=current,
                                    total=total,
                                    quality_warnings=count_quality_warnings(),
                                )
                            )

                        if cancelled and cancelled():
                            raise DownloadCancelledError(current, total)

                finally:
                    issues = _parse_quality_warnings(list(caught_warnings))
                    if issues:
                        self._add_quality_issues_unlocked(
                            symbol, schema, issues, dataset
                        )

        reader = self._backend.read_range(dataset, symbol, schema, start, end)

        # Check downloaded data has actual rows (not just empty parquet schema)
        total_rows = reader.scan().select(pl.len()).collect().item()
        if total_rows == 0:
            # Clean up lock file and empty artifacts
            self._backend.cleanup(dataset, symbol, schema)
            raise EmptyDataError(symbol, dataset)

        return CachedData(reader, start=start, end=end)

    def get(
        self,
        symbol: str,
        schema: str,
        start: date,
        end: date,
        dataset: str = "GLBX.MDP3",
    ) -> CachedData:
        """Get data from cache.

        Args:
            symbol: Symbol to get
            schema: Data schema
            start: Start date
            end: End date
            dataset: Databento dataset

        Returns:
            CachedData wrapper.

        Raises:
            CacheMissError: If data is not fully cached or partitions are missing.
        """
        check = self.check_cache(symbol, schema, start, end, dataset, verify_files=True)
        if check.status != CacheStatus.COMPLETE:
            if check.status == CacheStatus.EMPTY:
                msg = f"No cached data for {symbol}/{schema}"
            else:
                msg = f"Missing data for {symbol}/{schema}: {check.missing_ranges}"
            raise CacheMissError(msg)

        reader = self._backend.read_range(dataset, symbol, schema, start, end)
        return CachedData(reader, start=start, end=end)

    def ensure(
        self,
        symbol: str,
        schema: str,
        start: date,
        end: date,
        dataset: str = "GLBX.MDP3",
        stype: str | None = None,
    ) -> CachedData:
        """Ensure data is cached, downloading if needed.

        Args:
            symbol: Symbol to get
            schema: Data schema
            start: Start date
            end: End date
            dataset: Databento dataset
            stype: Symbol type (stype_in) to pass to the API on download.

        Returns:
            CachedData wrapper.
        """
        try:
            return self.get(symbol, schema, start, end, dataset)
        except CacheMissError:
            return self.download(symbol, schema, start, end, dataset, stype=stype)

    def get_update_range(
        self,
        cached_info: CachedDataInfo,
        end: date | None = None,
    ) -> tuple[date, date] | None:
        """Get the date range needed to update cached data.

        The end date is clamped to the dataset's available range on Databento
        (data has ~24h embargo). For specific futures contracts (e.g., NQH25),
        it is further capped at the contract's expiration date.

        Args:
            cached_info: Cached data info from list_cached()
            end: End date (defaults to yesterday UTC, clamped to available data)

        Returns:
            Tuple of (start, end) dates if update is needed, None if up to date
            or contract has expired.
        """
        if not cached_info.ranges:
            return None

        last_cached = cached_info.ranges[-1].end
        start = last_cached + timedelta(days=1)
        end_date = end or (utc_today() - timedelta(days=1))

        # Clamp to data actually available on Databento
        available_end = self._get_available_end(cached_info.dataset)
        if available_end is not None:
            end_date = min(end_date, available_end)

        # For specific futures contracts, cap at expiration date
        if is_supported_contract(cached_info.symbol):
            try:
                root, month, year = parse_contract_symbol(cached_info.symbol)
                expiration = get_expiration_date(root, month, year)
                if last_cached >= expiration:
                    # Contract already expired and fully cached
                    return None
                end_date = min(end_date, expiration)
            except ValueError:
                pass  # Not a valid futures symbol, use default behavior

        if start > end_date:
            return None

        return (start, end_date)

    def update(
        self,
        symbol: str,
        schema: str,
        end: date | None = None,
        on_progress: "Callable[[DownloadProgress], None] | None" = None,
        cancelled: "Callable[[], bool] | None" = None,
    ) -> CachedData | None:
        """Update cached data from last cached date to yesterday (UTC).

        Historical data has a 24-hour embargo, so yesterday UTC is the default.

        Args:
            symbol: Symbol to update
            schema: Data schema
            end: End date (defaults to yesterday)
            on_progress: Progress callback
            cancelled: Cancellation check callback

        Returns:
            CachedData if new data was downloaded, None if already up to date.

        Raises:
            CacheMissError: If no existing cached data exists.
        """
        cached_info = self.info(symbol, schema)

        if cached_info is None or not cached_info.ranges:
            raise CacheMissError(
                f"No cached data for {symbol}/{schema}. Use download() first."
            )

        update_range = self.get_update_range(cached_info, end)
        if update_range is None:
            return None

        start, end_date = update_range
        return self.download(
            symbol,
            schema,
            start,
            end_date,
            cached_info.dataset,
            on_progress=on_progress,
            cancelled=cancelled,
        )

    def update_all(
        self,
        end: date | None = None,
        on_progress: "Callable[[DownloadProgress], None] | None" = None,
        cancelled: "Callable[[], bool] | None" = None,
    ) -> "UpdateAllResult":
        """Update all cached data from last cached date to yesterday (UTC).

        Historical data has a 24-hour embargo, so yesterday UTC is the default.

        Args:
            end: End date (defaults to yesterday)
            on_progress: Progress callback (called for each item)
            cancelled: Cancellation check callback

        Returns:
            UpdateAllResult with counts and any errors.
        """
        all_cached = self.list_cached()
        end_date = end or (utc_today() - timedelta(days=1))

        updated: list[CachedDataInfo] = []
        up_to_date: list[CachedDataInfo] = []
        errors: list[tuple[CachedDataInfo, Exception]] = []

        for item in all_cached:
            update_range = self.get_update_range(item, end_date)
            if update_range is None:
                up_to_date.append(item)
                continue

            start, item_end = update_range
            try:
                self.download(
                    item.symbol,
                    item.schema_,
                    start,
                    item_end,
                    item.dataset,
                    on_progress=on_progress,
                    cancelled=cancelled,
                )
                updated.append(item)
            except Exception as e:  # noqa: BLE001
                errors.append((item, e))

        return UpdateAllResult(
            updated=updated,
            up_to_date=up_to_date,
            errors=errors,
        )

    def list_cached(self, dataset: str | None = None) -> list[CachedDataInfo]:
        """List all cached data.

        Args:
            dataset: Filter by dataset. If None, list all.

        Returns:
            List of CachedDataInfo.
        """
        results: list[CachedDataInfo] = []

        for ds, symbol_key, schema in self._backend.list_keys(dataset):
            meta = self._backend.load_meta(ds, symbol_key, schema)
            if meta:
                size = self._backend.size_bytes(ds, symbol_key, schema)
                results.append(
                    CachedDataInfo(
                        dataset=ds,
                        symbol=meta.symbol,
                        schema=meta.schema_,
                        ranges=meta.ranges,
                        size_bytes=size,
                        quality_issues=meta.quality_issues,
                    )
                )

        return results

    def info(
        self, symbol: str, schema: str, dataset: str = "GLBX.MDP3"
    ) -> CachedDataInfo | None:
        """Get info about cached data for a symbol/schema.

        Returns:
            CachedDataInfo or None if not cached.
        """
        meta = self._backend.load_meta(dataset, symbol, schema)
        if meta is None:
            return None

        size = self._backend.size_bytes(dataset, symbol, schema)

        return CachedDataInfo(
            dataset=dataset,
            symbol=symbol,
            schema=schema,
            ranges=meta.ranges,
            size_bytes=size,
            quality_issues=meta.quality_issues,
        )

    def validate_metadata(
        self, dataset: str | None = None, *, fix: bool = False
    ) -> list[tuple[str, str, str, str]]:
        """Validate metadata date ranges against actual stored data.

        Checks that metadata start/end dates match the actual data in storage.
        Mismatches can occur when partition boundaries (month-end dates) are
        stored in metadata but the actual data ends earlier (e.g., expired
        futures contracts).

        Args:
            dataset: Filter by dataset. If None, validate all.
            fix: If True, fix mismatched metadata.

        Returns:
            List of (dataset, symbol, schema, message) tuples for each mismatch.
        """
        results: list[tuple[str, str, str, str]] = []

        for ds, symbol_key, schema in self._backend.list_keys(dataset):
            meta = self._backend.load_meta(ds, symbol_key, schema)
            if meta is None:
                continue
            fixed = self._validate_and_fix_meta(meta)
            if fixed is not None:
                old_end = meta.ranges[-1].end if meta.ranges else None
                new_end = fixed.ranges[-1].end if fixed.ranges else None
                msg = f"{old_end} -> {new_end}"
                results.append((ds, meta.symbol, meta.schema_, msg))
                if fix:
                    self._backend.save_meta(fixed)
                    logger.info(
                        "Fixed metadata for %s/%s: %s",
                        meta.symbol,
                        meta.schema_,
                        msg,
                    )

        return results

    def repair_metadata(self, dataset: str | None = None) -> list[tuple[str, str, str]]:
        """Find and rebuild metadata for orphaned partition data.

        Scans for stored data that has no metadata and rebuilds it from the data.

        Args:
            dataset: Filter by dataset. If None, scan all.

        Returns:
            List of (dataset, symbol, schema) tuples that were repaired.
        """
        repaired: list[tuple[str, str, str]] = []

        for ds, symbol_key, schema in self._backend.list_orphans(dataset):
            rebuilt = self._rebuild_meta_from_files(ds, symbol_key, schema)
            if rebuilt:
                self._backend.save_meta(rebuilt)
                repaired.append((ds, rebuilt.symbol, schema))
                logger.info(
                    "Rebuilt metadata for %s/%s/%s",
                    ds,
                    rebuilt.symbol,
                    schema,
                )

        return repaired

    def _add_quality_issues_unlocked(
        self,
        symbol: str,
        schema: str,
        issues: list[DataQualityIssue],
        dataset: str,
    ) -> None:
        """Add quality issues without acquiring lock (caller must hold lock)."""
        if not issues:
            return

        meta = self._backend.load_meta(dataset, symbol, schema)
        if meta is None:
            return

        existing_dates = {i.date for i in meta.quality_issues}
        for issue in issues:
            if issue.date not in existing_dates:
                meta.quality_issues.append(issue)
                existing_dates.add(issue.date)

        meta.quality_issues.sort(key=lambda i: i.date)
        self._backend.save_meta(meta)

    def get_quality_issues(
        self,
        symbol: str,
        schema: str,
        dataset: str = "GLBX.MDP3",
        start: date | None = None,
        end: date | None = None,
    ) -> list[DataQualityIssue]:
        """Get data quality issues for a symbol.

        Args:
            symbol: Symbol
            schema: Data schema
            dataset: Databento dataset
            start: Optional start date filter
            end: Optional end date filter

        Returns:
            List of quality issues, optionally filtered by date range.
        """
        meta = self._backend.load_meta(dataset, symbol, schema)
        if meta is None:
            return []

        issues = meta.quality_issues
        if start is not None:
            issues = [i for i in issues if i.date >= start]
        if end is not None:
            issues = [i for i in issues if i.date <= end]

        return issues
