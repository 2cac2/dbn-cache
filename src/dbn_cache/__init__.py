"""Databento data cache utility.

Uses lazy imports to avoid loading heavy dependencies (polars, pandas,
exchange_calendars, databento) until they're actually needed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .cache import DataCache
    from .client import DatabentoClient
    from .exceptions import (
        CacheMissError,
        DownloadCancelledError,
        EmptyDataError,
        MissingAPIKeyError,
        PartialCacheError,
    )
    from .futures import (
        generate_quarterly_contracts,
        get_contract_dates,
        get_expiration_date,
        get_front_month_contract,
        get_next_contract,
        is_supported_contract,
        is_supported_root,
        parse_contract_symbol,
        to_databento_symbol,
    )
    from .historical import CacheStore, Historical
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
        UpdateAllResult,
    )
    from .storage import (
        DataReader,
        FilesystemBackend,
        PartitionKey,
        StorageBackend,
        create_backend,
    )


__all__ = [
    "CacheCheckResult",
    "CacheMissError",
    "CacheStatus",
    "CacheStore",
    "CachedData",
    "CachedDataInfo",
    "DataCache",
    "DataQualityIssue",
    "DataReader",
    "DatabentoClient",
    "DateRange",
    "DownloadCancelledError",
    "DownloadProgress",
    "DownloadStatus",
    "EmptyDataError",
    "FilesystemBackend",
    "Historical",
    "MissingAPIKeyError",
    "PartialCacheError",
    "PartitionInfo",
    "PartitionKey",
    "StorageBackend",
    "UpdateAllResult",
    "create_backend",
    "generate_quarterly_contracts",
    "get_contract_dates",
    "get_expiration_date",
    "get_front_month_contract",
    "get_next_contract",
    "is_supported_contract",
    "is_supported_root",
    "parse_contract_symbol",
    "to_databento_symbol",
]


def __getattr__(name: str):
    """Lazy import public API members."""
    if name == "DataCache":
        from .cache import DataCache

        return DataCache
    if name == "DatabentoClient":
        from .client import DatabentoClient

        return DatabentoClient
    if name in (
        "CacheMissError",
        "DownloadCancelledError",
        "EmptyDataError",
        "MissingAPIKeyError",
        "PartialCacheError",
    ):
        from . import exceptions

        return getattr(exceptions, name)
    if name in (
        "generate_quarterly_contracts",
        "get_contract_dates",
        "get_expiration_date",
        "get_front_month_contract",
        "get_next_contract",
        "is_supported_contract",
        "is_supported_root",
        "parse_contract_symbol",
        "to_databento_symbol",
    ):
        from . import futures

        return getattr(futures, name)
    if name in (
        "CacheCheckResult",
        "CachedData",
        "CachedDataInfo",
        "CacheStatus",
        "DataQualityIssue",
        "DateRange",
        "DownloadProgress",
        "DownloadStatus",
        "PartitionInfo",
        "UpdateAllResult",
    ):
        from . import models

        return getattr(models, name)
    if name in ("Historical", "CacheStore"):
        from . import historical

        return getattr(historical, name)
    if name in (
        "DataReader",
        "FilesystemBackend",
        "PartitionKey",
        "StorageBackend",
        "create_backend",
    ):
        from . import storage

        return getattr(storage, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
