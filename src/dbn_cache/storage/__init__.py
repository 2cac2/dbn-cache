"""Pluggable storage backends for the data cache.

``FilesystemBackend`` (partitioned Parquet on disk) is the default. ``SqlBackend``
(SQLAlchemy; SQLite by default, Postgres/MySQL for shared caches) is imported
lazily so importing this package never pulls in ``sqlalchemy``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .base import DataReader, PartitionKey, StorageBackend
from .factory import create_backend, scheme_of
from .filesystem import FilesystemBackend

if TYPE_CHECKING:
    from .sql import SqlBackend

__all__ = [
    "DataReader",
    "FilesystemBackend",
    "PartitionKey",
    "SqlBackend",
    "StorageBackend",
    "create_backend",
    "scheme_of",
]


def __getattr__(name: str) -> object:
    if name == "SqlBackend":
        from .sql import SqlBackend

        return SqlBackend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
