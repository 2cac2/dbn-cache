"""Backend selection from a connection URL."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base import StorageBackend

# SQLAlchemy URL schemes routed to the SQL backend. DuckDB works opportunistically
# via the ``duckdb-engine`` SQLAlchemy dialect if installed.
SQL_SCHEMES = frozenset(
    {
        "sqlite",
        "postgresql",
        "postgres",
        "mysql",
        "mariadb",
        "mssql",
        "oracle",
        "duckdb",
    }
)


def scheme_of(url: str) -> str:
    """Return the lowercased dialect scheme of a connection URL (no driver)."""
    if "://" not in url:
        return ""
    return url.split("://", 1)[0].split("+", 1)[0].lower()


def create_backend(url: str, cache_dir: Path | None = None) -> StorageBackend:
    """Create a storage backend for a connection ``url``.

    Args:
        url: A SQLAlchemy-style connection URL (e.g. ``sqlite:///cache.db``,
            ``postgresql://user:pw@host/db``), or ``file:///path`` to select the
            filesystem (Parquet-on-disk) backend explicitly.
        cache_dir: Directory for sidecar lock files (and the SQLite file's parent
            when a relative SQLite path is used).

    Raises:
        ValueError: If the URL scheme is not supported.
    """
    scheme = scheme_of(url)
    if scheme in ("file", "filesystem"):
        from ..utils import get_default_cache_dir
        from .filesystem import FilesystemBackend

        path_part = url.split("://", 1)[1] if "://" in url else ""
        if path_part:
            directory = Path(path_part)
        else:
            directory = cache_dir or get_default_cache_dir()
        return FilesystemBackend(directory)
    if scheme in SQL_SCHEMES:
        from .sql import SqlBackend

        return SqlBackend(url, cache_dir=cache_dir)
    msg = (
        f"Unsupported cache URL scheme: {scheme!r}. "
        "Use a SQLAlchemy connection URL such as 'sqlite:///path/to/cache.db', "
        "'postgresql://user:pw@host/db', 'mysql://user:pw@host/db', or "
        "'file:///path/to/dir' for the filesystem backend."
    )
    raise ValueError(msg)
