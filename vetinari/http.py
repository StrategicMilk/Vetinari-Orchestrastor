"""Shared HTTP session factory for connection pooling and lifecycle management.

Provides a centralized way to create ``requests.Session`` instances with
consistent retry middleware, timeout defaults, and connection pooling.
All adapters, tools, and internal HTTP callers should use this module
instead of creating raw ``requests.Session()`` or calling ``requests.get/post``
directly.
"""

from __future__ import annotations

import logging
import threading

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_TIMEOUT = 30  # seconds — override per-call with timeout= kwarg
DEFAULT_RETRIES = 3  # total retry attempts for transient failures
DEFAULT_BACKOFF_FACTOR = 0.5  # exponential backoff: 0.5s, 1s, 2s
DEFAULT_POOL_CONNECTIONS = 10  # urllib3 connection pool size
DEFAULT_POOL_MAXSIZE = 20  # max connections per host

# Retry on these HTTP status codes (server errors + rate limiting)
RETRY_STATUS_CODES = frozenset({429, 500, 502, 503, 504})

# Registry of all sessions created via this module for cleanup on shutdown
_MAX_SESSIONS = 100  # prevent unbounded session accumulation
_sessions: list[requests.Session] = []
_sessions_lock = threading.Lock()


def create_session(
    *,
    retries: int = DEFAULT_RETRIES,
    backoff_factor: float = DEFAULT_BACKOFF_FACTOR,
    timeout: float = DEFAULT_TIMEOUT,
    pool_connections: int = DEFAULT_POOL_CONNECTIONS,
    pool_maxsize: int = DEFAULT_POOL_MAXSIZE,
    headers: dict[str, str] | None = None,
) -> requests.Session:
    """Create a configured ``requests.Session`` with retry and pooling.

    The session is registered for cleanup on shutdown via ``close_all()``.

    Args:
        retries: Maximum number of retry attempts for transient failures.
        backoff_factor: Exponential backoff factor between retries.
        timeout: Default timeout in seconds (applied per-request if not overridden).
        pool_connections: Number of connection pools to cache.
        pool_maxsize: Maximum number of connections per pool.
        headers: Optional default headers to set on the session.

    Returns:
        A configured ``requests.Session`` instance.
    """
    session = requests.Session()

    retry_strategy = Retry(
        total=retries,
        backoff_factor=backoff_factor,
        status_forcelist=list(RETRY_STATUS_CODES),
        allowed_methods=["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD"],
        raise_on_status=False,
    )

    adapter = HTTPAdapter(
        max_retries=retry_strategy,
        pool_connections=pool_connections,
        pool_maxsize=pool_maxsize,
    )

    session.mount("http://", adapter)
    session.mount("https://", adapter)

    if headers:
        session.headers.update(headers)

    # Store default timeout as a custom attribute for middleware use
    session._default_timeout = timeout  # type: ignore[attr-defined]

    with _sessions_lock:
        # Evict oldest session if at capacity to prevent unbounded growth
        if len(_sessions) >= _MAX_SESSIONS:
            oldest = _sessions.pop(0)
            try:
                oldest.close()
            except Exception:
                logger.warning("Error closing evicted HTTP session", exc_info=True)
        _sessions.append(session)

    return session


def close_all() -> None:
    """Close all sessions created by this module.

    Called during graceful shutdown to release connection pools and
    underlying sockets. Safe to call multiple times.
    """
    with _sessions_lock:
        for session in _sessions:
            try:
                session.close()
            except Exception:
                logger.warning("Error closing HTTP session", exc_info=True)
        _sessions.clear()
    logger.info("All HTTP sessions closed")


def session_count() -> int:
    """Return the number of active tracked sessions.

    Returns:
        Number of sessions currently registered.
    """
    with _sessions_lock:
        return len(_sessions)
