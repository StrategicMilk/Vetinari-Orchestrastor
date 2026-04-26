"""Tests for vetinari.http — shared HTTP session factory."""

from __future__ import annotations

import pytest
import requests

from vetinari.http import close_all, create_session, session_count


class TestCreateSession:
    """Test session creation and configuration."""

    def setup_method(self) -> None:
        """Reset session registry before each test."""
        close_all()

    def teardown_method(self) -> None:
        """Clean up sessions after each test."""
        close_all()

    def test_create_session_returns_session(self) -> None:
        """create_session() returns a requests.Session instance."""
        session = create_session()
        assert isinstance(session, requests.Session)

    def test_create_session_has_retry_adapter(self) -> None:
        """Created session has HTTPAdapter with retry on both http and https."""
        session = create_session()
        http_adapter = session.get_adapter("http://example.com")
        https_adapter = session.get_adapter("https://example.com")
        assert http_adapter is not None
        assert hasattr(http_adapter, "max_retries")
        assert https_adapter is not None
        assert hasattr(https_adapter, "max_retries")
        # Verify retry is configured
        assert http_adapter.max_retries.total == 3  # default retries

    def test_create_session_custom_retries(self) -> None:
        """Custom retry count is respected."""
        session = create_session(retries=5)
        adapter = session.get_adapter("http://example.com")
        assert adapter.max_retries.total == 5

    def test_create_session_custom_headers(self) -> None:
        """Custom headers are set on the session."""
        session = create_session(headers={"X-Custom": "test-value"})
        assert session.headers["X-Custom"] == "test-value"

    def test_create_session_default_timeout_attribute(self) -> None:
        """Default timeout is stored as custom attribute."""
        session = create_session(timeout=42)
        assert session._default_timeout == 42  # type: ignore[attr-defined]

    def test_session_count_tracks_created_sessions(self) -> None:
        """session_count() correctly tracks number of active sessions."""
        assert session_count() == 0
        create_session()
        assert session_count() == 1
        create_session()
        assert session_count() == 2


class TestCloseAll:
    """Test session cleanup."""

    def setup_method(self) -> None:
        close_all()

    def test_close_all_clears_registry(self) -> None:
        """close_all() resets session count to zero."""
        create_session()
        create_session()
        assert session_count() == 2
        close_all()
        assert session_count() == 0

    def test_close_all_idempotent(self) -> None:
        """Calling close_all() multiple times is safe."""
        create_session()
        close_all()
        close_all()  # Second call should not raise
        assert session_count() == 0

    def test_close_all_handles_already_closed_session(self) -> None:
        """close_all() handles sessions that were already closed."""
        session = create_session()
        session.close()  # Close manually first
        close_all()  # Should not raise
        assert session_count() == 0
