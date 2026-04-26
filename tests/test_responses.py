"""Tests for vetinari.web.responses — standard API response envelope helpers."""

from __future__ import annotations

import pytest


class TestSuccessResponse:
    """Tests for success_response."""

    def test_returns_dict(self):
        """success_response returns a plain dict, not a Flask Response."""
        from vetinari.web.responses import success_response

        result = success_response({"items": [1, 2]})
        assert isinstance(result, dict)

    def test_status_field_is_ok(self):
        """Top-level status field is always 'ok'."""
        from vetinari.web.responses import success_response

        assert success_response()["status"] == "ok"

    def test_data_field_carries_payload(self):
        """Payload is nested under the data key."""
        from vetinari.web.responses import success_response

        payload = {"x": 42, "y": "hello"}
        result = success_response(payload)
        assert result["data"] == payload

    def test_default_code_is_200(self):
        """Default HTTP code in envelope is 200."""
        from vetinari.web.responses import success_response

        assert success_response()["code"] == 200

    def test_custom_code_is_stored(self):
        """Non-default code is preserved in the envelope body."""
        from vetinari.web.responses import success_response

        assert success_response(code=201)["code"] == 201

    def test_none_data_allowed(self):
        """data=None is a valid payload (e.g. for delete endpoints)."""
        from vetinari.web.responses import success_response

        result = success_response(None)
        assert result["data"] is None
        assert result["status"] == "ok"

    @pytest.mark.parametrize(
        "payload",
        [
            [],
            [1, 2, 3],
            "a string",
            42,
            True,
        ],
    )
    def test_non_dict_payloads_accepted(self, payload):
        """success_response accepts any JSON-serializable data type."""
        from vetinari.web.responses import success_response

        assert success_response(payload)["data"] == payload
