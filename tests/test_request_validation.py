"""Tests for shared Litestar request body validation helpers."""

from __future__ import annotations

from vetinari.web.request_validation import json_object_body


def test_json_object_body_accepts_missing_and_object_payloads() -> None:
    assert json_object_body(None) == {}
    assert json_object_body({"goal": "ship"}) == {"goal": "ship"}


def test_json_object_body_rejects_non_object_payloads() -> None:
    assert json_object_body(["not", "an", "object"]) is None
    assert json_object_body("not an object") is None
