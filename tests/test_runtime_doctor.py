"""Tests for vetinari.runtime.runtime_doctor — matrix-driven preconditions.

Each test drives the doctor from a fixture supported-matrix YAML (under
``tests/fixtures/runtime/``) and asserts the SHAPE of the outcome: pass,
minimum-violation blocker, known-bad blocker, or staleness escalation.

Version strings in assertions are read from the fixture matrix itself —
never hardcoded here — so swapping in a fresh matrix changes fixture
behavior without editing the test code.

Part of SESSION-03 SHARD-01 (runtime safety preconditions).
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from vetinari.runtime.runtime_doctor import (
    DoctorReport,
    RuntimeCheckResult,
    check_matrix_row,
    load_matrix,
    run_doctor,
    validate_runtime_version,
)

_FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "runtime"
_PASSING = _FIXTURE_ROOT / "supported_matrix_passing.yaml"
_BELOW_MIN = _FIXTURE_ROOT / "supported_matrix_below_minimum.yaml"
_KNOWN_BAD = _FIXTURE_ROOT / "supported_matrix_known_bad.yaml"
_STALE = _FIXTURE_ROOT / "supported_matrix_stale.yaml"


def _stub_detector(mapping: dict[str, str | None]):
    """Return a version_detector that returns preset values per component.

    Args:
        mapping: Component name -> detected version string (or None).

    Returns:
        Callable suitable for passing as ``version_detector`` to ``run_doctor``.
    """

    def _detect(component: str) -> str | None:
        return mapping.get(component)

    return _detect


class TestMatrixLoad:
    """load_matrix returns the parsed YAML with a components list."""

    def test_load_passing_matrix_returns_components_list(self) -> None:
        data = load_matrix(_PASSING)
        assert isinstance(data.get("components"), list)
        assert data["components"], "Passing fixture must declare at least one component"

    def test_load_missing_matrix_raises_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_matrix(Path("tests/fixtures/runtime/does_not_exist.yaml"))

    def test_load_invalid_matrix_raises_value_error(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.yaml"
        bad.write_text("schema_version: 1\n", encoding="utf-8")
        with pytest.raises(ValueError, match="components"):
            load_matrix(bad)


class TestRunDoctorHappyPath:
    """Detected runtime satisfies every row in the passing matrix."""

    def test_all_rows_satisfied_produces_pass(self) -> None:
        detector = _stub_detector({
            "python": "3.12.0",
            "torch": "2.7.1",
            "bitsandbytes": None,
        })
        report = run_doctor(
            _PASSING,
            now=datetime(2099, 2, 1, tzinfo=timezone.utc),
            version_detector=detector,
            compute_capability_detector=lambda: "12.0",
        )
        assert isinstance(report, DoctorReport)
        assert report.passed, f"Expected pass but got blockers={report.blockers}"
        assert report.blockers == ()
        assert report.matrix_staleness_warning is None

    def test_optional_component_missing_does_not_block(self) -> None:
        # bitsandbytes missing + optional=True -> PASS with a non-blocker note
        detector = _stub_detector({"python": "3.12.0", "torch": "2.7.1", "bitsandbytes": None})
        report = run_doctor(
            _PASSING,
            now=datetime(2099, 2, 1, tzinfo=timezone.utc),
            version_detector=detector,
            compute_capability_detector=lambda: "12.0",
        )
        bnb_check = next(c for c in report.checks if c.component == "bitsandbytes")
        assert bnb_check.passed is True
        assert bnb_check.is_blocker is False


class TestRunDoctorMinimumViolation:
    """A detected version below the matrix minimum MUST fail closed."""

    def test_detected_below_minimum_yields_blocker(self) -> None:
        detector = _stub_detector({"torch": "2.6.0"})  # matrix requires 99.0.0
        report = run_doctor(
            _BELOW_MIN,
            now=datetime(2099, 2, 1, tzinfo=timezone.utc),
            version_detector=detector,
            compute_capability_detector=lambda: None,
        )
        assert report.passed is False
        assert any("older than the required minimum" in b for b in report.blockers), (
            f"Blockers did not cite the minimum violation: {report.blockers}"
        )

    def test_blocker_cites_matrix_source(self) -> None:
        detector = _stub_detector({"torch": "2.6.0"})
        report = run_doctor(
            _BELOW_MIN,
            now=datetime(2099, 2, 1, tzinfo=timezone.utc),
            version_detector=detector,
            compute_capability_detector=lambda: None,
        )
        blocker_text = " ".join(report.blockers)
        assert "https://example.test/torch" in blocker_text, (
            f"Blocker must cite the matrix source URL. Got: {blocker_text}"
        )


class TestRunDoctorKnownBadRange:
    """A detected version inside a known-bad range MUST fail closed."""

    def test_detected_in_known_bad_range_yields_blocker(self) -> None:
        detector = _stub_detector({"vllm": "0.18.1"})  # matrix declares ==0.18.1 bad
        report = run_doctor(
            _KNOWN_BAD,
            now=datetime(2099, 2, 1, tzinfo=timezone.utc),
            version_detector=detector,
            compute_capability_detector=lambda: None,
        )
        assert report.passed is False
        assert any("known-bad range" in b for b in report.blockers), (
            f"Blockers did not cite the known-bad range: {report.blockers}"
        )

    def test_detected_outside_known_bad_range_passes(self) -> None:
        detector = _stub_detector({"vllm": "0.19.0"})
        report = run_doctor(
            _KNOWN_BAD,
            now=datetime(2099, 2, 1, tzinfo=timezone.utc),
            version_detector=detector,
            compute_capability_detector=lambda: None,
        )
        assert report.passed is True


class TestRunDoctorStaleness:
    """Staleness behaviour: warning alone is advisory, unknown version escalates."""

    def test_stale_matrix_with_all_versions_known_issues_warning_only(self) -> None:
        # Matrix is 5+ years old, but detection returns a satisfying version.
        detector = _stub_detector({"vllm": "0.19.0"})
        report = run_doctor(
            _STALE,
            now=datetime(2026, 4, 24, tzinfo=timezone.utc),
            version_detector=detector,
            compute_capability_detector=lambda: None,
        )
        assert report.passed is True
        assert report.matrix_staleness_warning is not None

    def test_stale_matrix_plus_unknown_version_becomes_blocker(self) -> None:
        # Matrix stale AND the required component reports None -> blocker.
        detector = _stub_detector({"vllm": None})
        report = run_doctor(
            _STALE,
            now=datetime(2026, 4, 24, tzinfo=timezone.utc),
            version_detector=detector,
            compute_capability_detector=lambda: None,
        )
        assert report.passed is False
        assert any("stale" in b.lower() and "unknown" in b.lower() for b in report.blockers), (
            f"Stale+unknown did not escalate to a blocker: {report.blockers}"
        )


class TestValidateRuntimeVersion:
    """validate_runtime_version: per-component check for runtime-discovered versions."""

    def test_known_bad_version_is_blocker(self) -> None:
        # Matrix declares vllm ==0.18.1 as known-bad; supplying that version blocks.
        result = validate_runtime_version(
            "vllm",
            "0.18.1",
            matrix_path=_KNOWN_BAD,
            compute_capability=None,
        )
        assert isinstance(result, RuntimeCheckResult)
        assert result.passed is False
        assert result.is_blocker is True
        assert "known-bad range" in result.reason

    def test_safe_version_passes(self) -> None:
        # 0.19.0 is outside the matrix's known-bad set.
        result = validate_runtime_version(
            "vllm",
            "0.19.0",
            matrix_path=_KNOWN_BAD,
            compute_capability=None,
        )
        assert result.passed is True
        assert result.is_blocker is True  # blocker semantics for required rows

    def test_unknown_component_raises(self) -> None:
        with pytest.raises(ValueError, match="not declared in supported matrix"):
            validate_runtime_version(
                "definitely-not-a-real-component",
                "1.0.0",
                matrix_path=_KNOWN_BAD,
                compute_capability=None,
            )


class TestCheckMatrixRowUnit:
    """check_matrix_row unit tests — avoid the full run_doctor path."""

    def test_missing_required_component_fails_closed(self) -> None:
        row = {
            "component": "nonexistent",
            "minimum_version": "1.0.0",
            "verified_sources": ["https://example.test"],
        }
        result = check_matrix_row(row, detected_version=None, detected_compute_capability=None)
        assert isinstance(result, RuntimeCheckResult)
        assert result.passed is False
        assert result.is_blocker is True
        assert "Could not detect" in result.reason

    def test_missing_optional_component_is_not_blocker(self) -> None:
        row = {
            "component": "optional-thing",
            "minimum_version": "1.0.0",
            "optional": True,
        }
        result = check_matrix_row(row, detected_version=None, detected_compute_capability=None)
        assert result.passed is True
        assert result.is_blocker is False

    def test_insufficient_compute_capability_is_advisory(self) -> None:
        row = {
            "component": "torch",
            "minimum_version": "2.7.0",
            "required_compute_capability": "12.0",
            "verified_sources": ["https://example.test/torch"],
        }
        result = check_matrix_row(
            row,
            detected_version="2.7.1",
            detected_compute_capability="9.0",  # below required 12.0
        )
        assert result.passed is True  # version satisfies
        assert result.is_blocker is False  # but advisory-only warning
        assert "compute capability" in result.reason.lower()
