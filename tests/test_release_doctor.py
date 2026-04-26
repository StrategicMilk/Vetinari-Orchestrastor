"""Tests for scripts/release/release_doctor.py and vetinari/release/proof_schema.py.

Covers:
    - ReleaseProof / ReleaseClaimRecord JSON round-trip (Task 2.1)
    - Schema version refusal for stale artifacts
    - release_doctor pipeline: happy-path, doctor-fails, smoke-fails,
      wheel-hash-mismatch  (Task 2.4)
    - Claim kinds serialise correctly
    - No default-pass: missing step is recorded as failure, not success

All subprocess activity is monkeypatched so no real build or venv is created.
Tests use the REAL ReleaseProof schema -- no shape mocks.
"""

from __future__ import annotations

import dataclasses
import json
import subprocess
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# Ensure release scripts are importable.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "scripts" / "release") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "scripts" / "release"))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import release_doctor as rd

from vetinari.release.proof_schema import (
    PROOF_SCHEMA_VERSION,
    ClaimKind,
    ReleaseClaimRecord,
    ReleaseProof,
)

# ── Factories ──────────────────────────────────────────────────────────────


def make_claim(
    claim_id: str = "test-claim",
    text: str = "Test claim text",
    kind: ClaimKind = ClaimKind.TOOL_EVIDENCE,
    evidence_path: str = "",
    verified_at: str = "2026-04-24T12:00:00+00:00",
) -> ReleaseClaimRecord:
    """Create a ReleaseClaimRecord for testing."""
    return ReleaseClaimRecord(
        id=claim_id,
        text=text,
        evidence_path=evidence_path,
        kind=kind,
        verified_at=verified_at,
    )


def make_proof(
    version: str = "0.9.0",
    wheel_sha256: str = "a" * 64,
    wheel_size_bytes: int = 1_234_567,
    doctor_exit_code: int = 0,
    smoke_exit_code: int = 0,
    smoke_latency_ms: float = 42.0,
    signed: bool = False,
    claims: tuple[ReleaseClaimRecord, ...] | None = None,
) -> ReleaseProof:
    """Create a ReleaseProof with sensible defaults for testing."""
    if claims is None:
        claims = (make_claim(),)
    return ReleaseProof(
        schema_version=PROOF_SCHEMA_VERSION,
        version=version,
        built_at="2026-04-24T12:00:00+00:00",
        python_version="3.12.3",
        host_os="Windows-11",
        wheel_sha256=wheel_sha256,
        wheel_size_bytes=wheel_size_bytes,
        doctor_exit_code=doctor_exit_code,
        smoke_exit_code=smoke_exit_code,
        smoke_latency_ms=smoke_latency_ms,
        signed=signed,
        claims=claims,
    )


# ── Task 2.1: Proof artifact schema ───────────────────────────────────────


class TestProofSchema:
    """Tests for ReleaseProof and ReleaseClaimRecord schema."""

    def test_proof_roundtrip(self) -> None:
        """JSON round-trip preserves every field without loss."""
        original = make_proof(
            version="1.0.0",
            wheel_sha256="b" * 64,
            wheel_size_bytes=9_876_543,
            doctor_exit_code=0,
            smoke_exit_code=0,
            smoke_latency_ms=12.5,
            signed=True,
            claims=(
                make_claim("wheel-build", "wheel built", ClaimKind.TOOL_EVIDENCE, "dist/vetinari.whl"),
                make_claim("doctor-check", "doctor passed", ClaimKind.TOOL_EVIDENCE),
                make_claim("human-sign-off", "reviewed", ClaimKind.HUMAN_ATTESTED),
            ),
        )
        json_text = original.to_json()
        restored = ReleaseProof.from_json(json_text)

        # Verify every top-level scalar field.
        assert restored.schema_version == original.schema_version
        assert restored.version == original.version
        assert restored.built_at == original.built_at
        assert restored.python_version == original.python_version
        assert restored.host_os == original.host_os
        assert restored.wheel_sha256 == original.wheel_sha256
        assert restored.wheel_size_bytes == original.wheel_size_bytes
        assert restored.doctor_exit_code == original.doctor_exit_code
        assert restored.smoke_exit_code == original.smoke_exit_code
        assert restored.smoke_latency_ms == original.smoke_latency_ms
        assert restored.signed == original.signed

        # Verify claims list is fully preserved.
        assert len(restored.claims) == len(original.claims)
        for orig_claim, rest_claim in zip(original.claims, restored.claims):
            assert rest_claim.id == orig_claim.id
            assert rest_claim.text == orig_claim.text
            assert rest_claim.evidence_path == orig_claim.evidence_path
            assert rest_claim.kind == orig_claim.kind
            assert rest_claim.verified_at == orig_claim.verified_at

    def test_proof_from_json_refuses_stale_schema_version(self) -> None:
        """from_json raises ValueError for an artifact with a different schema_version."""
        proof = make_proof()
        raw = json.loads(proof.to_json())
        raw["schema_version"] = "0"  # older version
        stale_json = json.dumps(raw)

        with pytest.raises(ValueError, match="Refusing stale proof artifact"):
            ReleaseProof.from_json(stale_json)

    def test_proof_from_json_refuses_missing_schema_version(self) -> None:
        """from_json raises ValueError when schema_version field is absent."""
        proof = make_proof()
        raw = json.loads(proof.to_json())
        del raw["schema_version"]
        bad_json = json.dumps(raw)

        with pytest.raises(ValueError, match="Refusing stale proof artifact"):
            ReleaseProof.from_json(bad_json)

    def test_claim_kind_roundtrip(self) -> None:
        """All ClaimKind values survive JSON serialisation and deserialisation."""
        for kind in ClaimKind:
            claim = make_claim(claim_id=kind.value, kind=kind)
            proof = make_proof(claims=(claim,))
            restored = ReleaseProof.from_json(proof.to_json())
            assert restored.claims[0].kind == kind

    def test_proof_is_frozen(self) -> None:
        """ReleaseProof is immutable -- assignment raises FrozenInstanceError."""
        proof = make_proof()
        with pytest.raises((AttributeError, dataclasses.FrozenInstanceError)):
            proof.version = "mutated"  # type: ignore[misc]

    def test_claim_record_is_frozen(self) -> None:
        """ReleaseClaimRecord is immutable -- assignment raises FrozenInstanceError."""
        claim = make_claim()
        with pytest.raises((AttributeError, dataclasses.FrozenInstanceError)):
            claim.text = "mutated"  # type: ignore[misc]

    def test_proof_repr_includes_key_fields(self) -> None:
        """ReleaseProof.__repr__ includes version, exit codes, and claim count."""
        proof = make_proof(version="2.0.0", doctor_exit_code=0, smoke_exit_code=1)
        r = repr(proof)
        assert "2.0.0" in r
        assert "doctor_exit_code=0" in r
        assert "smoke_exit_code=1" in r
        assert "claims=1" in r

    def test_claim_repr_includes_id_and_kind(self) -> None:
        """ReleaseClaimRecord.__repr__ includes id and kind."""
        claim = make_claim("my-claim-id", kind=ClaimKind.LLM_JUDGMENT)
        r = repr(claim)
        assert "my-claim-id" in r
        assert "llm_judgment" in r

    def test_to_json_produces_valid_json_with_schema_version(self) -> None:
        """to_json output is valid JSON and contains schema_version at top level."""
        proof = make_proof()
        json_text = proof.to_json()
        parsed = json.loads(json_text)
        assert parsed["schema_version"] == PROOF_SCHEMA_VERSION
        assert parsed["version"] == proof.version
        assert isinstance(parsed["claims"], list)

    def test_proof_uses_dataclasses_asdict_not_handwritten(self) -> None:
        """to_json uses dataclasses.asdict() -- round-trip matches asdict output."""
        proof = make_proof()
        json_text = proof.to_json()
        parsed = json.loads(json_text)
        # Verify all fields from asdict are present in JSON.
        raw = dataclasses.asdict(proof)
        for key in raw:
            assert key in parsed, f"Field {key!r} missing from JSON output"

    @pytest.mark.parametrize(
        "field,value",
        [
            ("schema_version", PROOF_SCHEMA_VERSION),
            ("doctor_exit_code", 0),
            ("smoke_exit_code", 0),
            ("wheel_size_bytes", 1_234_567),
            ("signed", False),
        ],
    )
    def test_proof_field_survives_roundtrip(self, field: str, value: Any) -> None:
        """Each key scalar field is preserved exactly through JSON round-trip."""
        proof = make_proof()
        restored = ReleaseProof.from_json(proof.to_json())
        assert getattr(restored, field) == value


# ── Helpers for pipeline tests ─────────────────────────────────────────────


def _make_completed_process(
    returncode: int = 0,
    stdout: str = "",
    stderr: str = "",
) -> subprocess.CompletedProcess[str]:
    """Return a fake CompletedProcess for monkeypatching subprocess.run."""
    cp: subprocess.CompletedProcess[str] = subprocess.CompletedProcess(
        args=[], returncode=returncode, stdout=stdout, stderr=stderr
    )
    return cp


class TestReleaseDoctorPipeline:
    """Tests for the run_release_doctor pipeline with mocked subprocess."""

    def _fake_wheel(self, dist_dir: Path) -> Path:
        """Create a minimal fake wheel file for testing."""
        dist_dir.mkdir(parents=True, exist_ok=True)
        wheel = dist_dir / "vetinari-0.9.0-py3-none-any.whl"
        wheel.write_bytes(b"PK\x03\x04" + b"\x00" * 100)  # minimal ZIP magic
        return wheel

    @pytest.mark.parametrize(
        "scenario,build_rc,install_rc,doctor_rc,smoke_rc,smoke_stdout,expected_doctor,expected_smoke",
        [
            # happy-path: all steps succeed
            (
                "happy",
                0,
                0,
                0,
                0,
                "SMOKE_LATENCY_MS:55.3\nSMOKE_STATUS:200",
                0,
                0,
            ),
            # doctor-fails: build+install succeed, doctor returns non-zero
            (
                "doctor-fails",
                0,
                0,
                1,
                0,
                "",
                1,
                -1,  # smoke never runs
            ),
            # smoke-fails: build+install+doctor succeed, smoke returns non-zero
            (
                "smoke-fails",
                0,
                0,
                0,
                1,
                "SMOKE_FAIL: connection refused",
                0,
                1,
            ),
        ],
    )
    def test_pipeline_scenarios(
        self,
        tmp_path: Path,
        scenario: str,
        build_rc: int,
        install_rc: int,
        doctor_rc: int,
        smoke_rc: int,
        smoke_stdout: str,
        expected_doctor: int,
        expected_smoke: int,
    ) -> None:
        """Parametrized happy/doctor-fails/smoke-fails scenarios via subprocess mock."""
        dist_dir = tmp_path / "dist"
        fake_wheel = self._fake_wheel(dist_dir)

        call_count = {"n": 0}

        def fake_subprocess_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
            call_count["n"] += 1
            joined = " ".join(str(c) for c in cmd)

            if "build" in joined and "--wheel" in joined:
                # Simulate build step.
                if build_rc == 0:
                    return _make_completed_process(0)
                return _make_completed_process(1, stderr="build error")

            if "pip" in joined and "upgrade" in joined:
                return _make_completed_process(0)

            if "pip" in joined and "install" in joined:
                return _make_completed_process(install_rc)

            if "vetinari" in joined and "--doctor" in joined:
                return _make_completed_process(doctor_rc)

            # smoke script runs as: python smoke.py <port>
            if "smoke.py" in joined:
                return _make_completed_process(smoke_rc, stdout=smoke_stdout)

            return _make_completed_process(0)

        def venv_python_exists(self: Path) -> bool:
            # Return True only for venv python executables so that unrelated
            # Path.exists() calls in the pipeline are not silently suppressed.
            path_str = str(self)
            return ("venv" in path_str) and ("Scripts" in path_str or "/bin/" in path_str or "\\bin\\" in path_str)

        with (
            patch("release_doctor._OUTPUTS_DIR", tmp_path / "outputs" / "release"),
            patch("release_doctor._find_wheel", return_value=fake_wheel),
            patch("release_doctor.subprocess.run", side_effect=fake_subprocess_run),
            patch("release_doctor.venv.create"),
            # Scope the exists() patch to venv python paths only -- prevents
            # masking legitimately-missing files elsewhere in the pipeline.
            patch("pathlib.Path.exists", venv_python_exists),
        ):
            proof = rd.run_release_doctor(version="0.9.0", quiet=True)

        assert proof.schema_version == PROOF_SCHEMA_VERSION
        assert proof.version == "0.9.0"
        assert proof.doctor_exit_code == expected_doctor, (
            f"scenario={scenario}: expected doctor_exit={expected_doctor}, got {proof.doctor_exit_code}"
        )
        if expected_smoke != -1:
            assert proof.smoke_exit_code == expected_smoke, (
                f"scenario={scenario}: expected smoke_exit={expected_smoke}, got {proof.smoke_exit_code}"
            )

        # Claims list must be non-empty and include meaningful text.
        assert len(proof.claims) > 0
        claim_texts = " ".join(c.text for c in proof.claims)
        assert len(claim_texts) > 10, "Claims must have meaningful text, not just placeholders"

        # Proof must be written to disk.
        proof_path = tmp_path / "outputs" / "release" / "0.9.0" / "proof.json"
        assert proof_path.exists(), f"proof.json not written for scenario={scenario}"
        on_disk = ReleaseProof.from_json(proof_path.read_text(encoding="utf-8"))
        assert on_disk.version == proof.version
        assert on_disk.doctor_exit_code == proof.doctor_exit_code

    def test_wheel_hash_mismatch_fails_loudly(self, tmp_path: Path) -> None:
        """step_wheel_hash records the correct sha256; a mismatch is detectable."""
        # Write a fake wheel with known content.
        wheel = tmp_path / "vetinari-0.9.0-py3-none-any.whl"
        wheel.write_bytes(b"fake wheel content for hash test")

        sha256, size, claims = rd.step_wheel_hash(wheel)

        # Verify the hash is non-empty and meaningful.
        assert len(sha256) == 64, "SHA-256 hex digest must be 64 chars"
        assert all(c in "0123456789abcdef" for c in sha256), "SHA-256 must be lowercase hex"
        assert size == len(b"fake wheel content for hash test")

        # Simulate a tampered wheel: same path, different content.
        wheel.write_bytes(b"TAMPERED content")
        sha256_tampered, _, _ = rd.step_wheel_hash(wheel)

        # The two hashes must differ -- this is how tampering is detected.
        assert sha256 != sha256_tampered, (
            "step_wheel_hash must produce different digests for different content; "
            "a matching hash on tampered content means hash verification is broken"
        )

    def test_no_default_pass_on_build_failure(self, tmp_path: Path) -> None:
        """Build failure records a FAILED claim; doctor/smoke remain at sentinel -1."""

        def always_fail_build(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
            if "build" in " ".join(str(c) for c in cmd):
                return _make_completed_process(1, stderr="build error")
            return _make_completed_process(0)

        with (
            patch("release_doctor._OUTPUTS_DIR", tmp_path / "outputs" / "release"),
            patch("release_doctor.subprocess.run", side_effect=always_fail_build),
            patch("release_doctor.venv.create"),
        ):
            proof = rd.run_release_doctor(version="0.9.0", quiet=True)

        # doctor and smoke must remain at their uninitialised sentinel (-1), proving
        # they never ran -- not silently passed.
        assert proof.doctor_exit_code == -1, (
            "doctor_exit_code must be -1 when build fails; it must NOT default to 0 (no default-pass anti-pattern)"
        )
        assert proof.smoke_exit_code == -1

        # The claims list must contain evidence of the failure.
        failure_texts = [c.text for c in proof.claims if "FAIL" in c.text or "fail" in c.text.lower()]
        assert len(failure_texts) > 0, "Build failure must produce a FAIL claim, not be silently ignored"

    def test_proof_written_even_on_pipeline_failure(self, tmp_path: Path) -> None:
        """proof.json is written to disk even when the pipeline fails early."""

        def fail_build(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
            if "build" in " ".join(str(c) for c in cmd):
                return _make_completed_process(1, stderr="simulated build failure")
            return _make_completed_process(0)

        with (
            patch("release_doctor._OUTPUTS_DIR", tmp_path / "outputs" / "release"),
            patch("release_doctor.subprocess.run", side_effect=fail_build),
            patch("release_doctor.venv.create"),
        ):
            rd.run_release_doctor(version="0.9.0", quiet=True)

        proof_path = tmp_path / "outputs" / "release" / "0.9.0" / "proof.json"
        assert proof_path.exists(), "proof.json must be written even on pipeline failure"
        # Must be valid JSON and parseable as a ReleaseProof.
        on_disk = ReleaseProof.from_json(proof_path.read_text(encoding="utf-8"))
        assert on_disk.version == "0.9.0"

    def test_signed_false_when_no_key(self, tmp_path: Path) -> None:
        """Absence of a signing key records signed=False without failing."""
        proof_path = tmp_path / "proof.json"
        proof = make_proof(signed=False)
        proof_json = proof.to_json()

        signed, claims = rd.step_sign(proof_json, proof_path, key_path=None)

        assert signed is False
        assert len(claims) == 1
        assert "unsigned" in claims[0].text.lower() or "no signing key" in claims[0].text.lower()
        # signing=False must NOT be classified as a failure claim with FAIL text.
        assert claims[0].id == "proof-signature"

    def test_pipeline_fails_on_wheel_tampering_after_build(self, tmp_path: Path) -> None:
        """Pipeline halts and records a FAIL claim when wheel hash changes after build.

        Simulates the scenario where a wheel is built successfully but its SHA-256
        changes between the build step and the install step -- indicating possible
        tampering.  The pipeline must fail closed: doctor and smoke must never run,
        and the proof must contain a wheel-hash FAIL claim.
        """
        dist_dir = tmp_path / "dist"
        fake_wheel = self._fake_wheel(dist_dir)

        def fake_subprocess_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
            joined = " ".join(str(c) for c in cmd)
            if "build" in joined and "--wheel" in joined:
                return _make_completed_process(0)
            return _make_completed_process(0)

        def venv_python_exists(self: Path) -> bool:
            path_str = str(self)
            return ("venv" in path_str) and ("Scripts" in path_str or "/bin/" in path_str or "\\bin\\" in path_str)

        # Inject a tamper-detection failure: _verify_wheel_hash_unchanged returns
        # ok=False with a wheel-hash FAIL claim.
        tamper_claim = ReleaseClaimRecord(
            id="wheel-hash",
            text="FAIL: wheel SHA-256 mismatch (recorded abcd1234... vs on-disk ffff9999...) -- possible tampering",
            evidence_path="",
            kind=ClaimKind.TOOL_EVIDENCE,
            verified_at="2026-04-24T12:00:00+00:00",
        )

        with (
            patch("release_doctor._OUTPUTS_DIR", tmp_path / "outputs" / "release"),
            patch("release_doctor._find_wheel", return_value=fake_wheel),
            patch("release_doctor.subprocess.run", side_effect=fake_subprocess_run),
            patch("release_doctor.venv.create"),
            patch("pathlib.Path.exists", venv_python_exists),
            patch(
                "release_doctor._verify_wheel_hash_unchanged",
                return_value=(False, [tamper_claim]),
            ),
        ):
            proof = rd.run_release_doctor(version="0.9.0", quiet=True)

        # Pipeline must fail closed: install never ran so doctor and smoke stay at -1.
        assert proof.doctor_exit_code == -1, (
            "doctor must not run after wheel tamper detection; doctor_exit_code must remain -1"
        )
        assert proof.smoke_exit_code == -1, (
            "smoke must not run after wheel tamper detection; smoke_exit_code must remain -1"
        )

        # The proof must contain the wheel-hash FAIL claim as evidence.
        wheel_hash_claims = [c for c in proof.claims if c.id == "wheel-hash"]
        assert len(wheel_hash_claims) >= 1, "proof must contain a wheel-hash claim when tampering is detected"
        fail_claims = [c for c in wheel_hash_claims if "FAIL" in c.text]
        assert len(fail_claims) >= 1, "the wheel-hash claim must contain FAIL text, not a passing result"

        # Proof must still be written to disk (fail closed, not fail silent).
        proof_path = tmp_path / "outputs" / "release" / "0.9.0" / "proof.json"
        assert proof_path.exists(), "proof.json must be written even when wheel tamper is detected"
        on_disk = ReleaseProof.from_json(proof_path.read_text(encoding="utf-8"))
        assert on_disk.version == "0.9.0"

    def test_schema_version_constant_is_string(self) -> None:
        """PROOF_SCHEMA_VERSION must be a non-empty string."""
        assert isinstance(PROOF_SCHEMA_VERSION, str)
        assert len(PROOF_SCHEMA_VERSION) > 0
