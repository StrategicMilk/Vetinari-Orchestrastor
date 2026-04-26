"""Behavior tests for SESSION-33E.2 batch-C check-script defect fixes.

Each test is a targeted regression guard that would FAIL against the old
(pre-fix) version of the script it covers.  Tests are organized by defect
number, matching the SESSION-33E.2 spec.

Defects covered:
  1  check_test_quality.py — context-manager patch not detected (VET242)
  2  check_test_quality.py — mock assert methods not counted as assertions (VET241)
  3  check_test_quality.py — comment/string lines cause false positives
  4  check_test_quality.py — multiline ambiguous-status call not detected (VET240)
  5  (covered in test_governance_helpers.py — VET242/VET243 extension)
  6  check_config_wiring.py — comment/string mentions counted as live wiring
  7  check_config_wiring.py — bare substring match too loose
  8  check_wiring_audit.py — singleton inner null-check not verified (VET200)
  9  check_wiring_audit.py — multiline event calls missed (VET202/VET203)
  10 check_wiring_audit.py — comment/string event names counted as publishers
  11 handoff_bundle.py + check_handoff_bundle.py — hardcoded absolute path
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"


def _import_script(name: str):
    """Import a scripts/ module by stem, bypassing the package system.

    Args:
        name: Script filename stem (e.g. ``"check_test_quality"``).

    Returns:
        Imported module object.
    """
    path = _SCRIPTS_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None, f"Could not find script: {path}"
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


# ---------------------------------------------------------------------------
# Defect 1 — context-manager patch detection (VET242)
# ---------------------------------------------------------------------------


class TestDefect1ContextManagerPatch:
    """check_test_quality: context-manager form of patch() must fire VET242."""

    def test_ctx_manager_patch_fires_vet242(self, tmp_path: Path) -> None:
        """check_self_mocking() must detect ``with patch("vetinari.X"):``.

        Old bug: only ``@patch("vetinari.X")`` decorator form was matched;
        context-manager form was invisible to the checker.

        The check fires when the test file name maps to a real module and the
        patched target is a locally defined symbol in that module.
        """
        mod = _import_script("check_test_quality")

        # test_preflight.py patching vetinari.preflight.detect_hardware —
        # module_stem is "vetinari.preflight", which exists in the repo.
        fake = tmp_path / "test_preflight.py"
        fake.write_text(
            'from unittest.mock import patch\n'
            'def test_ctx_patch():\n'
            '    with patch("vetinari.preflight.detect_hardware"):\n'
            '        assert True\n',
            encoding="utf-8",
        )
        lines = fake.read_text(encoding="utf-8").splitlines()
        violations = mod.check_self_mocking(fake, lines)

        assert violations, (
            "check_self_mocking() returned no violation for context-manager "
            "patch of 'vetinari.preflight.detect_hardware' in test_preflight.py"
        )
        codes = [code for _ln, code, _sev, _msg in violations]
        assert "VET242" in codes, f"Expected VET242; got: {codes}"

    def test_decorator_patch_still_fires_vet242(self, tmp_path: Path) -> None:
        """Decorator form @patch() must still fire VET242 after the fix."""
        mod = _import_script("check_test_quality")

        # test_preflight.py patching vetinari.preflight.prompt_and_install —
        # module_stem is "vetinari.preflight", which exists in the repo.
        fake = tmp_path / "test_preflight.py"
        fake.write_text(
            'from unittest.mock import patch\n'
            '@patch("vetinari.preflight.prompt_and_install")\n'
            'def test_dec_patch(mock_install):\n'
            '    assert True\n',
            encoding="utf-8",
        )
        lines = fake.read_text(encoding="utf-8").splitlines()
        violations = mod.check_self_mocking(fake, lines)

        codes = [code for _ln, code, _sev, _msg in violations]
        assert "VET242" in codes, (
            "Decorator-form @patch of 'vetinari.preflight.*' in test_preflight.py must still fire VET242"
        )


# ---------------------------------------------------------------------------
# Defect 2 — mock assert methods counted as assertions (VET241)
# ---------------------------------------------------------------------------


class TestDefect2MockAssertMethods:
    """check_test_quality: .assert_called_once_with() etc. must count as assertions."""

    def test_assert_called_once_with_counts_as_assertion(self, tmp_path: Path) -> None:
        """A test body with only .assert_called_once_with() must NOT trigger VET241.

        Old bug: mock assert methods were not recognised, so a test that used
        only mock asserts was (incorrectly) flagged as zero-assert.
        """
        mod = _import_script("check_test_quality")

        fake = tmp_path / "test_mock_methods.py"
        fake.write_text(
            'from unittest.mock import MagicMock\n'
            'def test_with_mock_assert():\n'
            '    m = MagicMock()\n'
            '    m("x")\n'
            '    m.assert_called_once_with("x")\n',
            encoding="utf-8",
        )
        lines = fake.read_text(encoding="utf-8").splitlines()
        violations = mod.check_zero_assert(fake, lines)

        # Mock assert methods should satisfy the assertion requirement
        vet241_violations = [v for v in violations if v[1] == "VET241"]
        assert not vet241_violations, (
            "check_zero_assert() incorrectly flagged a test that uses "
            ".assert_called_once_with() as zero-assert"
        )

    @pytest.mark.parametrize(
        "method",
        [
            "assert_called_once_with",
            "assert_called_once",
            "assert_called_with",
            "assert_called",
            "assert_not_called",
        ],
    )
    def test_all_mock_assert_variants_count(self, tmp_path: Path, method: str) -> None:
        """Each mock assert method variant must satisfy the assertion requirement."""
        mod = _import_script("check_test_quality")

        fake = tmp_path / f"test_{method}.py"
        fake.write_text(
            f'from unittest.mock import MagicMock\n'
            f'def test_uses_{method}():\n'
            f'    m = MagicMock()\n'
            f'    m.{method}()\n',
            encoding="utf-8",
        )
        lines = fake.read_text(encoding="utf-8").splitlines()
        violations = mod.check_zero_assert(fake, lines)

        vet241 = [v for v in violations if v[1] == "VET241"]
        assert not vet241, (
            f"check_zero_assert() incorrectly flagged .{method}() as zero-assert"
        )


# ---------------------------------------------------------------------------
# Defect 3 — comment/string false positives (VET241)
# ---------------------------------------------------------------------------


class TestDefect3CommentFalsePositives:
    """check_test_quality: comment/docstring lines must not count as assertions."""

    def test_commented_assert_does_not_satisfy_assertion(self, tmp_path: Path) -> None:
        """A commented-out assert must NOT satisfy the zero-assert check.

        Old bug: comment lines were scanned as code, so ``# assert x``
        was counted as an assertion and the test was not flagged.
        """
        mod = _import_script("check_test_quality")

        fake = tmp_path / "test_comment_assert.py"
        fake.write_text(
            'def test_only_comment():\n'
            '    x = 1 + 1\n'
            '    # assert x == 2\n',
            encoding="utf-8",
        )
        lines = fake.read_text(encoding="utf-8").splitlines()
        violations = mod.check_zero_assert(fake, lines)

        vet241 = [v for v in violations if v[1] == "VET241"]
        assert vet241, (
            "check_zero_assert() should flag a test whose only 'assert' is in a comment"
        )

    def test_docstring_assert_does_not_satisfy_assertion(self, tmp_path: Path) -> None:
        """An assert inside a docstring must NOT satisfy the zero-assert check."""
        mod = _import_script("check_test_quality")

        fake = tmp_path / "test_docstring_assert.py"
        fake.write_text(
            'def test_docstring_only():\n'
            '    """assert x == 1 -- example"""\n'
            '    x = 1\n',
            encoding="utf-8",
        )
        lines = fake.read_text(encoding="utf-8").splitlines()
        violations = mod.check_zero_assert(fake, lines)

        vet241 = [v for v in violations if v[1] == "VET241"]
        assert vet241, (
            "check_zero_assert() should flag a test whose only 'assert' is in a docstring"
        )


# ---------------------------------------------------------------------------
# Defect 4 — multiline ambiguous-status detection (VET240)
# ---------------------------------------------------------------------------


class TestDefect4MultilineAmbiguousStatus:
    """check_test_quality: ambiguous-status calls spanning multiple lines must fire VET240."""

    def test_multiline_ambiguous_status_fires_vet240(self, tmp_path: Path) -> None:
        """check_ambiguous_status() must detect a bare status assert with backslash continuation.

        Old bug: only single physical lines were checked.  A bare status assert
        using a trailing backslash continuation::

            assert result.status \\
                # bare truthiness check split across lines

        has ``.status`` at the end of the first physical line (before the ``\\``),
        and the multiline joiner must strip the backslash and join so the regex
        still fires on the resulting logical line.
        """
        mod = _import_script("check_test_quality")

        fake = tmp_path / "test_multi_status.py"
        # The backslash continuation: "assert result.status \\" joined with next line
        # produces "assert result.status  x = 1" — but the regex requires $ or ,/#
        # after .status.  Use a comment-only continuation so the joined form ends
        # at .status (comment lines are skipped by the joiner).
        fake.write_text(
            'def test_multiline_status():\n'
            '    result = run()\n'
            '    assert result.status\\\n'
            '\n',
            encoding="utf-8",
        )
        lines = fake.read_text(encoding="utf-8").splitlines()
        violations = mod.check_ambiguous_status(fake, lines)

        assert violations, (
            "check_ambiguous_status() did not detect bare status assert with "
            "trailing backslash continuation"
        )
        codes = [code for _ln, code, _sev, _msg in violations]
        assert "VET240" in codes, f"Expected VET240; got: {codes}"

    def test_single_line_ambiguous_status_still_fires(self, tmp_path: Path) -> None:
        """Single-line bare status truthiness check must still fire VET240."""
        mod = _import_script("check_test_quality")

        fake = tmp_path / "test_single_status.py"
        fake.write_text(
            'def test_single_status():\n'
            '    result = run()\n'
            '    assert result.status\n',
            encoding="utf-8",
        )
        lines = fake.read_text(encoding="utf-8").splitlines()
        violations = mod.check_ambiguous_status(fake, lines)

        codes = [code for _ln, code, _sev, _msg in violations]
        assert "VET240" in codes, "Single-line bare status truthiness check must fire VET240"


# ---------------------------------------------------------------------------
# Defect 6 — config wiring: comment/string mentions not counted as live wiring
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# SESSION-34D1 - stronger test-quality proof checks
# ---------------------------------------------------------------------------


class TestSession34D1TestQualityProof:
    """check_test_quality: weak proof patterns must be detected explicitly."""

    def test_string_assert_does_not_satisfy_zero_assert(self, tmp_path: Path) -> None:
        """An assert-looking string literal must not count as executable assertion."""
        mod = _import_script("check_test_quality")

        fake = tmp_path / "test_string_assert.py"
        fake.write_text(
            'def test_string_only():\n'
            '    message = "assert response.status_code == 200"\n'
            '    return message\n',
            encoding="utf-8",
        )
        violations = mod.check_zero_assert(fake, fake.read_text(encoding="utf-8").splitlines())

        assert "VET241" in [code for _ln, code, _sev, _msg in violations]

    def test_bare_pytest_raises_reference_does_not_satisfy_zero_assert(self, tmp_path: Path) -> None:
        """A bare pytest.raises reference is not an exception-path assertion."""
        mod = _import_script("check_test_quality")

        fake = tmp_path / "test_bare_raises.py"
        fake.write_text(
            'import pytest\n'
            'def test_bare_raises_reference():\n'
            '    checker = pytest.raises\n'
            '    checker\n',
            encoding="utf-8",
        )
        violations = mod.check_zero_assert(fake, fake.read_text(encoding="utf-8").splitlines())

        assert "VET241" in [code for _ln, code, _sev, _msg in violations]

    def test_not_500_assertion_fires_vet244(self, tmp_path: Path) -> None:
        """A synthetic status_code != 500 assertion must be flagged."""
        mod = _import_script("check_test_quality")
        path = tmp_path / "test_status_weak.py"
        path.write_text(
            'def test_route_only_checks_not_500():\n'
            '    response = object()\n'
            '    assert response.status_code != 500\n',
            encoding="utf-8",
        )
        lines = path.read_text(encoding="utf-8").splitlines()

        violations = mod.check_weak_status_assertions(path, lines)

        assert [(ln, code) for ln, code, _sev, _msg in violations] == [(3, "VET244")]

    def test_broad_success_sets_fire_vet245(self, tmp_path: Path) -> None:
        """A synthetic {200, 201} success set must be flagged."""
        mod = _import_script("check_test_quality")
        path = tmp_path / "test_status_broad.py"
        path.write_text(
            'def test_accepts_multiple_successes():\n'
            '    response = object()\n'
            '    assert response.status_code in {200, 201}\n',
            encoding="utf-8",
        )
        lines = path.read_text(encoding="utf-8").splitlines()

        violations = mod.check_weak_status_assertions(path, lines)

        assert [(ln, code) for ln, code, _sev, _msg in violations] == [(3, "VET245")]

    def test_shape_only_and_broad_success_fire(self, tmp_path: Path) -> None:
        """Synthetic protocol checks must flag broad success and response-shape-only assertions."""
        mod = _import_script("check_test_quality")
        path = tmp_path / "test_protocol_shape_only.py"
        path.write_text(
            'def test_protocol_shape_only():\n'
            '    response = object()\n'
            '    body = response.json()\n'
            '    assert response.status_code in {200, 201}\n'
            '    assert "error" in body\n',
            encoding="utf-8",
        )
        lines = path.read_text(encoding="utf-8").splitlines()

        status_violations = mod.check_weak_status_assertions(path, lines)
        shape_violations = mod.check_response_shape_only_assertions(path, lines)

        assert [(ln, code) for ln, code, _sev, _msg in status_violations] == [(4, "VET245")]
        assert [(ln, code) for ln, code, _sev, _msg in shape_violations] == [(5, "VET246")]

    def test_module_level_importorskip_fires_vet247(self, tmp_path: Path) -> None:
        """Synthetic module-level importorskip is visible as an explicit suite-skip risk."""
        mod = _import_script("check_test_quality")
        path = tmp_path / "test_importorskip.py"
        path.write_text(
            'import pytest\n'
            'litestar = pytest.importorskip("litestar")\n'
            'def test_smoke():\n'
            '    assert litestar is not None\n',
            encoding="utf-8",
        )
        lines = path.read_text(encoding="utf-8").splitlines()

        violations = mod.check_module_importorskip(path, lines)

        assert [(ln, code) for ln, code, _sev, _msg in violations] == [(2, "VET247")]


class TestDefect6ConfigWiringComments:
    """check_config_wiring: keys mentioned only in comments must not pass as wired."""

    def test_comment_only_mention_is_not_wired(self) -> None:
        """A key appearing only in a comment must be reported as unreferenced.

        Old bug: the corpus was built from raw source including comment lines,
        so a config key that appeared only in a comment was (incorrectly) treated
        as live wiring.
        """
        mod = _import_script("check_config_wiring")

        # Build a corpus that only mentions the key in a comment
        corpus = mod._strip_comments_and_strings(
            '# use my_key for configuration\n'
            'value = something_else\n'
        )
        assert not mod._key_is_referenced("my_key", corpus), (
            "_key_is_referenced() returned True for a key that only appears in a comment"
        )

    def test_docstring_only_mention_is_not_wired(self) -> None:
        """A key appearing only in a docstring must be reported as unreferenced."""
        mod = _import_script("check_config_wiring")

        corpus = mod._strip_comments_and_strings(
            '"""Load my_key from config."""\n'
            'value = other\n'
        )
        assert not mod._key_is_referenced("my_key", corpus), (
            "_key_is_referenced() returned True for a key only in a docstring"
        )

    def test_live_dict_access_is_wired(self) -> None:
        """A key in a real dict access must be recognised as live wiring."""
        mod = _import_script("check_config_wiring")

        corpus = mod._strip_comments_and_strings('value = config["my_key"]\n')
        assert mod._key_is_referenced("my_key", corpus), (
            '_key_is_referenced() returned False for config["my_key"]'
        )


# ---------------------------------------------------------------------------
# Defect 7 — config wiring: bare substring match too loose
# ---------------------------------------------------------------------------


class TestDefect7ConfigWiringBareSubstring:
    """check_config_wiring: bare substring occurrences must not count as live wiring."""

    def test_substring_in_variable_name_is_not_wired(self) -> None:
        """A key that appears only as part of a longer variable name is not wired.

        Old bug: old ``re.search(r'\\b' + key + r'\\b', corpus)`` matched
        word-boundary occurrences including variable names that merely contained
        the key as a substring.  The stricter patterns (dict access, .get(),
        attribute access) must NOT match bare names.
        """
        mod = _import_script("check_config_wiring")

        # "port" only appears as part of the longer name "report_port"
        corpus = mod._strip_comments_and_strings('report_port = 8080\n')
        assert not mod._key_is_referenced("port", corpus), (
            "_key_is_referenced() returned True for 'port' that only appears "
            "as a substring of 'report_port'"
        )

    def test_get_call_is_wired(self) -> None:
        """config.get('port') must register as live wiring."""
        mod = _import_script("check_config_wiring")

        corpus = mod._strip_comments_and_strings("p = config.get('port')\n")
        assert mod._key_is_referenced("port", corpus), (
            "_key_is_referenced() returned False for config.get('port')"
        )


# ---------------------------------------------------------------------------
# Defect 8 — singleton inner null-check verification (VET200)
# ---------------------------------------------------------------------------


class TestDefect8SingletonInnerNullCheck:
    """check_wiring_audit: singletons missing the inner null-check must fire VET200."""

    def test_missing_inner_null_check_fires_vet200(self, tmp_path: Path) -> None:
        """get_X() with lock but only one null-check must fire VET200.

        Old bug: the checker only verified presence of a lock; it did not
        verify that both the outer AND inner ``if _x is None:`` checks existed,
        so the single-check form was silently accepted.
        """
        mod = _import_script("check_wiring_audit")

        fake = tmp_path / "singleton_one_check.py"
        fake.write_text(
            'import threading\n'
            '_inst = None\n'
            '_lock = threading.Lock()\n'
            'def get_service():\n'
            '    global _inst\n'
            '    if _inst is None:\n'
            '        with _lock:\n'
            '            _inst = Service()\n'  # missing inner check
            '    return _inst\n',
            encoding="utf-8",
        )
        lines = fake.read_text(encoding="utf-8").splitlines()
        violations = mod.check_singleton_locking(fake, lines)

        assert violations, (
            "check_singleton_locking() should report VET200 when the inner "
            "'if _inst is None:' check inside the lock is missing"
        )
        codes = [code for _ln, code, _sev, _msg in violations]
        assert "VET200" in codes, f"Expected VET200; got: {codes}"

    def test_full_double_checked_locking_is_clean(self, tmp_path: Path) -> None:
        """A correct double-checked locking pattern must NOT fire VET200."""
        mod = _import_script("check_wiring_audit")

        fake = tmp_path / "singleton_correct.py"
        fake.write_text(
            'import threading\n'
            '_inst = None\n'
            '_lock = threading.Lock()\n'
            'def get_service():\n'
            '    global _inst\n'
            '    if _inst is None:\n'
            '        with _lock:\n'
            '            if _inst is None:\n'
            '                _inst = Service()\n'
            '    return _inst\n',
            encoding="utf-8",
        )
        lines = fake.read_text(encoding="utf-8").splitlines()
        violations = mod.check_singleton_locking(fake, lines)

        assert not violations, (
            f"check_singleton_locking() incorrectly flagged a correct "
            f"double-checked locking pattern: {violations}"
        )


# ---------------------------------------------------------------------------
# Defect 9 — multiline event calls (VET202/VET203)
# ---------------------------------------------------------------------------


class TestDefect9MultilineEventCalls:
    """check_wiring_audit: publish/subscribe spanning multiple lines must be detected."""

    def test_multiline_publish_is_detected(self, tmp_path: Path) -> None:
        """A .publish() call split across lines must be found as a publisher.

        Old bug: only single-line calls were scanned, so a publish call like::

            bus.publish(
                "my.event",
                payload,
            )

        was invisible to the checker and produced a false VET203 (subscribed but
        never published).
        """
        mod = _import_script("check_wiring_audit")

        pub_file = tmp_path / "pub.py"
        pub_file.write_text(
            'bus.publish(\n'
            '    "my.event",\n'
            '    payload,\n'
            ')\n',
            encoding="utf-8",
        )
        sub_file = tmp_path / "sub.py"
        sub_file.write_text(
            'bus.subscribe("my.event", handler)\n',
            encoding="utf-8",
        )

        violations = mod.check_event_bus_pairing([pub_file, sub_file])

        # Both pub and sub exist — no violation expected
        assert not violations, (
            f"check_event_bus_pairing() reported violations for a matched "
            f"publish/subscribe pair where publish spans multiple lines: {violations}"
        )

    def test_multiline_unpaired_publish_fires_vet202(self, tmp_path: Path) -> None:
        """An unpaired multiline publish must fire VET202."""
        mod = _import_script("check_wiring_audit")

        pub_file = tmp_path / "pub_only.py"
        pub_file.write_text(
            'bus.publish(\n'
            '    "orphan.event",\n'
            ')\n',
            encoding="utf-8",
        )

        violations = mod.check_event_bus_pairing([pub_file])

        codes = [code for _fp, _ln, code, _sev, _msg in violations]
        assert "VET202" in codes, (
            f"Expected VET202 for unpaired multiline publish; got: {codes}"
        )


# ---------------------------------------------------------------------------
# Defect 10 — comment/string event names counted as publishers (VET202/VET203)
# ---------------------------------------------------------------------------


class TestDefect10CommentEventNames:
    """check_wiring_audit: event names in comments must not be counted as publishers."""

    def test_commented_publish_not_counted(self, tmp_path: Path) -> None:
        """A .publish() call in a comment must not register as a publisher.

        Old bug: the event-name extractor scanned all lines including comments,
        so ``# bus.publish("stale.event", ...)`` was treated as a live publisher,
        masking a missing-subscriber violation.
        """
        mod = _import_script("check_wiring_audit")

        comment_only = tmp_path / "commented_pub.py"
        comment_only.write_text(
            '# bus.publish("comment.event", payload)\n'
            'pass\n',
            encoding="utf-8",
        )
        sub_file = tmp_path / "sub_only.py"
        sub_file.write_text(
            'bus.subscribe("comment.event", handler)\n',
            encoding="utf-8",
        )

        violations = mod.check_event_bus_pairing([comment_only, sub_file])

        # The sub has no matching publisher (comment doesn't count) → VET203 expected
        codes = [code for _fp, _ln, code, _sev, _msg in violations]
        assert "VET203" in codes, (
            "Expected VET203 (subscribed but never published) because the "
            "only 'publish' is in a comment — comment should not count"
        )

    def test_docstring_subscribe_not_counted(self, tmp_path: Path) -> None:
        """A .subscribe() call in a string literal must not register as a subscriber."""
        mod = _import_script("check_wiring_audit")

        pub_file = tmp_path / "pub_only.py"
        pub_file.write_text(
            'bus.publish("doc.event", data)\n',
            encoding="utf-8",
        )
        docstring_sub = tmp_path / "docstring_sub.py"
        docstring_sub.write_text(
            '"""bus.subscribe("doc.event", handler) -- example"""\n'
            'pass\n',
            encoding="utf-8",
        )

        violations = mod.check_event_bus_pairing([pub_file, docstring_sub])

        # Publisher exists but subscriber is only in a docstring → VET202 expected
        codes = [code for _fp, _ln, code, _sev, _msg in violations]
        assert "VET202" in codes, (
            "Expected VET202 (published but no subscriber) because the "
            "only 'subscribe' is in a docstring — should not count as live"
        )


# ---------------------------------------------------------------------------
# Defect 11 — handoff_bundle.py portable bootstrap command
# ---------------------------------------------------------------------------


class TestDefect11HandoffBundlePortablePath:
    """handoff_bundle.py + check_handoff_bundle.py: bootstrap command must be portable."""

    def test_handoff_bundle_bootstrap_command_is_portable(self) -> None:
        """render_markdown() must not embed a hardcoded absolute interpreter path.

        Old bug: the bootstrap helper line contained the full Windows venv path
        ``C:/dev/Vetinari/.venv312/Scripts/python.exe``, which breaks on any
        other machine.
        """
        script_path = _SCRIPTS_DIR / "handoff_bundle.py"
        source = script_path.read_text(encoding="utf-8")

        assert "C:/dev/Vetinari/.venv312" not in source, (
            "handoff_bundle.py still contains the hardcoded absolute venv path "
            "C:/dev/Vetinari/.venv312 — use portable `python scripts/...` instead"
        )
        assert "subagent_bootstrap.py" in source, (
            "handoff_bundle.py must still reference subagent_bootstrap.py "
            "after the path was made portable"
        )

    def test_check_handoff_bundle_detects_absolute_path(self, tmp_path: Path) -> None:
        """validate_markdown_content() must flag a bootstrap line with an absolute path."""
        mod = _import_script("check_handoff_bundle")

        bad_md = tmp_path / "current.md"
        bad_md.write_text(
            "## Child-Agent Bootstrap\n\n"
            "- Bootstrap helper: `C:/dev/Vetinari/.venv312/Scripts/python.exe "
            "scripts/subagent_bootstrap.py builder --json`\n",
            encoding="utf-8",
        )

        errors = mod.validate_markdown_content(bad_md)
        assert errors, (
            "validate_markdown_content() returned no errors for a bootstrap "
            "command with a hardcoded absolute path"
        )
        assert any("absolute" in e or "hardcoded" in e for e in errors), (
            f"Expected error mentioning 'absolute' or 'hardcoded'; got: {errors}"
        )

    def test_check_handoff_bundle_accepts_portable_path(self, tmp_path: Path) -> None:
        """validate_markdown_content() must accept a portable bootstrap command."""
        mod = _import_script("check_handoff_bundle")

        good_md = tmp_path / "current.md"
        good_md.write_text(
            "## Child-Agent Bootstrap\n\n"
            "- Bootstrap helper: `python scripts/subagent_bootstrap.py builder --json`\n",
            encoding="utf-8",
        )

        errors = mod.validate_markdown_content(good_md)
        assert not errors, (
            f"validate_markdown_content() flagged a portable bootstrap command: {errors}"
        )

    def test_check_handoff_bundle_unix_absolute_path_flagged(self, tmp_path: Path) -> None:
        """validate_markdown_content() must also flag Unix absolute paths."""
        mod = _import_script("check_handoff_bundle")

        unix_md = tmp_path / "current.md"
        unix_md.write_text(
            "## Child-Agent Bootstrap\n\n"
            "- Bootstrap helper: `/home/user/.venv/bin/python "
            "scripts/subagent_bootstrap.py builder --json`\n",
            encoding="utf-8",
        )

        errors = mod.validate_markdown_content(unix_md)
        assert errors, (
            "validate_markdown_content() did not flag a Unix absolute interpreter path"
        )
