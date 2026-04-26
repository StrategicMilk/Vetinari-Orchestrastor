"""Tests for vetinari.orchestration.intake.

Covers Tier enum, IntakeFeatures, RequestIntake classification,
feature extraction, Thompson override, and singleton management.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from vetinari.orchestration.intake import (
    IntakeFeatures,
    RequestIntake,
    Tier,
    get_request_intake,
    reset_request_intake,
)

# ── Tier Enum ──────────────────────────────────────────────────────────


class TestTier:
    """Test Tier enum values."""

    def test_express_value(self) -> None:
        assert Tier.EXPRESS.value == "express"

    def test_standard_value(self) -> None:
        assert Tier.STANDARD.value == "standard"

    def test_custom_value(self) -> None:
        assert Tier.CUSTOM.value == "custom"

    def test_tier_from_string(self) -> None:
        assert Tier("express") is Tier.EXPRESS
        assert Tier("standard") is Tier.STANDARD
        assert Tier("custom") is Tier.CUSTOM


# ── IntakeFeatures ─────────────────────────────────────────────────────


class TestIntakeFeatures:
    """Test IntakeFeatures dataclass defaults."""

    def test_defaults(self) -> None:
        f = IntakeFeatures()
        assert f.word_count == 0
        assert f.file_count == 0
        assert f.has_ambiguous_words is False
        assert f.question_marks == 0
        assert f.cross_cutting_keywords == 0
        assert f.domain_novelty_score == 0.0
        assert f.pattern_key == ""
        assert f.confidence == 1.0


# ── Express Classification ─────────────────────────────────────────────


class TestExpressClassification:
    """Test Express tier classification (short, clear, single-file)."""

    def test_short_clear_goal(self) -> None:
        intake = RequestIntake()
        assert intake.classify("Fix typo in README") == Tier.EXPRESS

    def test_single_file_mention(self) -> None:
        intake = RequestIntake()
        assert intake.classify("Update config.yaml") == Tier.EXPRESS

    def test_very_short_goal(self) -> None:
        intake = RequestIntake()
        assert intake.classify("Add docstring") == Tier.EXPRESS

    def test_too_many_words_goes_standard(self) -> None:
        intake = RequestIntake()
        goal = "Add comprehensive error handling to the user authentication module and also ensure that all edge cases are covered with proper logging and unit tests for each branch"
        assert intake.classify(goal) != Tier.EXPRESS

    def test_ambiguous_words_prevent_express(self) -> None:
        intake = RequestIntake()
        assert intake.classify("Do something") != Tier.EXPRESS

    def test_question_marks_prevent_express(self) -> None:
        intake = RequestIntake()
        assert intake.classify("Is this right?") != Tier.EXPRESS


# ── Standard Classification ────────────────────────────────────────────


class TestStandardClassification:
    """Test Standard tier classification (moderate complexity)."""

    def test_multi_file_clear_goal(self) -> None:
        intake = RequestIntake()
        goal = "Add input validation to the user registration form in auth.py and views.py and update the corresponding API endpoint handler with proper error messages for each field"
        assert intake.classify(goal) == Tier.STANDARD

    def test_moderate_complexity(self) -> None:
        intake = RequestIntake()
        goal = "Implement pagination for the task list endpoint with offset and limit parameters including response headers for total count and next page URL"
        assert intake.classify(goal) == Tier.STANDARD

    def test_clear_but_longer_goal(self) -> None:
        intake = RequestIntake()
        goal = "Add a new REST endpoint for retrieving task history with date filtering and sorting options and include comprehensive unit tests for all query parameter combinations"
        assert intake.classify(goal) == Tier.STANDARD


# ── Custom Classification ──────────────────────────────────────────────


class TestCustomClassification:
    """Test Custom tier classification (ambiguous, cross-cutting, novel)."""

    def test_ambiguous_words(self) -> None:
        intake = RequestIntake()
        goal = "Maybe do something with the stuff and things somehow"
        assert intake.classify(goal) == Tier.CUSTOM

    def test_cross_cutting_concerns(self) -> None:
        intake = RequestIntake()
        goal = "Refactor the database schema, migrate the API, and update security across all files"
        assert intake.classify(goal) == Tier.CUSTOM

    def test_novel_domain(self) -> None:
        intake = RequestIntake()
        goal = "Implement machine learning neural transformer pipeline for real-time streaming predictions"
        assert intake.classify(goal) == Tier.CUSTOM

    def test_high_novelty_score(self) -> None:
        intake = RequestIntake()
        goal = "Build a kubernetes microservice with graphql and event-driven architecture"
        assert intake.classify(goal) == Tier.CUSTOM


# ── Feature Extraction ─────────────────────────────────────────────────


class TestFeatureExtraction:
    """Test _extract_features internals."""

    def test_word_count(self) -> None:
        intake = RequestIntake()
        features = intake._extract_features("one two three", {})
        assert features.word_count == 3

    def test_file_count_from_goal(self) -> None:
        intake = RequestIntake()
        features = intake._extract_features("Update config.yaml and main.py", {})
        assert features.file_count >= 2

    def test_file_count_from_context(self) -> None:
        intake = RequestIntake()
        features = intake._extract_features("Fix bug", {"file_count": 5})
        assert features.file_count >= 5

    def test_ambiguous_detection(self) -> None:
        intake = RequestIntake()
        features = intake._extract_features("Fix something in the code", {})
        assert features.has_ambiguous_words is True

    def test_no_ambiguous(self) -> None:
        intake = RequestIntake()
        features = intake._extract_features("Add docstring to parse_config", {})
        assert features.has_ambiguous_words is False

    def test_question_marks(self) -> None:
        intake = RequestIntake()
        features = intake._extract_features("What does this do? How does it work?", {})
        assert features.question_marks == 2

    def test_cross_cutting_count(self) -> None:
        intake = RequestIntake()
        features = intake._extract_features("Refactor the database and migrate the schema", {})
        assert features.cross_cutting_keywords >= 2

    def test_novelty_score_zero(self) -> None:
        intake = RequestIntake()
        features = intake._extract_features("Add a comment", {})
        assert features.domain_novelty_score == 0.0

    def test_novelty_score_high(self) -> None:
        intake = RequestIntake()
        features = intake._extract_features("Build machine learning neural transformer", {})
        assert features.domain_novelty_score >= 0.5

    def test_pattern_key_nonempty(self) -> None:
        intake = RequestIntake()
        features = intake._extract_features("Add error handling", {})
        assert len(features.pattern_key) == 16

    def test_pattern_key_deterministic(self) -> None:
        intake = RequestIntake()
        f1 = intake._extract_features("Add error handling", {})
        f2 = intake._extract_features("Add error handling", {})
        assert f1.pattern_key == f2.pattern_key

    def test_confidence_high_for_clear_goal(self) -> None:
        intake = RequestIntake()
        features = intake._extract_features("Add type hints to parse_config function", {})
        assert features.confidence >= 0.8

    def test_confidence_low_for_vague_goal(self) -> None:
        intake = RequestIntake()
        features = intake._extract_features("Fix it?", {})
        assert features.confidence < 0.7

    def test_confidence_clamped_to_zero(self) -> None:
        intake = RequestIntake()
        features = intake._extract_features("?? something??", {})
        assert features.confidence >= 0.0


# ── Thompson Override ──────────────────────────────────────────────────


class TestThompsonOverride:
    """Test Thompson Sampling tier override."""

    def test_no_thompson_uses_rules(self) -> None:
        intake = RequestIntake(thompson=None)
        assert intake.classify("Fix typo") == Tier.EXPRESS

    def test_thompson_insufficient_data(self) -> None:
        thompson = MagicMock()
        thompson.has_sufficient_data.return_value = False
        intake = RequestIntake(thompson=thompson)
        # Should fall through to rule-based
        assert intake.classify("Fix typo") == Tier.EXPRESS

    def test_thompson_override_to_standard(self) -> None:
        thompson = MagicMock()
        thompson.has_sufficient_data.return_value = True
        thompson.select_tier.return_value = "standard"
        intake = RequestIntake(thompson=thompson)
        # Rule-based would say EXPRESS, but Thompson overrides to STANDARD
        assert intake.classify("Fix typo") == Tier.STANDARD

    def test_thompson_override_to_custom(self) -> None:
        thompson = MagicMock()
        thompson.has_sufficient_data.return_value = True
        thompson.select_tier.return_value = "custom"
        intake = RequestIntake(thompson=thompson)
        assert intake.classify("Fix typo") == Tier.CUSTOM

    def test_thompson_exception_falls_back(self) -> None:
        thompson = MagicMock()
        thompson.has_sufficient_data.side_effect = RuntimeError("broken")
        intake = RequestIntake(thompson=thompson)
        # Should degrade gracefully to rule-based
        assert intake.classify("Fix typo") == Tier.EXPRESS


# ── classify_with_features ─────────────────────────────────────────────


class TestClassifyWithFeatures:
    """Test classify_with_features returns both tier and features."""

    def test_returns_tuple(self) -> None:
        intake = RequestIntake()
        tier, features = intake.classify_with_features("Fix typo")
        assert isinstance(tier, Tier)
        assert isinstance(features, IntakeFeatures)

    def test_features_match_goal(self) -> None:
        intake = RequestIntake()
        _, features = intake.classify_with_features("Update config.yaml")
        assert features.word_count == 2
        assert features.file_count >= 1


# ── Singleton ──────────────────────────────────────────────────────────


class TestSingleton:
    """Test singleton management."""

    def setup_method(self) -> None:
        reset_request_intake()

    def teardown_method(self) -> None:
        reset_request_intake()

    def test_returns_same_instance(self) -> None:
        i1 = get_request_intake()
        i2 = get_request_intake()
        assert i1 is i2

    def test_is_request_intake(self) -> None:
        assert isinstance(get_request_intake(), RequestIntake)

    def test_reset_creates_new_instance(self) -> None:
        i1 = get_request_intake()
        reset_request_intake()
        i2 = get_request_intake()
        assert i1 is not i2
