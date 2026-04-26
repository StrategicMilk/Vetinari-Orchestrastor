"""Tests for vetinari.ml.classifiers — GoalClassifier, DefectClassifier, AmbiguityDetector."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from vetinari.ml.classifiers import (
    AmbiguityDetector,
    AmbiguityResult,
    DefectClassification,
    DefectClassifier,
    GoalClassification,
    GoalClassifier,
)
from vetinari.types import GoalCategory


class TestGoalClassifier:
    """Tests for GoalClassifier aligned to GoalCategory enum."""

    @pytest.fixture
    def classifier(self) -> GoalClassifier:
        """Provide a GoalClassifier instance."""
        return GoalClassifier()

    def test_classify_code_goal(self, classifier: GoalClassifier) -> None:
        """Code keywords route to GoalCategory.CODE."""
        result = classifier.classify("implement a REST API endpoint")
        assert result.category == GoalCategory.CODE.value
        assert result.confidence > 0.0
        assert len(result.keyword_matches) > 0

    def test_classify_research_goal(self, classifier: GoalClassifier) -> None:
        """Research keywords route to GoalCategory.RESEARCH."""
        result = classifier.classify("research the best options and analyze the situation")
        assert result.category == GoalCategory.RESEARCH.value
        assert result.confidence > 0.0

    def test_classify_docs_goal(self, classifier: GoalClassifier) -> None:
        """Documentation keywords route to GoalCategory.DOCS."""
        result = classifier.classify("write docstring and readme for the module")
        assert result.category == GoalCategory.DOCS.value

    def test_classify_security_goal(self, classifier: GoalClassifier) -> None:
        """Security keywords route to GoalCategory.SECURITY."""
        result = classifier.classify("audit security vulnerabilities and check owasp compliance")
        assert result.category == GoalCategory.SECURITY.value

    def test_classify_data_goal(self, classifier: GoalClassifier) -> None:
        """Data keywords route to GoalCategory.DATA."""
        result = classifier.classify("create a database schema migration for the table")
        assert result.category == GoalCategory.DATA.value

    def test_classify_devops_goal(self, classifier: GoalClassifier) -> None:
        """DevOps keywords route to GoalCategory.DEVOPS."""
        result = classifier.classify("deploy the docker container and set up pipeline")
        assert result.category == GoalCategory.DEVOPS.value

    def test_classify_ui_goal(self, classifier: GoalClassifier) -> None:
        """UI keywords route to GoalCategory.UI."""
        result = classifier.classify("design a new frontend component layout")
        assert result.category == GoalCategory.UI.value

    def test_classify_image_goal(self, classifier: GoalClassifier) -> None:
        """Image keywords route to GoalCategory.IMAGE."""
        result = classifier.classify("generate a logo icon for the project")
        assert result.category == GoalCategory.IMAGE.value

    def test_classify_empty_string_returns_general(self, classifier: GoalClassifier) -> None:
        """Empty input returns GoalCategory.GENERAL with mid confidence."""
        result = classifier.classify("")
        assert result.category == GoalCategory.GENERAL.value
        assert result.confidence == 0.5

    def test_classify_whitespace_only_returns_general(self, classifier: GoalClassifier) -> None:
        """Whitespace-only input returns 'general'."""
        result = classifier.classify("   ")
        assert result.category == GoalCategory.GENERAL.value

    def test_classify_unknown_text_returns_general(self, classifier: GoalClassifier) -> None:
        """Text with no keyword matches returns 'general' with low confidence."""
        result = classifier.classify("xyz zyx zzz nonsense words here")
        assert result.category == GoalCategory.GENERAL.value
        assert result.confidence == 0.3

    def test_all_categories_have_keyword_coverage(self, classifier: GoalClassifier) -> None:
        """Every non-general category can be triggered by at least one keyword."""
        category_probes = {
            GoalCategory.CODE.value: "implement a function and develop code",
            GoalCategory.RESEARCH.value: "research the options and analyze",
            GoalCategory.DOCS.value: "document this module with readme",
            GoalCategory.SECURITY.value: "audit security vulnerabilities",
            GoalCategory.DATA.value: "database schema migration sql",
            GoalCategory.DEVOPS.value: "deploy docker kubernetes pipeline",
            GoalCategory.UI.value: "ui ux frontend design wireframe",
            GoalCategory.IMAGE.value: "generate logo icon mockup diagram",
        }
        for expected_category, probe in category_probes.items():
            result = classifier.classify(probe)
            assert result.category == expected_category, (
                f"Expected '{expected_category}' for probe '{probe}', got '{result.category}'"
            )

    def test_keyword_matches_present_in_result(self, classifier: GoalClassifier) -> None:
        """Matched keywords appear in the result."""
        result = classifier.classify("implement a class and module")
        assert "implement" in result.keyword_matches or "class" in result.keyword_matches

    def test_confidence_between_zero_and_one(self, classifier: GoalClassifier) -> None:
        """Confidence is always within [0.0, 1.0]."""
        texts = [
            "implement a REST API",
            "debug the login bug",
            "research options",
            "",
            "xyz abc",
        ]
        for text in texts:
            result = classifier.classify(text)
            assert 0.0 <= result.confidence <= 1.0, f"Confidence out of range for '{text}': {result.confidence}"


class TestGoalClassification:
    """Tests for GoalClassification dataclass."""

    def test_to_dict_keys(self) -> None:
        """to_dict() returns the expected keys."""
        gc = GoalClassification(category=GoalCategory.CODE.value, confidence=0.75, keyword_matches=["implement"])
        d = gc.to_dict()
        assert set(d.keys()) == {"category", "confidence", "keyword_matches"}

    def test_to_dict_values(self) -> None:
        """to_dict() serializes values correctly."""
        gc = GoalClassification(category=GoalCategory.CODE.value, confidence=0.555555, keyword_matches=["test"])
        d = gc.to_dict()
        assert d["category"] == "code"
        assert d["confidence"] == 0.556  # rounded to 3 decimal places
        assert d["keyword_matches"] == ["test"]

    def test_to_dict_empty_matches(self) -> None:
        """to_dict() handles empty keyword_matches list."""
        gc = GoalClassification(category=GoalCategory.GENERAL.value, confidence=0.3)
        d = gc.to_dict()
        assert d["keyword_matches"] == []


class TestDefectClassifier:
    """Tests for DefectClassifier."""

    @pytest.fixture
    def classifier(self) -> DefectClassifier:
        """Provide a DefectClassifier instance."""
        return DefectClassifier()

    def test_classify_hallucinated_import(self, classifier: DefectClassifier) -> None:
        """Import-related errors classify as hallucinated_import."""
        result = classifier.classify("ModuleNotFoundError: No module named 'requests'")
        assert result.category == "hallucinated_import"
        assert result.confidence > 0.0

    def test_classify_ambiguous_spec(self, classifier: DefectClassifier) -> None:
        """Vague/unclear rejection reasons classify as ambiguous_spec."""
        result = classifier.classify("The requirements are unclear and ambiguous")
        assert result.category == "ambiguous_spec"
        assert result.confidence > 0.0

    def test_classify_style_violation(self, classifier: DefectClassifier) -> None:
        """Style-related rejections classify as style_violation."""
        result = classifier.classify("ruff lint error: missing type hint and docstring")
        assert result.category == "style_violation"

    def test_classify_logic_error(self, classifier: DefectClassifier) -> None:
        """Logic errors classify correctly."""
        result = classifier.classify("logic error causing incorrect output and wrong result")
        assert result.category == "logic_error"

    def test_classify_integration_error(self, classifier: DefectClassifier) -> None:
        """Integration errors classify correctly."""
        result = classifier.classify("api mismatch: version conflict in interface signature")
        assert result.category == "integration_error"

    def test_classify_empty_text_returns_default(self, classifier: DefectClassifier) -> None:
        """Empty rejection text returns default category with low confidence."""
        result = classifier.classify("")
        assert result.category == "logic_error"
        assert result.confidence == 0.3

    def test_classify_no_keyword_match_returns_default(self, classifier: DefectClassifier) -> None:
        """Text with no keyword matches returns default with lowest confidence."""
        result = classifier.classify("xyz zyx zzz random noise")
        assert result.category == "logic_error"
        assert result.confidence == 0.2

    def test_classify_uses_code_diff(self, classifier: DefectClassifier) -> None:
        """Code diff context influences classification."""
        result = classifier.classify(
            "failed quality gate",
            code_diff="import nonexistent_module\nfrom fake_pkg import thing",
        )
        assert result.category == "hallucinated_import"

    def test_confidence_between_zero_and_one(self, classifier: DefectClassifier) -> None:
        """Confidence is always within [0.0, 1.0]."""
        texts = [
            "module not found importerror",
            "unclear ambiguous spec",
            "logic error wrong result",
            "",
        ]
        for text in texts:
            result = classifier.classify(text)
            assert 0.0 <= result.confidence <= 1.0, f"Confidence out of range for '{text}': {result.confidence}"

    def test_evidence_present_when_matched(self, classifier: DefectClassifier) -> None:
        """Evidence list is non-empty when keywords matched."""
        result = classifier.classify("ImportError: cannot import name xyz")
        assert len(result.evidence) > 0


class TestDefectClassification:
    """Tests for DefectClassification dataclass."""

    def test_to_dict_keys(self) -> None:
        """to_dict() returns expected keys."""
        dc = DefectClassification(category="hallucinated_import", confidence=0.8, evidence=["import"])
        d = dc.to_dict()
        assert set(d.keys()) == {"category", "confidence", "evidence"}

    def test_to_dict_rounds_confidence(self) -> None:
        """to_dict() rounds confidence to 3 decimal places."""
        dc = DefectClassification(category="logic_error", confidence=0.123456)
        d = dc.to_dict()
        assert d["confidence"] == 0.123

    def test_to_dict_empty_evidence(self) -> None:
        """to_dict() handles empty evidence list."""
        dc = DefectClassification(category="logic_error", confidence=0.2)
        d = dc.to_dict()
        assert d["evidence"] == []


class TestAmbiguityDetector:
    """Tests for AmbiguityDetector."""

    @pytest.fixture
    def detector(self) -> AmbiguityDetector:
        """Provide an AmbiguityDetector instance."""
        return AmbiguityDetector()

    def test_specific_request_not_ambiguous(self, detector: AmbiguityDetector) -> None:
        """Specific request with file reference is not ambiguous."""
        result = detector.detect("Add logging to vetinari/cli.py at line 45")
        assert result.is_ambiguous is False

    def test_vague_request_is_ambiguous(self, detector: AmbiguityDetector) -> None:
        """Vague request with hedge words is ambiguous."""
        result = detector.detect("maybe do something with the stuff somehow")
        assert result.is_ambiguous is True

    def test_empty_request_is_ambiguous(self, detector: AmbiguityDetector) -> None:
        """Empty request is ambiguous with high confidence."""
        result = detector.detect("")
        assert result.is_ambiguous is True
        assert result.confidence >= 0.8

    def test_whitespace_only_is_ambiguous(self, detector: AmbiguityDetector) -> None:
        """Whitespace-only request is ambiguous."""
        result = detector.detect("   ")
        assert result.is_ambiguous is True

    def test_file_references_reduce_ambiguity(self, detector: AmbiguityDetector) -> None:
        """File references lower the ambiguity score."""
        vague = detector.detect("do something maybe")
        specific = detector.detect("Update the timeout value in vetinari/config.yaml and vetinari/settings.py")
        vague_score = vague.features.get("specificity", 0.0)
        specific_score = specific.features.get("specificity", 0.0)
        assert specific_score > vague_score

    def test_features_dict_has_expected_keys(self, detector: AmbiguityDetector) -> None:
        """Features dict includes all expected keys for non-empty input."""
        result = detector.detect("Add logging to cli.py")
        expected_keys = {
            "hedge_count",
            "hedge_density",
            "question_marks",
            "conditional_count",
            "file_refs",
            "func_refs",
            "specificity",
            "word_count",
            "length_score",
        }
        assert expected_keys.issubset(result.features.keys())

    def test_confidence_between_zero_and_one(self, detector: AmbiguityDetector) -> None:
        """Confidence is always within [0.0, 1.0]."""
        texts = [
            "maybe something somewhere",
            "Add logging to vetinari/cli.py",
            "",
            "fix the bug in parse_token() function",
        ]
        for text in texts:
            result = detector.detect(text)
            assert 0.0 <= result.confidence <= 1.0, f"Confidence out of range for '{text}': {result.confidence}"

    def test_hedge_words_increase_ambiguity(self, detector: AmbiguityDetector) -> None:
        """Hedge words push ambiguity score up."""
        without_hedge = detector.detect("Add error handling to the login function")
        with_hedge = detector.detect("Maybe possibly add something kind of like error handling if possible somehow")
        assert with_hedge.features["hedge_count"] > without_hedge.features["hedge_count"]

    def test_question_marks_counted(self, detector: AmbiguityDetector) -> None:
        """Question marks are captured in features."""
        result = detector.detect("Should I add logging? Or maybe not?")
        assert result.features["question_marks"] == 2

    def test_word_count_in_features(self, detector: AmbiguityDetector) -> None:
        """Word count is accurate in features."""
        result = detector.detect("fix the bug")
        assert result.features["word_count"] == 3

    # ── Item 16.8 new feature tests ───────────────────────────────────────────

    def test_features_include_vague_pronoun_fields(self, detector: AmbiguityDetector) -> None:
        """Feature dict now includes vague_pronoun_count, vague_pronoun_density, has_missing_subject."""
        result = detector.detect("Fix it and update this thing")
        for key in ("vague_pronoun_count", "vague_pronoun_density", "has_missing_subject"):
            assert key in result.features, f"Missing new feature key: {key!r}"

    def test_vague_pronouns_detected(self, detector: AmbiguityDetector) -> None:
        """Requests using 'it', 'this', 'that' accumulate a non-zero vague pronoun count."""
        result = detector.detect("Fix it and make this work so that they are happy")
        assert result.features["vague_pronoun_count"] > 0
        assert result.features["vague_pronoun_density"] > 0.0

    def test_no_vague_pronouns_in_specific_request(self, detector: AmbiguityDetector) -> None:
        """A request with concrete nouns has zero vague pronoun count."""
        result = detector.detect("Add logging to vetinari/cli.py in the parse_args() function")
        # Pronouns like 'it/this/that' should not appear in this text
        assert result.features["vague_pronoun_count"] == 0

    def test_missing_subject_detected(self, detector: AmbiguityDetector) -> None:
        """'fix it' pattern triggers has_missing_subject=True."""
        result = detector.detect("fix it")
        assert result.features["has_missing_subject"] is True

    def test_no_missing_subject_for_concrete_request(self, detector: AmbiguityDetector) -> None:
        """A request with an explicit object does not trigger missing subject."""
        result = detector.detect("Implement a binary search function in vetinari/search.py")
        assert result.features["has_missing_subject"] is False

    def test_borderline_threshold_tightened(self, detector: AmbiguityDetector) -> None:
        """Borderline LLM assist fires at 0.25, not 0.4 — verify via import."""
        import inspect

        from vetinari.ml import classifiers as _cls_module

        source = inspect.getsource(_cls_module.AmbiguityDetector.detect)
        assert "0.25" in source, "Borderline threshold must be 0.25 (Item 16.8)"
        assert "0.4" not in source, "Old threshold 0.4 must be replaced"

    def test_vague_request_with_pronouns_is_more_ambiguous(self, detector: AmbiguityDetector) -> None:
        """Adding vague pronouns to a request increases the ambiguity score."""
        clear = detector.detect("Update the timeout in the configuration file")
        vague = detector.detect("Update it in this thing")
        # The vague version should have more vague pronouns
        assert vague.features["vague_pronoun_count"] >= clear.features["vague_pronoun_count"]


class TestAmbiguityThresholdBehavior:
    """Behavioral tests for the 0.25 borderline LLM-assist threshold in AmbiguityDetector."""

    @pytest.fixture
    def detector(self) -> AmbiguityDetector:
        """Provide an AmbiguityDetector instance."""
        return AmbiguityDetector()

    def test_score_above_threshold_triggers_llm(self, detector: AmbiguityDetector) -> None:
        """A request that scores above 0.25 triggers the LLM assist call.

        The description uses multiple hedge words ('maybe', 'possibly', 'somehow',
        'something', 'kind of') and vague pronouns ('it', 'this', 'that') to push
        the ambiguity score well above the 0.25 borderline threshold. We mock
        check_ambiguity_via_llm at its import location inside the classifiers module
        and assert it was called.
        """
        # hedge_density is the dominant signal: hedge_density * 3.0
        # "maybe possibly somehow something kind of" = 4 hedge words in ~8 total words
        # hedge_density ~ 4/8 = 0.5 → ambiguity contribution = 0.5 * 3.0 = 1.5
        # vague pronouns 'it' 'this' 'that' add further — score will be >> 0.25
        vague_text = "maybe possibly do something with it somehow so that this works"

        # check_ambiguity_via_llm is imported lazily inside AmbiguityDetector.detect().
        # Patch at the source module so the late import picks up the mock.
        with patch(
            "vetinari.llm_helpers.check_ambiguity_via_llm",
            return_value=(True, "What should 'it' refer to?"),
        ) as mock_llm:
            result = detector.detect(vague_text)

        (
            mock_llm.assert_called_once_with(vague_text),
            ("check_ambiguity_via_llm must be called when ambiguity score > 0.25"),
        )
        # LLM returned True, so the result must reflect that
        assert result.is_ambiguous is True
        assert "llm_override" in result.features

    def test_score_below_threshold_does_not_trigger_llm(self, detector: AmbiguityDetector) -> None:
        """A clear, specific request that scores below 0.25 does NOT trigger LLM assist.

        The description is concrete (file ref, function name, class name) with no
        hedge words, question marks, or conditionals. Specificity pushes the score
        negative (specificity * 2.0 subtracted), keeping it below 0.25.
        """
        # File reference + function reference + class reference = high specificity
        # specificity = (file_refs + func_refs + class_refs * 0.5) / word_count
        # With no hedge words, conditional phrases, or question marks, the score
        # will be deeply negative (clamped to 0.0) — well below 0.25.
        clear_text = "Add error handling to parse_token() in vetinari/auth.py for the TokenValidator class"

        with patch(
            "vetinari.llm_helpers.check_ambiguity_via_llm",
        ) as mock_llm:
            result = detector.detect(clear_text)

        mock_llm.assert_not_called(), ("check_ambiguity_via_llm must NOT be called when ambiguity score <= 0.25")
        assert result.is_ambiguous is False


class TestAmbiguityResult:
    """Tests for AmbiguityResult dataclass."""

    def test_to_dict_keys(self) -> None:
        """to_dict() returns expected keys."""
        ar = AmbiguityResult(is_ambiguous=True, confidence=0.8, features={"x": 1})
        d = ar.to_dict()
        assert set(d.keys()) == {"is_ambiguous", "confidence", "features"}

    def test_to_dict_rounds_confidence(self) -> None:
        """to_dict() rounds confidence to 3 decimal places."""
        ar = AmbiguityResult(is_ambiguous=False, confidence=0.666666)
        d = ar.to_dict()
        assert d["confidence"] == 0.667

    def test_to_dict_preserves_is_ambiguous(self) -> None:
        """to_dict() preserves boolean is_ambiguous value."""
        ar_true = AmbiguityResult(is_ambiguous=True, confidence=0.9)
        ar_false = AmbiguityResult(is_ambiguous=False, confidence=0.1)
        assert ar_true.to_dict()["is_ambiguous"] is True
        assert ar_false.to_dict()["is_ambiguous"] is False

    def test_to_dict_empty_features(self) -> None:
        """to_dict() handles empty features dict."""
        ar = AmbiguityResult(is_ambiguous=False, confidence=0.0)
        d = ar.to_dict()
        assert d["features"] == {}


class TestClassifierImports:
    """Smoke tests for module import paths."""

    def test_import_from_classifiers_module(self) -> None:
        """Direct imports from vetinari.ml.classifiers work."""
        from vetinari.ml.classifiers import (
            AmbiguityDetector,
            AmbiguityResult,
            DefectClassification,
            DefectClassifier,
            GoalClassification,
            GoalClassifier,
        )

        assert callable(AmbiguityDetector)
        assert callable(AmbiguityResult)
        assert callable(DefectClassification)
        assert callable(DefectClassifier)
        assert callable(GoalClassification)
        assert callable(GoalClassifier)

    def test_import_from_ml_package(self) -> None:
        """Re-exports from vetinari.ml package work."""
        from vetinari.ml import (
            AmbiguityDetector,
            AmbiguityResult,
            DefectClassification,
            DefectClassifier,
            GoalClassification,
            GoalClassifier,
        )

        assert callable(AmbiguityDetector)
        assert callable(GoalClassifier)

    def test_vetinari_import_still_works(self) -> None:
        """Top-level vetinari import is unaffected."""
        import vetinari

        assert hasattr(vetinari, "__version__")
