"""Tests for vetinari.validation.document_judge."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from vetinari.validation import DocumentJudge, JudgeConfig, QualityReport


class TestJudgeConfig:
    """Tests for the JudgeConfig dataclass."""

    def test_defaults(self):
        config = JudgeConfig()
        assert config.model_id == ""
        assert config.temperature == 0.1
        assert config.max_tokens == 1024
        assert config.fallback_to_heuristic is True
        assert "accuracy" in config.dimensions_for_llm

    def test_custom_config(self):
        config = JudgeConfig(
            model_id="qwen-32b",
            temperature=0.5,
            dimensions_for_llm=["accuracy", "clarity"],
        )
        assert config.model_id == "qwen-32b"
        assert config.temperature == 0.5
        assert len(config.dimensions_for_llm) == 2


class TestDocumentJudge:
    """Tests for the DocumentJudge class."""

    def test_default_judge_uses_heuristic(self):
        judge = DocumentJudge()
        assert judge.uses_llm is False

    def test_judge_with_model_uses_llm(self):
        config = JudgeConfig(model_id="test-model")
        judge = DocumentJudge(config)
        assert judge.uses_llm is True

    def test_evaluate_heuristic_only(self):
        judge = DocumentJudge()
        text = "# My Document\n\nThis is a well-written document about testing."
        report = judge.evaluate(text, doc_type="readme")
        assert isinstance(report, QualityReport)
        assert report.doc_type == "readme"
        assert 0.0 <= report.overall_score <= 1.0

    def test_evaluate_with_doc_type_enum(self):
        from vetinari.validation import DocumentType

        judge = DocumentJudge()
        text = "# ADR-001: Use PostgreSQL\n\n## Status\nAccepted\n"
        report = judge.evaluate(text, doc_type=DocumentType.ADR)
        assert report.doc_type == "adr"

    def test_evaluate_with_explicit_profile(self):
        from vetinari.validation import DocumentProfile

        judge = DocumentJudge()
        profile = DocumentProfile(
            doc_type="custom",
            min_score=0.50,
            dimension_weights={"accuracy": 1.0, "clarity": 1.0},
        )
        text = "A short document for testing profile override."
        report = judge.evaluate(text, doc_type="custom", profile=profile)
        assert isinstance(report, QualityReport)

    def test_get_config(self):
        config = JudgeConfig(model_id="test", temperature=0.3)
        judge = DocumentJudge(config)
        cfg = judge.get_config()
        assert cfg["model_id"] == "test"
        assert cfg["uses_llm"] is True
        assert cfg["temperature"] == 0.3
        assert isinstance(cfg["dimensions_for_llm"], list)
        assert cfg["fallback_to_heuristic"] is True

    def test_get_config_no_llm(self):
        judge = DocumentJudge()
        cfg = judge.get_config()
        assert cfg["model_id"] == ""
        assert cfg["uses_llm"] is False


class TestDocumentJudgeLLMPath:
    """Tests for the LLM evaluation path (mocked)."""

    def test_llm_failure_falls_back_to_heuristic(self):
        config = JudgeConfig(model_id="test-model", fallback_to_heuristic=True)
        judge = DocumentJudge(config)

        with patch.object(judge, "_llm_evaluate", side_effect=RuntimeError("no LLM")):
            text = "# Test Document\n\nContent for testing fallback behavior."
            report = judge.evaluate(text)
            # Should still return a valid report via heuristic fallback
            assert isinstance(report, QualityReport)
            assert report.overall_score > 0

    def test_llm_failure_raises_when_no_fallback(self):
        config = JudgeConfig(model_id="test-model", fallback_to_heuristic=False)
        judge = DocumentJudge(config)

        with patch.object(judge, "_llm_evaluate", side_effect=RuntimeError("no LLM")):
            import pytest

            with pytest.raises(RuntimeError, match="no LLM"):
                judge.evaluate("Some text")

    def test_parse_llm_response(self):
        config = JudgeConfig(dimensions_for_llm=["accuracy", "clarity"])
        judge = DocumentJudge(config)
        response = "accuracy: 0.85 | Good content\nclarity: 0.92 | Clear writing"
        scores = judge._parse_llm_response(response)
        assert abs(scores["accuracy"] - 0.85) < 0.01
        assert abs(scores["clarity"] - 0.92) < 0.01

    def test_parse_llm_response_clamps_scores(self):
        config = JudgeConfig(dimensions_for_llm=["accuracy"])
        judge = DocumentJudge(config)
        response = "accuracy: 1.5 | Impossibly good"
        scores = judge._parse_llm_response(response)
        assert scores["accuracy"] == 1.0

    def test_build_judge_prompt(self):
        config = JudgeConfig(dimensions_for_llm=["accuracy", "clarity"])
        judge = DocumentJudge(config)
        prompt = judge._build_judge_prompt("Test doc", "readme", "extra context")
        assert "accuracy" in prompt
        assert "clarity" in prompt
        assert "readme" in prompt
        assert "extra context" in prompt
        assert "Test doc" in prompt

    def test_merge_scores(self):
        from vetinari.validation import DimensionScore, evaluate_document

        judge = DocumentJudge()
        text = "# Test\n\nA document for merge testing."
        report = evaluate_document(text)

        from vetinari.validation import get_profile_for_type

        profile = get_profile_for_type("default")
        llm_scores = {"accuracy": 0.95}
        merged = judge._merge_scores(report, llm_scores, profile)

        accuracy_ds = next(ds for ds in merged.dimension_scores if ds.dimension == "accuracy")
        assert accuracy_ds.score == 0.95
        assert any("LLM judge" in f for f in accuracy_ds.findings)
