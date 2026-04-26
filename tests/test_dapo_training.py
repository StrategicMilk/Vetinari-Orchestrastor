"""Tests for vetinari.training.dapo — DAPO Training Stage."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from vetinari.training.dapo import (
    TIER_EFFICIENCY_WEIGHT,
    TIER_EXECUTION_WEIGHT,
    TIER_HEURISTIC_WEIGHT,
    TIER_LLM_JUDGE_WEIGHT,
    TIER_STATIC_WEIGHT,
    TIER_TEST_WEIGHT,
    DapoExecutionResult,
    DapoTrainingResult,
    RewardBreakdown,
    StageResult,
    TrainingStageOrchestrator,
    compute_dapo_reward,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def perfect_quality() -> dict:
    """Quality result where all scores are 1.0 and no rework."""
    return {
        "test_pass_rate": 1.0,
        "static_analysis_score": 1.0,
        "heuristic_score": 1.0,
        "score": 1.0,
        "rework_count": 0,
    }


@pytest.fixture
def empty_quality() -> dict:
    """Quality result with all zeroes."""
    return {}


@pytest.fixture
def passing_execution() -> DapoExecutionResult:
    """Successful execution result."""
    return DapoExecutionResult(success=True, exit_code=0)


@pytest.fixture
def failing_execution() -> DapoExecutionResult:
    """Failed execution result."""
    return DapoExecutionResult(success=False, error_message="SyntaxError", exit_code=1)


@pytest.fixture
def orchestrator() -> TrainingStageOrchestrator:
    """TrainingStageOrchestrator instance."""
    return TrainingStageOrchestrator()


class TestDapoScriptBuilders:
    def test_simpo_script_uses_requested_revision(self):
        from vetinari.training.dapo_scripts import build_simpo_script

        revision = "a" * 40
        script = build_simpo_script("owner/model", "pairs.jsonl", "out", model_revision=revision)

        assert f'revision="{revision}"' in script
        assert 'revision="main"' not in script

    def test_dapo_reward_dpo_script_uses_requested_revision(self):
        from vetinari.training.dapo_scripts import build_dapo_reward_dpo_script

        revision = "b" * 40
        script = build_dapo_reward_dpo_script("owner/model", "pairs.jsonl", "out", model_revision=revision)

        assert f'revision="{revision}"' in script
        assert 'revision="main"' not in script


# ---------------------------------------------------------------------------
# Tier weight tests
# ---------------------------------------------------------------------------


class TestTierWeights:
    def test_tier_weights_sum_to_one(self):
        """All tier weights must sum to exactly 1.0."""
        total = (
            TIER_EXECUTION_WEIGHT
            + TIER_TEST_WEIGHT
            + TIER_STATIC_WEIGHT
            + TIER_HEURISTIC_WEIGHT
            + TIER_LLM_JUDGE_WEIGHT
            + TIER_EFFICIENCY_WEIGHT
        )
        assert abs(total - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# compute_dapo_reward tests
# ---------------------------------------------------------------------------


class TestComputeDapoReward:
    def test_all_tiers_contribute_correctly(self, perfect_quality, passing_execution):
        """With all scores at 1.0 and successful execution, reward is 1.0."""
        result = compute_dapo_reward(perfect_quality, passing_execution)

        assert result.tier1_execution == pytest.approx(TIER_EXECUTION_WEIGHT)
        assert result.tier2_test == pytest.approx(TIER_TEST_WEIGHT)
        assert result.tier3_static == pytest.approx(TIER_STATIC_WEIGHT)
        assert result.tier4_heuristic == pytest.approx(TIER_HEURISTIC_WEIGHT)
        assert result.tier5_llm_judge == pytest.approx(TIER_LLM_JUDGE_WEIGHT)
        assert result.efficiency_bonus == pytest.approx(TIER_EFFICIENCY_WEIGHT)
        assert result.total_reward == pytest.approx(1.0)

    def test_hard_floor_when_execution_fails(self, perfect_quality, failing_execution):
        """Failed execution sets total_reward=0.0 and hard_floor_applied=True."""
        result = compute_dapo_reward(perfect_quality, failing_execution)

        assert result.total_reward == 0.0
        assert result.hard_floor_applied is True

    def test_hard_floor_zeros_all_tiers(self, perfect_quality, failing_execution):
        """When hard floor applies, no tier scores should be credited."""
        result = compute_dapo_reward(perfect_quality, failing_execution)

        assert result.tier1_execution == 0.0
        assert result.tier2_test == 0.0
        assert result.tier3_static == 0.0
        assert result.tier4_heuristic == 0.0
        assert result.tier5_llm_judge == 0.0
        assert result.efficiency_bonus == 0.0

    def test_efficiency_bonus_when_no_rework(self, passing_execution):
        """rework_count=0 grants the efficiency bonus."""
        quality = {"rework_count": 0}
        result = compute_dapo_reward(quality, passing_execution)

        assert result.efficiency_bonus == pytest.approx(TIER_EFFICIENCY_WEIGHT)

    def test_no_efficiency_bonus_when_rework_present(self, passing_execution):
        """rework_count>0 means no efficiency bonus."""
        quality = {"rework_count": 2}
        result = compute_dapo_reward(quality, passing_execution)

        assert result.efficiency_bonus == 0.0

    def test_reward_capped_at_one(self, perfect_quality):
        """total_reward must never exceed 1.0."""
        result = compute_dapo_reward(perfect_quality)
        assert result.total_reward <= 1.0

    def test_partial_scores_scale_correctly(self, passing_execution):
        """Partial quality scores produce proportional tier contributions."""
        quality = {
            "test_pass_rate": 0.5,
            "static_analysis_score": 0.8,
            "heuristic_score": 0.6,
            "score": 0.4,
            "rework_count": 1,
        }
        result = compute_dapo_reward(quality, passing_execution)

        assert result.tier2_test == pytest.approx(0.5 * TIER_TEST_WEIGHT)
        assert result.tier3_static == pytest.approx(0.8 * TIER_STATIC_WEIGHT)
        assert result.tier4_heuristic == pytest.approx(0.6 * TIER_HEURISTIC_WEIGHT)
        assert result.tier5_llm_judge == pytest.approx(0.4 * TIER_LLM_JUDGE_WEIGHT)
        assert result.efficiency_bonus == 0.0

    def test_no_execution_result_skips_tier1(self, perfect_quality):
        """Omitting execution_result means tier1 is 0 and no hard floor."""
        result = compute_dapo_reward(perfect_quality, execution_result=None)

        assert result.tier1_execution == 0.0
        assert result.hard_floor_applied is False
        # Other tiers still contribute
        assert result.tier2_test > 0.0

    def test_empty_quality_defaults_to_zeros(self, empty_quality):
        """Missing quality keys default to 0.0 with no efficiency bonus.

        Defect 4 fix: absent rework_count key means rework status is unknown,
        so no efficiency bonus is granted (only rework_count explicitly == 0 grants it).
        """
        result = compute_dapo_reward(empty_quality)

        assert result.tier2_test == 0.0
        assert result.tier3_static == 0.0
        assert result.tier4_heuristic == 0.0
        assert result.tier5_llm_judge == 0.0
        # rework_count key is absent — unknown status, no bonus
        assert result.efficiency_bonus == 0.0


# ---------------------------------------------------------------------------
# RewardBreakdown.to_dict tests
# ---------------------------------------------------------------------------


class TestRewardBreakdownToDict:
    def test_to_dict_serializes_all_fields(self, perfect_quality, passing_execution):
        """to_dict() includes all expected keys."""
        result = compute_dapo_reward(perfect_quality, passing_execution)
        d = result.to_dict()

        expected_keys = {
            "total_reward",
            "tier1_execution",
            "tier2_test",
            "tier3_static",
            "tier4_heuristic",
            "tier5_llm_judge",
            "efficiency_bonus",
            "hard_floor_applied",
        }
        assert set(d.keys()) == expected_keys

    def test_to_dict_rounds_floats_to_4dp(self):
        """Numeric values are rounded to 4 decimal places."""
        breakdown = RewardBreakdown(total_reward=0.333333333)
        d = breakdown.to_dict()
        assert d["total_reward"] == 0.3333

    def test_to_dict_hard_floor_is_bool(self, perfect_quality, failing_execution):
        """hard_floor_applied is a boolean in the dict output."""
        result = compute_dapo_reward(perfect_quality, failing_execution)
        d = result.to_dict()
        assert isinstance(d["hard_floor_applied"], bool)
        assert d["hard_floor_applied"] is True


# ---------------------------------------------------------------------------
# Dataclass field tests
# ---------------------------------------------------------------------------


class TestExecutionResultFields:
    def test_default_values(self):
        """ExecutionResult has expected defaults."""
        er = DapoExecutionResult()
        assert er.success is False
        assert er.error_message == ""
        assert er.exit_code == 0

    def test_custom_values(self):
        """ExecutionResult stores provided values."""
        er = DapoExecutionResult(success=True, error_message="ok", exit_code=0)
        assert er.success is True
        assert er.error_message == "ok"


class TestTrainingResultFields:
    def test_default_values(self):
        """TrainingResult has expected defaults."""
        tr = DapoTrainingResult()
        assert tr.success is False
        assert tr.final_model == ""
        assert tr.stage_failed == ""
        assert tr.error == ""
        assert tr.stage_results == []

    def test_stage_results_is_independent_list(self):
        """Each TrainingResult instance has its own stage_results list."""
        t1 = DapoTrainingResult()
        t2 = DapoTrainingResult()
        t1.stage_results.append(StageResult())
        assert len(t2.stage_results) == 0


# ---------------------------------------------------------------------------
# TrainingStageOrchestrator tests
# ---------------------------------------------------------------------------


class TestTrainingStageOrchestratorStages:
    def test_stages_has_three_entries(self):
        """STAGES must contain exactly 3 stage names."""
        assert len(TrainingStageOrchestrator.STAGES) == 3

    def test_stages_order(self):
        """STAGES must be in SFT -> SimPO -> DAPO order."""
        assert TrainingStageOrchestrator.STAGES == ["sft", "simpo", "dapo"]


class TestTrainingStageOrchestratorRunPipeline:
    @pytest.fixture(autouse=True)
    def _mock_training_deps(self, monkeypatch):
        """Mock subprocess.run and training data so tests don't need torch/trl."""
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout = ""
        mock_proc.stderr = ""
        monkeypatch.setattr("subprocess.run", lambda *a, **kw: mock_proc)

        # Prevent DAPO/SimPO from finding real training data on disk
        mock_collector = MagicMock()
        mock_collector.export_ranking_dataset.return_value = []
        mock_curator = MagicMock()
        mock_curator.return_value.curate_dpo.side_effect = ValueError("no data")

        monkeypatch.setattr(
            "vetinari.learning.training_data.get_training_collector",
            lambda: mock_collector,
        )
        # Use direct module reference — string path patching fails in full
        # suite when parent-package attributes diverge from sys.modules.
        import vetinari.training.pipeline as _pipeline_mod

        monkeypatch.setattr(_pipeline_mod, "DataCurator", mock_curator)
        # Also patch the sys.modules entry if it's a different object
        _sysmod_pipeline = sys.modules.get("vetinari.training.pipeline")
        if _sysmod_pipeline is not None and _sysmod_pipeline is not _pipeline_mod:
            monkeypatch.setattr(_sysmod_pipeline, "DataCurator", mock_curator)

        # Skip stage validation (subprocess mock doesn't create output dirs)
        # Use direct module reference — string path fails in full suite when
        # vetinari.training.dapo is both a module and an attribute.
        import vetinari.training.dapo as _dapo_mod

        monkeypatch.setattr(
            _dapo_mod.TrainingStageOrchestrator,
            "_validate_stage_output",
            lambda self, stage, inp, out: True,
        )

    def test_run_pipeline_returns_training_result(self, orchestrator, tmp_path):
        """run_pipeline returns a TrainingResult instance."""
        result = orchestrator.run_pipeline(
            base_model="test-model",
            dataset_path=tmp_path / "data.jsonl",
        )
        assert isinstance(result, DapoTrainingResult)

    def test_run_pipeline_succeeds_with_valid_inputs(self, orchestrator, tmp_path):
        """Pipeline runs all stages successfully with a valid base model."""
        result = orchestrator.run_pipeline(
            base_model="test-model",
            dataset_path=tmp_path / "data.jsonl",
        )
        assert result.success is True

    def test_run_pipeline_collects_all_stage_results(self, orchestrator, tmp_path):
        """All three stages produce a StageResult entry."""
        result = orchestrator.run_pipeline(
            base_model="test-model",
            dataset_path=tmp_path / "data.jsonl",
        )
        assert len(result.stage_results) == 3

    def test_run_pipeline_stage_names_match(self, orchestrator, tmp_path):
        """Stage results are labelled with the correct stage names."""
        result = orchestrator.run_pipeline(
            base_model="test-model",
            dataset_path=tmp_path / "data.jsonl",
        )
        names = [sr.stage_name for sr in result.stage_results]
        assert names == ["sft", "simpo", "dapo"]

    def test_run_pipeline_skips_when_prerequisites_not_met(self, orchestrator, tmp_path):
        """SFT skips gracefully when dataset_path does not exist."""
        # dataset_path does not exist on disk; sft readiness check returns False
        non_existent = tmp_path / "missing" / "data.jsonl"
        result = orchestrator.run_pipeline(
            base_model="test-model",
            dataset_path=non_existent,
        )
        # Pipeline should still succeed; SFT stage is skipped
        assert result.success is True
        sft_result = result.stage_results[0]
        assert sft_result.metrics.get("skipped") is True

    def test_run_pipeline_uses_config_overrides(self, orchestrator, tmp_path):
        """Config overrides are passed through to the DAPO stage."""
        config = {"dapo": {"min_groups": 10, "group_size": 8}}
        result = orchestrator.run_pipeline(
            base_model="test-model",
            dataset_path=tmp_path / "data.jsonl",
            config=config,
        )
        dapo_result = next(sr for sr in result.stage_results if sr.stage_name == "dapo")
        # DAPO stage may skip (no data / no trl) but should still produce a result
        assert dapo_result.stage_name == "dapo"
        assert dapo_result.success is True

    def test_run_pipeline_final_model_is_set(self, orchestrator, tmp_path):
        """final_model is non-empty on success."""
        result = orchestrator.run_pipeline(
            base_model="my-base-model",
            dataset_path=tmp_path / "data.jsonl",
        )
        assert result.final_model != ""


# ---------------------------------------------------------------------------
# run_sft_stage failure propagation tests (defect 2)
# ---------------------------------------------------------------------------


class TestRunSftStageFailurePropagation:
    def test_import_error_produces_skipped_success(self, tmp_path):
        """ImportError during SFT pipeline import returns success=True with skipped flag.

        Defect 2 fix: training libraries unavailable is a skip, not a failure.
        """
        from vetinari.training.dapo_stages import run_sft_stage

        # Remove the pipeline module so the late import inside run_sft_stage raises ImportError
        saved = sys.modules.pop("vetinari.training.pipeline", None)
        sys.modules["vetinari.training.pipeline"] = None  # type: ignore[assignment]
        try:
            result = run_sft_stage("my-model", tmp_path / "data.jsonl", {})
        finally:
            # Restore module state
            if saved is not None:
                sys.modules["vetinari.training.pipeline"] = saved
            else:
                sys.modules.pop("vetinari.training.pipeline", None)

        assert result.success is True
        assert result.metrics.get("skipped") is True
        assert result.stage_name == "sft"

    def test_runtime_failure_propagates_as_failure(self, monkeypatch, tmp_path):
        """RuntimeError during SFT pipeline run returns success=False (defect 2 fix).

        A GPU OOM or similar runtime failure must not be silently treated as a skip.
        """
        import vetinari.training.pipeline as _pipeline_mod
        from vetinari.training.dapo_stages import run_sft_stage

        def _raise_runtime(*args, **kwargs):
            raise RuntimeError("GPU OOM")

        monkeypatch.setattr(_pipeline_mod, "TrainingPipeline", _raise_runtime)
        # Ensure sys.modules returns the patched module
        monkeypatch.setitem(sys.modules, "vetinari.training.pipeline", _pipeline_mod)

        result = run_sft_stage("my-model", tmp_path / "data.jsonl", {})

        assert result.success is False
        assert "SFT pipeline failed" in result.error
        assert result.stage_name == "sft"


# ---------------------------------------------------------------------------
# _build_dapo_preference_pairs tests (defects 18, 19)
# ---------------------------------------------------------------------------


class TestBuildDapoPreferencePairs:
    def test_pairs_include_reward_scores(self):
        """Preference pairs carry chosen_reward and rejected_reward fields (defect 19)."""
        from vetinari.training.dapo_stages import _build_dapo_preference_pairs

        ranking_data = [
            {
                "prompt": "Write a function",
                "responses": [
                    {"response": "def f(): pass", "score": 0.9, "test_pass_rate": 0.8, "static_analysis_score": 0.7},
                    {
                        "response": "def f():\n    ...",
                        "score": 0.3,
                        "test_pass_rate": 0.2,
                        "static_analysis_score": 0.4,
                    },
                ],
            }
        ]
        pairs = _build_dapo_preference_pairs(ranking_data)
        assert len(pairs) == 1
        pair = pairs[0]
        assert "chosen_reward" in pair, "chosen_reward must be present (defect 19 fix)"
        assert "rejected_reward" in pair, "rejected_reward must be present (defect 19 fix)"
        assert pair["chosen_reward"] > pair["rejected_reward"], "chosen must outrank rejected"

    def test_execution_result_zero_reward_for_failed_execution(self):
        """A response with success=False gets zero reward via hard floor (defect 18)."""
        from vetinari.training.dapo_stages import _build_dapo_preference_pairs

        ranking_data = [
            {
                "prompt": "Fix the bug",
                "responses": [
                    # Both fail execution — hard floor gives 0.0
                    {"response": "bad code A", "score": 0.8, "test_pass_rate": 0.9, "success": False},
                    {"response": "bad code B", "score": 0.5, "test_pass_rate": 0.5, "success": False},
                ],
            }
        ]
        pairs = _build_dapo_preference_pairs(ranking_data)
        # Both rewards are 0.0 from hard floor — no meaningful pair (chosen == rejected)
        assert len(pairs) == 0, "Equal rewards must not produce a preference pair"

    def test_skips_groups_with_fewer_than_two_responses(self):
        """Groups with only one response are skipped."""
        from vetinari.training.dapo_stages import _build_dapo_preference_pairs

        ranking_data = [
            {"prompt": "Solo", "responses": [{"response": "only one", "score": 0.8}]},
        ]
        pairs = _build_dapo_preference_pairs(ranking_data)
        assert pairs == []

    def test_prompt_and_responses_in_pair(self):
        """Pair carries prompt, chosen, and rejected text."""
        from vetinari.training.dapo_stages import _build_dapo_preference_pairs

        ranking_data = [
            {
                "prompt": "hello",
                "responses": [
                    {"response": "good answer", "score": 1.0, "test_pass_rate": 1.0, "static_analysis_score": 1.0},
                    {"response": "bad answer", "score": 0.0, "test_pass_rate": 0.0, "static_analysis_score": 0.0},
                ],
            }
        ]
        pairs = _build_dapo_preference_pairs(ranking_data)
        assert len(pairs) == 1
        assert pairs[0]["prompt"] == "hello"
        assert pairs[0]["chosen"] == "good answer"
        assert pairs[0]["rejected"] == "bad answer"

    def test_execution_truth_beats_higher_reward_score(self):
        """A response with executes=True must be chosen over one with higher reward but executes=False.

        Defect 8 regression: when success=None is absent (skipping the execution tier),
        a non-executing response can accumulate a higher total reward than one that truly
        executes. The composite sort key (executes_rank, reward) must prevent this.
        """
        from vetinari.training.dapo_stages import _build_dapo_preference_pairs

        ranking_data = [
            {
                "prompt": "Implement sort",
                "responses": [
                    # Executes successfully but scores are modest
                    {
                        "response": "def sort(x): return sorted(x)",
                        "score": 0.5,
                        "test_pass_rate": 0.5,
                        "static_analysis_score": 0.5,
                        "success": True,
                    },
                    # Fails execution but has inflated quality scores
                    {
                        "response": "broken code",
                        "score": 0.9,
                        "test_pass_rate": 0.9,
                        "static_analysis_score": 0.9,
                        "success": False,
                    },
                ],
            }
        ]
        pairs = _build_dapo_preference_pairs(ranking_data)
        assert len(pairs) == 1
        pair = pairs[0]
        # The executing response must be chosen regardless of raw reward scores
        assert "sort" in pair["chosen"], (
            "Response with executes=True must be chosen over executes=False even with lower raw scores"
        )
        assert "broken" in pair["rejected"], "Non-executing response must be rejected"
        # Chosen reward is from the successfully-executing response (has execution tier bonus)
        assert pair["chosen_reward"] > pair["rejected_reward"], (
            "chosen_reward must exceed rejected_reward after composite sort"
        )

    def test_execution_none_ranks_below_true_but_above_false(self):
        """Responses with no execution data rank between executes=True and executes=False.

        Defect 8 regression: ensures executes_rank ordering is True > None > False.
        """
        from vetinari.training.dapo_stages import _build_dapo_preference_pairs

        ranking_data = [
            {
                "prompt": "Code task",
                "responses": [
                    # executes=True, moderate scores
                    {"response": "executes yes", "score": 0.4, "test_pass_rate": 0.4, "success": True},
                    # executes=False, high static scores
                    {"response": "executes no", "score": 0.95, "test_pass_rate": 0.95, "success": False},
                    # no success key, moderate scores
                    {"response": "executes unknown", "score": 0.6, "test_pass_rate": 0.6},
                ],
            }
        ]
        pairs = _build_dapo_preference_pairs(ranking_data)
        assert len(pairs) == 1
        # executes=True must be chosen; executes=False must be rejected
        assert "executes yes" in pairs[0]["chosen"]
        assert "executes no" in pairs[0]["rejected"]


# ---------------------------------------------------------------------------
# run_dapo_stage config override test (defect 20)
# ---------------------------------------------------------------------------


class TestRunDapoStageConfigOverride:
    def test_min_groups_config_override_causes_skip(self, tmp_path):
        """Setting min_groups higher than available data causes a graceful skip."""
        from vetinari.training.dapo_stages import run_dapo_stage

        mock_collector = MagicMock()
        # Return 3 groups — less than the min_groups=10 override
        mock_collector.export_ranking_dataset.return_value = [
            {"prompt": "p", "responses": []},
            {"prompt": "q", "responses": []},
            {"prompt": "r", "responses": []},
        ]

        with patch(
            "vetinari.learning.training_data.get_training_collector",
            return_value=mock_collector,
        ):
            result = run_dapo_stage(
                model="test-model",
                dataset_path=tmp_path / "data.jsonl",
                config={"dapo": {"min_groups": 10}},
            )

        assert result.success is True
        assert result.metrics.get("skipped") is True
        assert "insufficient_groups" in result.metrics.get("reason", "")

    def test_run_dapo_stage_skips_gracefully_when_no_collector(self, tmp_path):
        """run_dapo_stage returns a successful skip when the collector is unavailable."""
        from vetinari.training.dapo_stages import run_dapo_stage

        with patch(
            "vetinari.learning.training_data.get_training_collector",
            side_effect=ImportError("collector not available"),
        ):
            result = run_dapo_stage(
                model="base-model",
                dataset_path=tmp_path / "data.jsonl",
                config={},
            )

        assert result.success is True
        assert result.metrics.get("skipped") is True


# ---------------------------------------------------------------------------
# run_dapo_stage real-data tests (defect 10)
# ---------------------------------------------------------------------------


def _make_ranking_groups(n: int) -> list[dict]:
    """Return n ranking groups each with two clearly differentiated responses."""
    groups = []
    for i in range(n):
        groups.append({
            "prompt": f"Write function {i}",
            "responses": [
                {
                    "response": f"def f{i}(): return {i}",
                    "score": 0.9,
                    "test_pass_rate": 0.9,
                    "static_analysis_score": 0.8,
                    "success": True,
                },
                {
                    "response": f"broken_{i}",
                    "score": 0.1,
                    "test_pass_rate": 0.0,
                    "static_analysis_score": 0.2,
                    "success": False,
                },
            ],
        })
    return groups


class TestRunDapoStageRealData:
    """Tests that exercise run_dapo_stage past the skip gate (defect 10).

    All tests provide enough ranking groups and mock trl so the actual
    preference-pair building and dataset-writing logic runs — not just
    the early-return skip paths that existed before this fix.
    """

    def test_run_dapo_stage_writes_preference_dataset(self, tmp_path):
        """run_dapo_stage writes a JSONL preference dataset when data is sufficient.

        Defect 10 regression: previously all tests hit an early skip return.
        This test provides 6 groups (>= default min_groups=5) and patches trl
        as unavailable, which causes _execute_dapo_reward_dpo_training to
        write the file then return skipped=True with pairs_saved metric
        rather than never reaching dataset write at all.
        """
        from vetinari.training.dapo_stages import run_dapo_stage

        mock_collector = MagicMock()
        mock_collector.export_ranking_dataset.return_value = _make_ranking_groups(6)

        with (
            patch(
                "vetinari.learning.training_data.get_training_collector",
                return_value=mock_collector,
            ),
            patch.dict("sys.modules", {"trl": None}),
        ):
            result = run_dapo_stage(
                model="base-model",
                dataset_path=tmp_path / "data.jsonl",
                config={"dapo": {"min_groups": 5}},
            )

        # trl is absent so training is skipped, but preference data must have been saved
        assert result.success is True
        assert result.metrics.get("pairs_saved", 0) > 0, (
            "preference pairs must be written to disk even when trl is unavailable"
        )
        # The JSONL dataset file must exist on disk
        dapo_output_dir = tmp_path / "dapo_output"
        assert dapo_output_dir.exists(), "dapo_output directory must be created"
        dataset_file = dapo_output_dir / "dapo_preferences.jsonl"
        assert dataset_file.exists(), "dapo_preferences.jsonl must be written to disk"
        lines = dataset_file.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) > 0, "preference dataset must contain at least one pair"

    def test_run_dapo_stage_subprocess_called_when_trl_present(self, tmp_path):
        """run_dapo_stage invokes the training subprocess when trl is installed.

        Defect 10 regression: exercises the subprocess path, not just skip paths.
        The subprocess is mocked to return success so no real training happens.
        """
        import types

        from vetinari.training.dapo_stages import run_dapo_stage

        mock_collector = MagicMock()
        mock_collector.export_ranking_dataset.return_value = _make_ranking_groups(6)

        # Provide a fake trl module so the import check passes
        fake_trl = types.ModuleType("trl")

        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stderr = ""

        with (
            patch(
                "vetinari.learning.training_data.get_training_collector",
                return_value=mock_collector,
            ),
            patch.dict("sys.modules", {"trl": fake_trl}),
            patch("subprocess.run", return_value=mock_proc) as mock_run,
        ):
            result = run_dapo_stage(
                model="base-model",
                dataset_path=tmp_path / "data.jsonl",
                config={"dapo": {"min_groups": 5}},
            )

        assert result.success is True
        assert result.metrics.get("skipped") is not True, (
            "stage must not be skipped when trl is present and data is sufficient"
        )
        mock_run.assert_called_once(), "subprocess.run must be called to execute training script"

    def test_run_dapo_stage_preference_pairs_execution_truth_ordering(self, tmp_path):
        """Preference pairs built by run_dapo_stage respect execution-truth ordering (defect 8+10).

        Provides groups where one response executes and one does not. Verifies that
        the chosen response in each written pair is the executing one.
        """
        import json as _json

        from vetinari.training.dapo_stages import run_dapo_stage

        # Provide 6 groups where the executing response has lower raw scores
        groups = []
        for i in range(6):
            groups.append({
                "prompt": f"Task {i}",
                "responses": [
                    # executes=True, modest scores
                    {"response": f"good_{i}", "score": 0.4, "test_pass_rate": 0.4, "success": True},
                    # executes=False, inflated scores — must still be rejected
                    {"response": f"bad_{i}", "score": 0.95, "test_pass_rate": 0.95, "success": False},
                ],
            })

        mock_collector = MagicMock()
        mock_collector.export_ranking_dataset.return_value = groups

        with (
            patch(
                "vetinari.learning.training_data.get_training_collector",
                return_value=mock_collector,
            ),
            patch.dict("sys.modules", {"trl": None}),
        ):
            result = run_dapo_stage(
                model="base-model",
                dataset_path=tmp_path / "data.jsonl",
                config={"dapo": {"min_groups": 5}},
            )

        assert result.success is True
        dataset_file = tmp_path / "dapo_output" / "dapo_preferences.jsonl"
        assert dataset_file.exists()

        pairs = [_json.loads(line) for line in dataset_file.read_text(encoding="utf-8").strip().splitlines()]
        assert len(pairs) > 0
        for pair in pairs:
            assert "good_" in pair["chosen"], f"executing response must be chosen; got chosen={pair['chosen']!r}"
            assert "bad_" in pair["rejected"], (
                f"non-executing response must be rejected; got rejected={pair['rejected']!r}"
            )


# ---------------------------------------------------------------------------
# Import smoke test
# ---------------------------------------------------------------------------


class TestImports:
    def test_can_import_compute_dapo_reward(self):
        """compute_dapo_reward is importable from vetinari.training.dapo."""
        from vetinari.training.dapo import compute_dapo_reward as fn

        assert callable(fn)

    def test_can_import_training_stage_orchestrator(self):
        """TrainingStageOrchestrator is importable from vetinari.training.dapo."""
        from vetinari.training.dapo import TrainingStageOrchestrator as cls

        assert cls is not None
        assert callable(cls)
