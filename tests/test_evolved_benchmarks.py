"""Tests for self-evolved benchmarks — four reframing operations."""

from __future__ import annotations

import random
from pathlib import Path

import pytest

from vetinari.testing.evolved_benchmarks import BenchmarkEvolver, EvolvedPrompt


def _evolver(seed: int = 42, output_dir: Path | None = None) -> BenchmarkEvolver:
    """Convenience factory with a fixed seed for reproducibility."""
    return BenchmarkEvolver(output_dir=output_dir, seed=seed)


# -- test_paraphrase_changes_text ---------------------------------------------


def test_paraphrase_changes_text() -> None:
    """paraphrase() returns a string different from the input when synonyms apply."""
    evolver = _evolver()
    prompt = "Write a function to check the list for errors."
    result = evolver.paraphrase(prompt)

    # Output is a string and differs from the original (synonyms substituted)
    assert isinstance(result, str)
    assert len(result) > 0
    # At least one known term was substituted
    assert result != prompt


# -- test_add_noise_adds_constraints ------------------------------------------


def test_add_noise_adds_constraints() -> None:
    """add_noise() returns a modified version of the prompt with deliberate imperfections."""
    evolver = _evolver()
    prompt = "Write a function to check the list and return the error."
    result = evolver.add_noise(prompt)

    assert isinstance(result, str)
    assert len(result) > 0
    # The token count should be preserved (noise doesn't add/remove words)
    assert len(result.split()) == len(prompt.split())


# -- test_reverse_polarity_negates --------------------------------------------


def test_reverse_polarity_negates() -> None:
    """reverse_polarity() transforms an imperative prompt into a negative form."""
    evolver = _evolver()
    prompt = "Write a function to parse JSON."
    result = evolver.reverse_polarity(prompt)

    assert isinstance(result, str)
    # The action verb pattern produces a "mistakes/avoid/NOT" style phrase
    result_lower = result.lower()
    assert any(keyword in result_lower for keyword in ("avoid", "mistake", "not", "wrong"))
    # The core subject should still appear
    assert "json" in result_lower


# -- test_increase_complexity_adds_depth --------------------------------------


def test_increase_complexity_adds_depth() -> None:
    """increase_complexity() appends additional constraint text to the prompt."""
    evolver = _evolver()
    prompt = "Write a sorting function."
    result = evolver.increase_complexity(prompt)

    assert isinstance(result, str)
    # Result must be longer than the input
    assert len(result) > len(prompt)
    # The suffix marker is present
    assert "Additionally:" in result


# -- test_evolve_corpus_4x_expansion ------------------------------------------


def test_evolve_corpus_4x_expansion() -> None:
    """evolve_corpus() with N seeds produces exactly 4*N EvolvedPrompt objects."""
    evolver = _evolver()
    seeds = [
        "Write a function that sorts a list.",
        "Create a class for managing errors.",
        "Implement a simple cache.",
        "Design a retry mechanism.",
        "Build a logging helper.",
    ]
    evolved = evolver.evolve_corpus(seeds)

    assert len(evolved) == 4 * len(seeds)
    assert all(isinstance(ep, EvolvedPrompt) for ep in evolved)


# -- test_evolved_corpus_provenance -------------------------------------------


def test_evolved_corpus_provenance() -> None:
    """Each EvolvedPrompt records the original seed and the operation name."""
    evolver = _evolver()
    seeds = ["Write a function to sort a list.", "Create a class for caching."]
    evolved = evolver.evolve_corpus(seeds)

    operations = {ep.operation for ep in evolved}
    assert operations == {"paraphrase", "add_noise", "reverse_polarity", "increase_complexity"}

    for ep in evolved:
        assert ep.original in seeds
        assert len(ep.transformed) > 0
        assert ep.operation in operations
        assert ep.timestamp  # non-empty ISO timestamp


# -- test_deterministic_with_seed ---------------------------------------------


def test_deterministic_with_seed() -> None:
    """Same seed produces identical evolved prompts on repeated calls."""
    seeds = ["Write a function to return the list.", "Check the error in the function."]

    evolver_a = BenchmarkEvolver(seed=99)
    evolver_b = BenchmarkEvolver(seed=99)

    result_a = evolver_a.evolve_corpus(seeds)
    result_b = evolver_b.evolve_corpus(seeds)

    assert len(result_a) == len(result_b)
    for a, b in zip(result_a, result_b):
        assert a.transformed == b.transformed
        assert a.operation == b.operation
        assert a.original == b.original


# -- test_store_corpus ---------------------------------------------------------


def test_store_corpus_persists_yaml(tmp_path) -> None:
    """store_corpus writes evolved prompts to a timestamped YAML file."""
    evolver = _evolver(output_dir=tmp_path)
    seeds = ["Explain recursion."]
    evolved = evolver.evolve_corpus(seeds)

    out_path = evolver.store_corpus(evolved)

    assert out_path.exists()
    assert out_path.suffix == ".yaml"
    assert out_path.parent == tmp_path

    import yaml

    with open(out_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    assert data["total"] == len(evolved)
    assert len(data["evolved_benchmarks"]) == len(evolved)
    assert data["evolved_benchmarks"][0]["original"] == "Explain recursion."
