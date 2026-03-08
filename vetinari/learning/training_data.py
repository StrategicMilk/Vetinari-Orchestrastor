"""
Vetinari Training Data Collector
==================================
Records every agent execution to a JSONL file, enabling:

1. **SFT (Supervised Fine-Tuning)** — export high-quality completions
   (score >= threshold) as training pairs for local model fine-tuning.
2. **DPO (Direct Preference Optimization)** — export preference pairs
   (best vs. worst response for the same task type) for alignment training.
3. **Prompt dataset** — export prompt variant performance for DSPy optimization.

Design principles
-----------------
- Zero-overhead on the critical path: all I/O is fire-and-forget via a
  background thread queue.
- Thread-safe JSONL append with advisory file locking.
- Configurable via environment variables.
- Hardware-aware: records VRAM snapshot at time of inference.

Usage::

    from vetinari.learning.training_data import get_training_collector

    collector = get_training_collector()
    collector.record(
        task=task_description,
        prompt=system_prompt + user_prompt,
        response=llm_output,
        score=0.85,
        model_id="qwen3-vl-32b-gemini-heretic-uncensored-thinking",
        task_type="coding",
        latency_ms=4200,
        tokens_used=1024,
    )

    # Export for fine-tuning
    sft_data = collector.export_sft_dataset(min_score=0.8)
    dpo_data = collector.export_dpo_dataset()
"""

from __future__ import annotations

import json
import logging
import os
import queue
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_DEFAULT_PATH = os.environ.get(
    "VETINARI_TRAINING_DATA_PATH",
    "./training_data.jsonl",
)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class TrainingRecord:
    """A single execution record for training data collection."""
    record_id: str
    timestamp: str
    task: str                      # Task description / user prompt
    prompt: str                    # Full prompt sent to LLM (system + user)
    response: str                  # LLM output
    score: float                   # Quality score (0.0-1.0)
    model_id: str                  # Model that produced the response
    task_type: str                 # coding / research / analysis / etc.
    prompt_variant_id: str = ""    # For A/B tracking
    agent_type: str = ""           # Agent that executed this task
    latency_ms: int = 0
    tokens_used: int = 0
    success: bool = True
    vram_used_gb: float = 0.0
    benchmark_suite: str = ""             # Benchmark suite name (e.g. "toolbench", "swe-bench")
    benchmark_pass: bool = False          # Whether the output passed benchmark validation
    benchmark_score: float = 0.0          # Benchmark score 0.0-1.0 (pass_rate or avg_score)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_sft_pair(self) -> Dict[str, str]:
        """Convert to a (prompt, completion) pair for SFT training."""
        return {
            "prompt": self.prompt,
            "completion": self.response,
            "score": self.score,
            "task_type": self.task_type,
        }


# ---------------------------------------------------------------------------
# Collector
# ---------------------------------------------------------------------------

class TrainingDataCollector:
    """Thread-safe training data recorder with background I/O."""

    _instance: Optional["TrainingDataCollector"] = None
    _cls_lock = threading.Lock()

    def __init__(self, output_path: str = _DEFAULT_PATH):
        self._output_path = Path(output_path)
        self._queue: queue.Queue = queue.Queue(maxsize=1000)
        self._lock = threading.Lock()
        self._shutdown = threading.Event()
        self._worker = threading.Thread(
            target=self._write_worker,
            name="TrainingDataWriter",
            daemon=True,
        )
        self._worker.start()
        self._record_count = 0

    # ------------------------------------------------------------------
    # Singleton
    # ------------------------------------------------------------------

    @classmethod
    def get_instance(cls, output_path: str = _DEFAULT_PATH) -> "TrainingDataCollector":
        with cls._cls_lock:
            if cls._instance is None:
                cls._instance = cls(output_path=output_path)
        return cls._instance

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record(
        self,
        task: str,
        prompt: str,
        response: str,
        score: float,
        model_id: str,
        task_type: str = "general",
        prompt_variant_id: str = "",
        agent_type: str = "",
        latency_ms: int = 0,
        tokens_used: int = 0,
        success: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
        benchmark_suite: str = "",
        benchmark_pass: bool = False,
        benchmark_score: float = 0.0,
    ) -> None:
        """Record a task execution (non-blocking; queued for background write).

        Args:
            benchmark_suite: Name of the benchmark suite that validated this output.
            benchmark_pass: Whether the output passed benchmark validation.
            benchmark_score: Benchmark score 0.0-1.0 (pass_rate or avg_score).
        """
        # Snapshot VRAM usage if available
        vram_used = 0.0
        try:
            from vetinari.vram_manager import get_vram_manager
            vram_used = get_vram_manager().get_used_vram_gb()
        except Exception:
            pass

        import uuid
        rec = TrainingRecord(
            record_id=f"tr_{uuid.uuid4().hex[:8]}",
            timestamp=datetime.now().isoformat(),
            task=task[:2000],
            prompt=prompt[:4000],
            response=response[:8000],
            score=round(score, 4),
            model_id=model_id,
            task_type=task_type,
            prompt_variant_id=prompt_variant_id,
            agent_type=agent_type,
            latency_ms=latency_ms,
            tokens_used=tokens_used,
            success=success,
            vram_used_gb=round(vram_used, 2),
            benchmark_suite=benchmark_suite,
            benchmark_pass=benchmark_pass,
            benchmark_score=round(benchmark_score, 4),
            metadata=metadata or {},
        )
        try:
            self._queue.put_nowait(rec)
        except queue.Full:
            logger.debug("[TrainingDataCollector] Queue full; dropping record")

    # ------------------------------------------------------------------
    # Background writer
    # ------------------------------------------------------------------

    def _write_worker(self) -> None:
        """Background thread: flush records to JSONL file."""
        while not self._shutdown.is_set():
            try:
                rec = self._queue.get(timeout=1.0)
                self._append(rec)
                self._queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.debug(f"[TrainingDataCollector] Write error: {e}")

    def _append(self, rec: TrainingRecord) -> None:
        """Append a record to the JSONL file (thread-safe)."""
        with self._lock:
            try:
                self._output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self._output_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec.to_dict(), ensure_ascii=False) + "\n")
                self._record_count += 1
            except Exception as e:
                logger.debug(f"[TrainingDataCollector] Append failed: {e}")

    def flush(self) -> None:
        """Wait for the queue to drain."""
        self._queue.join()

    def shutdown(self) -> None:
        """Flush remaining records and stop the background worker."""
        self.flush()
        self._shutdown.set()

    # ------------------------------------------------------------------
    # Export methods
    # ------------------------------------------------------------------

    def _load_all(self) -> List[TrainingRecord]:
        """Load all records from the JSONL file."""
        if not self._output_path.exists():
            return []
        records: List[TrainingRecord] = []
        with open(self._output_path, encoding="utf-8") as f:
            for line in f:
                try:
                    d = json.loads(line.strip())
                    records.append(TrainingRecord(**d))
                except Exception:
                    continue
        return records

    def export_sft_dataset(
        self,
        min_score: float = 0.8,
        task_type: Optional[str] = None,
        max_records: int = 10000,
    ) -> List[Dict[str, Any]]:
        """Export high-quality completions for supervised fine-tuning.

        Returns list of ``{"prompt": ..., "completion": ..., "score": ...}`` dicts.
        """
        self.flush()
        all_records = self._load_all()
        filtered = [
            r for r in all_records
            if r.score >= min_score
            and r.success
            and (task_type is None or r.task_type == task_type)
        ]
        # Sort by score descending, take top N
        filtered.sort(key=lambda r: -r.score)
        return [r.to_sft_pair() for r in filtered[:max_records]]

    def export_dpo_dataset(
        self,
        task_type: Optional[str] = None,
        min_score_gap: float = 0.2,
    ) -> List[Dict[str, Any]]:
        """Export preference pairs for DPO alignment training.

        Groups records by task text; pairs highest-scoring with lowest-scoring
        responses for the same task (if gap >= min_score_gap).

        Returns list of ``{"prompt": ..., "chosen": ..., "rejected": ...}`` dicts.
        """
        self.flush()
        all_records = self._load_all()
        if task_type:
            all_records = [r for r in all_records if r.task_type == task_type]

        # Group by task text (normalized)
        by_task: Dict[str, List[TrainingRecord]] = defaultdict(list)
        for r in all_records:
            key = r.task[:200].strip().lower()
            by_task[key].append(r)

        pairs: List[Dict[str, Any]] = []
        for task_key, group in by_task.items():
            if len(group) < 2:
                continue
            group.sort(key=lambda r: r.score)
            worst = group[0]
            best = group[-1]
            if (best.score - worst.score) >= min_score_gap:
                pairs.append({
                    "prompt": best.task,
                    "chosen": best.response,
                    "rejected": worst.response,
                    "chosen_score": best.score,
                    "rejected_score": worst.score,
                    "task_type": best.task_type,
                })
        return pairs

    def export_prompt_dataset(self) -> List[Dict[str, Any]]:
        """Export prompt variant performance for DSPy/A-B analysis."""
        self.flush()
        all_records = self._load_all()
        variant_records = [r for r in all_records if r.prompt_variant_id]
        return [
            {
                "prompt_variant_id": r.prompt_variant_id,
                "task_type": r.task_type,
                "score": r.score,
                "latency_ms": r.latency_ms,
                "tokens_used": r.tokens_used,
                "model_id": r.model_id,
            }
            for r in variant_records
        ]

    def export_hf_dataset(
        self,
        min_score: float = 0.8,
        task_type: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        """Export records in HuggingFace Datasets / Alpaca format.

        Returns list of ``{"instruction": ..., "input": ..., "output": ...}`` dicts.
        Does NOT require the ``datasets`` library — returns plain Python dicts.
        """
        self.flush()
        all_records = self._load_all()
        filtered = [
            r for r in all_records
            if r.score >= min_score
            and r.success
            and (task_type is None or r.task_type == task_type)
        ]
        filtered.sort(key=lambda r: -r.score)
        return [
            {
                "instruction": r.task,
                "input": r.prompt,
                "output": r.response,
            }
            for r in filtered
        ]

    def export_few_shot_examples(
        self,
        task_type: str,
        k: int = 5,
    ) -> List[Dict[str, str]]:
        """Return top-k highest-scoring examples for a specific task type.

        Returns list of ``{"input": ..., "output": ...}`` pairs suitable for
        few-shot prompt construction.
        """
        self.flush()
        all_records = self._load_all()
        filtered = [r for r in all_records if r.task_type == task_type and r.success]
        filtered.sort(key=lambda r: -r.score)
        return [
            {"input": r.task, "output": r.response}
            for r in filtered[:k]
        ]

    def export_ranking_dataset(self) -> List[Dict[str, Any]]:
        """Export grouped, ranked responses for reward-model training.

        Groups records by task text, ranks all responses best→worst by score.
        Returns list of ``{"prompt": ..., "responses": [...]}`` dicts where
        ``responses`` is ordered from highest to lowest score.
        """
        self.flush()
        all_records = self._load_all()

        by_task: Dict[str, List[TrainingRecord]] = defaultdict(list)
        for r in all_records:
            key = r.task[:200].strip().lower()
            by_task[key].append(r)

        result: List[Dict[str, Any]] = []
        for group in by_task.values():
            if len(group) < 2:
                continue
            group.sort(key=lambda r: -r.score)
            result.append({
                "prompt": group[0].task,
                "responses": [
                    {"response": r.response, "score": r.score}
                    for r in group
                ],
            })
        return result

    def get_stats(self) -> Dict[str, Any]:
        """Return a summary of collected training data."""
        self.flush()
        all_records = self._load_all()
        if not all_records:
            return {"total": 0, "queued": self._queue.qsize()}

        by_type: Dict[str, List[float]] = defaultdict(list)
        for r in all_records:
            by_type[r.task_type].append(r.score)

        return {
            "total": len(all_records),
            "queued": self._queue.qsize(),
            "sft_eligible": sum(1 for r in all_records if r.score >= 0.8),
            "avg_score": round(sum(r.score for r in all_records) / len(all_records), 3),
            "by_task_type": {
                tt: {
                    "count": len(scores),
                    "avg_score": round(sum(scores) / len(scores), 3),
                }
                for tt, scores in by_type.items()
            },
            "output_path": str(self._output_path),
        }


# ---------------------------------------------------------------------------
# Module-level accessor
# ---------------------------------------------------------------------------

_collector: Optional[TrainingDataCollector] = None
_collector_lock = threading.Lock()


def get_training_collector(
    output_path: str = _DEFAULT_PATH,
) -> TrainingDataCollector:
    """Return the global TrainingDataCollector singleton."""
    global _collector
    if _collector is None:
        with _collector_lock:
            if _collector is None:
                _collector = TrainingDataCollector.get_instance(output_path)
    return _collector
