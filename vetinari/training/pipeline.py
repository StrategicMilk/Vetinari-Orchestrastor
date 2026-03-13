"""Vetinari Training Pipeline.

============================
Orchestrates the complete fine-tuning workflow for local models:

Local training (fits on 5090 32GB VRAM):
  7B-14B parameter models via QLoRA (unsloth + trl)

Cloud training (for 32B+ models):
  Export data → upload → trigger RunPod/Lambda → download adapter → merge → deploy

Pipeline stages:
  1. DataCurator   — curate JSONL from TrainingDataCollector
  2. LocalTrainer  — QLoRA fine-tuning via unsloth/trl
  3. Evaluator     — benchmark fine-tuned vs. baseline
  4. GGUFConverter — convert adapter to GGUF for LM Studio
  5. Deployer      — copy GGUF to LM Studio models directory

Usage::

    from vetinari.training.pipeline import TrainingPipeline

    pipeline = TrainingPipeline()

    # Check what's available
    logger.debug(pipeline.check_requirements())

    # Curate and train on accumulated data
    run = pipeline.run(
        base_model="Qwen/Qwen2.5-Coder-7B-Instruct",
        task_type="coding",
        min_score=0.8,
        epochs=3,
    )
    logger.debug(run.output_model_path)
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_MODELS_DIR = Path(
    os.environ.get(
        "LMSTUDIO_MODELS_DIR",
        str(Path.home() / ".lmstudio" / "models"),
    )
)


@dataclass
class TrainingRun:
    """Result of a training pipeline run."""

    run_id: str
    timestamp: str
    base_model: str
    task_type: str
    training_examples: int
    epochs: int
    success: bool
    output_model_path: str = ""
    adapter_path: str = ""
    eval_score: float = 0.0
    baseline_score: float = 0.0
    error: str = ""


class DataCurator:
    """Curates high-quality training data from the TrainingDataCollector."""

    def curate(
        self,
        task_type: str | None = None,
        min_score: float = 0.8,
        max_examples: int = 5000,
        output_dir: str = ".",
    ) -> str:
        """Curate SFT training data and write to a JSONL file.

        Returns the path to the curated dataset file.
        """
        from vetinari.learning.training_data import get_training_collector

        collector = get_training_collector()
        data = collector.export_sft_dataset(
            min_score=min_score,
            task_type=task_type,
            max_records=max_examples,
        )

        if not data:
            raise ValueError(f"No training data meets criteria (score>={min_score}, type={task_type})")

        # Format for Alpaca-style fine-tuning
        formatted = []
        for d in data:
            formatted.append(
                {
                    "instruction": d["prompt"][:1000],
                    "input": "",
                    "output": d["completion"][:2000],
                }
            )

        out_path = Path(output_dir) / f"sft_{task_type or 'general'}_{datetime.now().strftime('%Y%m%d_%H%M')}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for item in formatted:
                f.write(json.dumps(item) + "\n")

        logger.info("[DataCurator] Wrote %d examples to %s", len(formatted), out_path)
        return str(out_path)

    def curate_dpo(
        self,
        task_type: str | None = None,
        min_score_gap: float = 0.2,
        output_dir: str = ".",
    ) -> str:
        """Curate DPO preference pairs and write to a JSONL file."""
        from vetinari.learning.training_data import get_training_collector

        collector = get_training_collector()
        pairs = collector.export_dpo_dataset(
            task_type=task_type,
            min_score_gap=min_score_gap,
        )

        if not pairs:
            raise ValueError("No preference pairs available for DPO training")

        out_path = Path(output_dir) / f"dpo_{task_type or 'general'}_{datetime.now().strftime('%Y%m%d_%H%M')}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for pair in pairs:
                f.write(json.dumps(pair) + "\n")

        logger.info("[DataCurator] Wrote %d DPO pairs to %s", len(pairs), out_path)
        return str(out_path)


class LocalTrainer:
    """QLoRA fine-tuning via unsloth (2x faster) or trl fallback."""

    def check_available(self) -> dict[str, bool]:
        """Check which training libraries are installed."""
        result = {}
        for lib in ["unsloth", "trl", "peft", "transformers", "bitsandbytes"]:
            try:
                __import__(lib)
                result[lib] = True
            except ImportError:
                result[lib] = False
        return result

    def train_qlora(
        self,
        base_model: str,
        dataset_path: str,
        output_dir: str,
        epochs: int = 3,
        batch_size: int = 4,
        learning_rate: float = 2e-4,
        max_seq_length: int = 2048,
        lora_r: int = 16,
        use_unsloth: bool = True,
    ) -> str:
        """Run QLoRA training.  Returns path to the saved adapter.

        VRAM budget (5090 32GB):
          7B Q4 model:  ~8GB VRAM for training
          14B Q4 model: ~14GB VRAM for training
          32B model:    too large — use cloud_trainer
        """
        avail = self.check_available()
        if not avail.get("trl") and not avail.get("transformers"):
            raise RuntimeError("Training libraries not installed. Run: pip install trl peft bitsandbytes transformers")

        if use_unsloth and avail.get("unsloth"):
            return self._train_with_unsloth(
                base_model, dataset_path, output_dir, epochs, batch_size, learning_rate, max_seq_length, lora_r
            )
        else:
            return self._train_with_trl(
                base_model, dataset_path, output_dir, epochs, batch_size, learning_rate, max_seq_length, lora_r
            )

    def _train_with_unsloth(
        self, base_model, dataset_path, output_dir, epochs, batch_size, lr, max_seq_len, lora_r
    ) -> str:
        """Train using unsloth for 2x speed."""
        import json as _json

        # Use json.dumps() for all string parameters to prevent injection
        script = f"""
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer, TrainingArguments
import torch

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name={_json.dumps(str(base_model))},
    max_seq_length={int(max_seq_len)},
    dtype=None,
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r={int(lora_r)},
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha={int(lora_r) * 2},
    lora_dropout=0.05,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
)

dataset = load_dataset("json", data_files={_json.dumps(str(dataset_path))}, split="train")

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    max_seq_length={int(max_seq_len)},
    args=TrainingArguments(
        output_dir={_json.dumps(str(output_dir))},
        num_train_epochs={int(epochs)},
        per_device_train_batch_size={int(batch_size)},
        learning_rate={float(lr)},
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        save_strategy="epoch",
        optim="adamw_8bit",
    ),
)
trainer.train()
model.save_pretrained({_json.dumps(str(output_dir) + "/lora_adapter")})
logger.debug("Training complete:", {_json.dumps(str(output_dir) + "/lora_adapter")})
"""
        script_path = Path(output_dir) / "train_script.py"
        script_path.parent.mkdir(parents=True, exist_ok=True)
        script_path.write_text(script, encoding="utf-8")

        proc = subprocess.run(  # noqa: S603
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"Training failed:\n{proc.stderr[-2000:]}")

        adapter_path = str(Path(output_dir) / "lora_adapter")
        return adapter_path

    def _train_with_trl(self, base_model, dataset_path, output_dir, epochs, batch_size, lr, max_seq_len, lora_r) -> str:
        """Train using standard trl (slower than unsloth)."""
        import json as _json

        # Use json.dumps() for all string parameters to prevent injection
        script = f"""
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
import torch

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    {_json.dumps(str(base_model))},
    quantization_config=bnb_config,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained({_json.dumps(str(base_model))})
tokenizer.pad_token = tokenizer.eos_token

lora_config = LoraConfig(
    r={int(lora_r)},
    lora_alpha={int(lora_r) * 2},
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)

dataset = load_dataset("json", data_files={_json.dumps(str(dataset_path))}, split="train")

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    args=SFTConfig(
        output_dir={_json.dumps(str(output_dir))},
        num_train_epochs={int(epochs)},
        per_device_train_batch_size={int(batch_size)},
        learning_rate={float(lr)},
        max_seq_length={int(max_seq_len)},
        logging_steps=10,
        save_strategy="epoch",
    ),
)
trainer.train()
model.save_pretrained({_json.dumps(str(output_dir) + "/lora_adapter")})
logger.debug("Training complete")
"""
        script_path = Path(output_dir) / "train_trl_script.py"
        script_path.parent.mkdir(parents=True, exist_ok=True)
        script_path.write_text(script, encoding="utf-8")

        proc = subprocess.run(  # noqa: S603
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"Training failed:\n{proc.stderr[-2000:]}")

        return str(Path(output_dir) / "lora_adapter")


class GGUFConverter:
    """Converts trained LoRA adapter to GGUF format for LM Studio."""

    def convert(
        self,
        base_model: str,
        adapter_path: str,
        output_dir: str,
        quantization: str = "q4_k_m",
    ) -> str:
        """Merge adapter with base model and convert to GGUF.

        Returns path to the .gguf file.
        Requires: llama.cpp and pip install llama-cpp-python
        """
        out_path = Path(output_dir) / f"model_{quantization}.gguf"

        # Step 1: Merge adapter weights
        merge_script = f"""
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer
import torch

model = AutoPeftModelForCausalLM.from_pretrained(
    "{adapter_path}",
    device_map="cpu",
    torch_dtype=torch.float16,
)
model = model.merge_and_unload()
model.save_pretrained("{output_dir}/merged")
tokenizer = AutoTokenizer.from_pretrained("{base_model}")
tokenizer.save_pretrained("{output_dir}/merged")
logger.debug("Merge complete: {output_dir}/merged")
"""
        merge_path = Path(output_dir) / "merge_script.py"
        merge_path.write_text(merge_script, encoding="utf-8")

        proc = subprocess.run(  # noqa: S603
            [sys.executable, str(merge_path)],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"Merge failed:\n{proc.stderr[-2000:]}")

        # Step 2: Convert to GGUF using llama.cpp convert script
        # This requires llama.cpp to be installed
        try:
            convert_cmd = [
                sys.executable,
                "-m",
                "llama_cpp.tools.convert",
                str(Path(output_dir) / "merged"),
                "--outfile",
                str(out_path),
                "--outtype",
                quantization.replace("-", "_"),
            ]
            proc2 = subprocess.run(convert_cmd, capture_output=True, text=True)  # noqa: S603
            if proc2.returncode == 0:
                return str(out_path)
        except Exception as e:
            logger.debug("[GGUFConverter] llama_cpp convert failed: %s", e)

        logger.warning(
            "[GGUFConverter] GGUF conversion requires llama.cpp: "
            "pip install llama-cpp-python and use llama.cpp/convert.py manually"
        )
        return str(Path(output_dir) / "merged")  # Return merged model dir


class ModelDeployer:
    """Deploys converted GGUF model to LM Studio."""

    def deploy(self, gguf_path: str, model_name: str) -> str:
        """Copy GGUF to LM Studio models directory."""
        src = Path(gguf_path)
        if not src.exists():
            raise FileNotFoundError(f"Model file not found: {gguf_path}")

        dest_dir = _MODELS_DIR / "vetinari" / model_name
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / src.name

        import shutil

        shutil.copy2(str(src), str(dest))

        logger.info("[ModelDeployer] Deployed %s to %s", src.name, dest)
        return str(dest)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


class TrainingPipeline:
    """Orchestrates the full training lifecycle."""

    def __init__(self):
        self._curator = DataCurator()
        self._trainer = LocalTrainer()
        self._converter = GGUFConverter()
        self._deployer = ModelDeployer()

    def check_requirements(self) -> dict[str, Any]:
        """Check what training capabilities are available."""
        avail = self._trainer.check_available()
        return {
            "libraries": avail,
            "ready_for_training": avail.get("trl", False) or avail.get("unsloth", False),
            "lmstudio_models_dir": str(_MODELS_DIR),
            "models_dir_exists": _MODELS_DIR.exists(),
        }

    def run(
        self,
        base_model: str,
        task_type: str | None = None,
        min_score: float = 0.8,
        epochs: int = 3,
        output_base_dir: str = "./training_runs",
    ) -> TrainingRun:
        """Run the complete training pipeline.

        Args:
            base_model:      HuggingFace model ID (e.g., "Qwen/Qwen2.5-Coder-7B-Instruct")
            task_type:       Only train on this task type (None = all types)
            min_score:       Minimum quality score for training examples
            epochs:          Number of training epochs
            output_base_dir: Directory for training artifacts

        Returns:
            TrainingRun with status and paths.
        """
        import uuid

        run_id = f"run_{uuid.uuid4().hex[:8]}"
        run_dir = Path(output_base_dir) / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        run = TrainingRun(
            run_id=run_id,
            timestamp=datetime.now().isoformat(),
            base_model=base_model,
            task_type=task_type or "all",
            training_examples=0,
            epochs=epochs,
            success=False,
        )

        try:
            # Stage 1: Curate data
            logger.info("[TrainingPipeline] %s: Curating training data...", run_id)
            dataset_path = self._curator.curate(
                task_type=task_type,
                min_score=min_score,
                output_dir=str(run_dir),
            )

            # Count examples
            with open(dataset_path) as f:
                run.training_examples = sum(1 for _ in f)
            logger.info("[TrainingPipeline] %s: %d training examples", run_id, run.training_examples)

            if run.training_examples < 10:
                run.error = f"Insufficient training data ({run.training_examples} examples)"
                return run

            # Stage 2: Train
            logger.info("[TrainingPipeline] %s: Starting QLoRA training...", run_id)
            adapter_path = self._trainer.train_qlora(
                base_model=base_model,
                dataset_path=dataset_path,
                output_dir=str(run_dir),
                epochs=epochs,
            )
            run.adapter_path = adapter_path
            logger.info("[TrainingPipeline] %s: Training complete -> %s", run_id, adapter_path)

            # Stage 3: Convert to GGUF
            logger.info("[TrainingPipeline] %s: Converting to GGUF...", run_id)
            gguf_path = self._converter.convert(
                base_model=base_model,
                adapter_path=adapter_path,
                output_dir=str(run_dir),
            )

            # Stage 4: Deploy
            model_name = f"vetinari-{task_type or 'general'}-{base_model.split('/')[-1]}"
            deployed_path = self._deployer.deploy(gguf_path, model_name)
            run.output_model_path = deployed_path

            run.success = True
            logger.info("[TrainingPipeline] %s: Complete! Model at %s", run_id, deployed_path)

        except Exception as e:
            run.error = str(e)
            logger.error("[TrainingPipeline] %s: Failed: %s", run_id, e)

        # Save run record
        with open(run_dir / "run.json", "w") as f:
            import dataclasses

            json.dump(dataclasses.asdict(run), f, indent=2)

        return run
