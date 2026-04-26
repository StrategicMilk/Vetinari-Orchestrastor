"""Training backend classes: LocalTrainer, GGUFConverter, ModelDeployer.

These classes handle the compute-intensive stages of the training pipeline:
QLoRA fine-tuning (via unsloth or trl), GGUF conversion, and deploying the
converted model to the local models directory.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import subprocess
import sys
from pathlib import Path
from types import ModuleType

from vetinari.constants import OPERATOR_MODELS_CACHE_DIR

logger = logging.getLogger(__name__)

_MODELS_DIR = Path(OPERATOR_MODELS_CACHE_DIR)

_NATIVE_MODELS_DIR = Path(OPERATOR_MODELS_CACHE_DIR) / "native"

_MISSING_MODULE = object()


def _revision_literal(model_revision: str | None) -> str:
    return json.dumps(str(model_revision)) if model_revision else "None"


def _local_files_only_literal(model_revision: str | None) -> str:
    return "False" if model_revision else "True"


def _safe_model_name(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "-" for ch in value)
    return cleaned.strip("-._") or "model"


def _is_module_available(module_name: str) -> bool:
    """Return True when a module is already loaded or discoverable without importing it."""
    existing = sys.modules.get(module_name, _MISSING_MODULE)
    if existing is None:
        return False
    if isinstance(existing, ModuleType):
        return True
    if existing is not _MISSING_MODULE:
        return True
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, AttributeError, ValueError):
        logger.debug("Module discovery failed for %s", module_name, exc_info=True)
        return False


class LocalTrainer:
    """QLoRA fine-tuning via unsloth (2x faster) or trl fallback."""

    def check_available(self) -> dict[str, bool]:
        """Check which training libraries are installed.

        Returns:
            Dict mapping library name to a bool indicating whether it can be
            discovered. Keys include: unsloth, trl, peft, transformers,
            bitsandbytes.
        """
        result = {}
        for lib in ["unsloth", "trl", "peft", "transformers", "bitsandbytes"]:
            result[lib] = _is_module_available(lib)
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
        model_revision: str | None = None,
    ) -> str:
        """Run QLoRA training.  Returns path to the saved adapter.

        VRAM budget (5090 32GB):
          7B Q4 model:  ~8GB VRAM for training
          14B Q4 model: ~14GB VRAM for training
          32B model:    too large — use cloud_trainer

        Args:
            base_model: The base model.
            dataset_path: The dataset path.
            output_dir: The output dir.
            epochs: The epochs.
            batch_size: The batch size.
            learning_rate: The learning rate.
            max_seq_length: The max seq length.
            lora_r: The lora r.
            use_unsloth: The use unsloth.
            model_revision: Optional immutable Hugging Face revision for remote base models.

        Returns:
            Filesystem path to the saved LoRA adapter directory (e.g.,
            ``<output_dir>/lora_adapter``), ready for GGUF conversion.

        Raises:
            RuntimeError: If required training libraries are not installed or if the training subprocess fails.
        """
        avail = self.check_available()
        if not avail.get("trl") and not avail.get("transformers"):
            raise RuntimeError("Training libraries not installed. Run: pip install trl peft bitsandbytes transformers")  # noqa: VET301 — user guidance string

        if use_unsloth and avail.get("unsloth"):
            return self._train_with_unsloth(
                base_model,
                dataset_path,
                output_dir,
                epochs,
                batch_size,
                learning_rate,
                max_seq_length,
                lora_r,
                model_revision,
            )
        return self._train_with_trl(
            base_model,
            dataset_path,
            output_dir,
            epochs,
            batch_size,
            learning_rate,
            max_seq_length,
            lora_r,
            model_revision,
        )

    def _train_with_unsloth(
        self,
        base_model,
        dataset_path,
        output_dir,
        epochs,
        batch_size,
        lr,
        max_seq_len,
        lora_r,
        model_revision=None,
    ) -> str:
        """Train using unsloth for 2x speed."""
        import json as _json

        # Use json.dumps() for all string parameters to prevent injection
        base_model_literal = _json.dumps(str(base_model))
        revision_literal = _revision_literal(model_revision)
        local_files_only_literal = _local_files_only_literal(model_revision)
        script = f"""
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer, TrainingArguments
import torch

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name={base_model_literal},
    revision={revision_literal},
    local_files_only={local_files_only_literal},
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
print("Training complete:", {_json.dumps(str(output_dir) + "/lora_adapter")})"""
        script_path = Path(output_dir) / "train_script.py"
        script_path.parent.mkdir(parents=True, exist_ok=True)
        script_path.write_text(script, encoding="utf-8")

        proc = subprocess.run(  # noqa: S603 - argv is controlled and shell interpolation is not used
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"Training failed:\n{proc.stderr[-2000:]}")

        return str(Path(output_dir) / "lora_adapter")

    def _train_with_trl(
        self,
        base_model,
        dataset_path,
        output_dir,
        epochs,
        batch_size,
        lr,
        max_seq_len,
        lora_r,
        model_revision=None,
    ) -> str:
        """Train using standard trl (slower than unsloth)."""
        import json as _json

        # Use json.dumps() for all string parameters to prevent injection
        base_model_literal = _json.dumps(str(base_model))
        revision_literal = _revision_literal(model_revision)
        local_files_only_literal = _local_files_only_literal(model_revision)
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
    {base_model_literal},
    revision={revision_literal},
    local_files_only={local_files_only_literal},
    quantization_config=bnb_config,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(
    {base_model_literal},
    revision={revision_literal},
    local_files_only={local_files_only_literal},
)
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
print("Training complete")"""
        script_path = Path(output_dir) / "train_trl_script.py"
        script_path.parent.mkdir(parents=True, exist_ok=True)
        script_path.write_text(script, encoding="utf-8")

        proc = subprocess.run(  # noqa: S603 - argv is controlled and shell interpolation is not used
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"Training failed:\n{proc.stderr[-2000:]}")

        return str(Path(output_dir) / "lora_adapter")


class GGUFConverter:
    """Converts trained LoRA adapter to GGUF format for local inference."""

    def convert(
        self,
        base_model: str,
        adapter_path: str,
        output_dir: str,
        quantization: str = "q4_k_m",
        model_revision: str | None = None,
    ) -> str:
        """Merge adapter with base model and convert to GGUF for local inference.

        Returns path to the .gguf file.  # noqa: VET301 — user guidance string
        Requires: llama-cpp-python (pip install llama-cpp-python)  # noqa: VET301 — user guidance string

        Args:
            base_model: The base model.
            adapter_path: The adapter path.
            output_dir: The output dir.
            quantization: The quantization.
            model_revision: Optional immutable Hugging Face revision for the base model tokenizer.

        Returns:
            Path to the converted ``.gguf`` file on success, or path to the
            merged model directory if GGUF conversion is unavailable
            (llama.cpp not installed).

        Raises:
            RuntimeError: If the adapter merge subprocess fails.
        """
        out_path = Path(output_dir) / f"model_{quantization}.gguf"

        # Step 1: Merge adapter weights
        _adapter = json.dumps(str(adapter_path))
        _outdir = json.dumps(str(output_dir))
        _basemodel = json.dumps(str(base_model))
        revision_literal = _revision_literal(model_revision)
        local_files_only_literal = _local_files_only_literal(model_revision)
        merge_script = f"""
import json as _json
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer
import torch

model = AutoPeftModelForCausalLM.from_pretrained(
    {_adapter},
    local_files_only=True,
    device_map="cpu",
    torch_dtype=torch.float16,
)
model = model.merge_and_unload()
model.save_pretrained({_outdir} + "/merged")
tokenizer = AutoTokenizer.from_pretrained(
    {_basemodel},
    revision={revision_literal},
    local_files_only={local_files_only_literal},
)
tokenizer.save_pretrained({_outdir} + "/merged")
print("Merge complete:", {_outdir} + "/merged")"""
        merge_path = Path(output_dir) / "merge_script.py"
        merge_path.write_text(merge_script, encoding="utf-8")

        proc = subprocess.run(  # noqa: S603 - argv is controlled and shell interpolation is not used
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
            proc2 = subprocess.run(convert_cmd, capture_output=True, text=True)  # noqa: S603 - argv is controlled and shell interpolation is not used
            if proc2.returncode == 0:
                return str(out_path)
        except Exception as e:
            logger.warning("[GGUFConverter] llama_cpp convert failed: %s", e)

        logger.warning(
            "[GGUFConverter] GGUF conversion requires llama.cpp: "  # noqa: VET301 — user guidance string
            "pip install llama-cpp-python and use llama.cpp/convert.py manually",  # noqa: VET301 — user guidance string
        )
        return str(Path(output_dir) / "merged")  # Return merged model dir


class ModelDeployer:
    """Deploys converted GGUF model to the local models directory."""

    def deploy(self, gguf_path: str, model_name: str) -> str:
        """Copy GGUF to local models directory.

        Args:
            gguf_path: Path to the source GGUF file.
            model_name: Subdirectory name under ``models/vetinari/`` where
                the file will be placed.

        Returns:
            Absolute path to the deployed GGUF file in the models directory.

        Raises:
            FileNotFoundError: If the GGUF model file does not exist at the given path.
        """
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

    def deploy_native(
        self,
        adapter_path: str,
        model_name: str,
        *,
        backend: str,
        model_format: str,
        base_model: str,
        base_model_revision: str | None = None,
        run_id: str = "",
        task_type: str = "general",
    ) -> dict[str, str]:
        """Copy a trained LoRA adapter into the native model artifact tree.

        The native path is used by vLLM and NIM handoff flows. It keeps the
        adapter separate from GGUF artifacts and writes enough provenance for a
        later server/package step to know the base model and exact revision.

        Args:
            adapter_path: Trained adapter file or directory to deploy.
            model_name: Logical model name used for the destination directory.
            backend: Native serving backend such as ``vllm`` or ``nim``.
            model_format: Native artifact format.
            base_model: Base model identifier for provenance.
            base_model_revision: Optional immutable base-model revision.
            run_id: Optional training run identifier.
            task_type: Task type associated with the adapter.

        Returns:
            Manifest dictionary describing the deployed adapter.

        Raises:
            FileNotFoundError: If the adapter path does not exist.
            OSError: If files cannot be copied or the manifest cannot be written.
        """
        import hashlib
        import shutil
        from datetime import datetime, timezone

        src = Path(adapter_path)
        if not src.exists():
            raise FileNotFoundError(f"Adapter path not found: {adapter_path}")

        safe_name = _safe_model_name(model_name)
        dest_dir = _NATIVE_MODELS_DIR / backend / model_format / "vetinari" / safe_name
        dest_dir.mkdir(parents=True, exist_ok=True)

        if src.is_dir():
            shutil.copytree(src, dest_dir, dirs_exist_ok=True)
            artifact_path = dest_dir
        else:
            artifact_path = dest_dir / src.name
            shutil.copy2(str(src), str(artifact_path))

        files: list[dict[str, object]] = []
        scan_root = artifact_path if artifact_path.is_dir() else artifact_path.parent
        for path in sorted(p for p in scan_root.rglob("*") if p.is_file()):
            if path.name == ".vetinari-training-manifest.json":
                continue
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            files.append({
                "filename": path.relative_to(scan_root).as_posix(),
                "size": path.stat().st_size,
                "sha256": digest,
            })

        manifest_path = dest_dir / ".vetinari-training-manifest.json"
        manifest = {
            "artifact_type": "trained_lora_adapter",
            "backend": backend,
            "format": model_format,
            "base_model": base_model,
            "base_model_revision": base_model_revision,
            "path": str(artifact_path),
            "adapter_source_path": str(src),
            "training_run_id": run_id,
            "task_type": task_type,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "files": files,
            "server_handoff": {
                "vllm_lora_path": str(artifact_path),
                "nim_model_manifest_path": str(manifest_path),
            },
        }
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

        logger.info("[ModelDeployer] Deployed native adapter to %s", artifact_path)
        return {"path": str(artifact_path), "manifest_path": str(manifest_path)}
