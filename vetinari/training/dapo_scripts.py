"""Training script builders for SimPO and DAPO-reward/DPO-loss subprocess execution.

Each function returns a Python source string that is written to disk and
executed in a subprocess so that heavy ML dependencies (torch, trl, peft,
transformers) are only loaded in the child process, not in the main server.

Naming honesty (2026-04-25): the "DAPO" path here uses DAPO-derived rewards
(see ``vetinari/training/dapo_rewards.py``) but the *training loss* is DPO
via ``trl.DPOTrainer`` with reward-weighted preference pairs. The script-
builder name ``build_dapo_reward_dpo_script`` reflects this hybrid. A future
real DAPO implementation (verl- or trl-DAPOTrainer-based) would warrant a
new function under a different name; the current path is honest about being
DAPO-reward-ranked DPO training.
"""

from __future__ import annotations

import json as _json


def _revision_literal(model_revision: str | None) -> str:
    return _json.dumps(str(model_revision)) if model_revision else "None"


def _local_files_only_literal(model_revision: str | None) -> str:
    return "False" if model_revision else "True"


def build_simpo_script(
    model: str,
    dpo_path: str,
    output_dir: str,
    epochs: int = 1,
    model_revision: str | None = None,
) -> str:
    """Return a Python script string for SimPO (reference-free alignment) training.

    The generated script uses trl's DPOTrainer with ``loss_type="simpo"``
    and 4-bit QLoRA via bitsandbytes. All string parameters are JSON-encoded
    to prevent injection via model names or paths that contain special characters.

    Args:
        model: Path or identifier of the base model to fine-tune.
        dpo_path: Path to the JSONL preference dataset (chosen/rejected pairs).
        output_dir: Directory for saving the LoRA adapter and training artefacts.
        epochs: Number of training epochs.
        model_revision: Optional immutable Hugging Face revision for remote base models.

    Returns:
        Python source code as a string, ready to be written to a ``.py`` file
        and executed with ``subprocess.run([sys.executable, script_path])``.
    """
    model_literal = _json.dumps(str(model))
    revision_literal = _revision_literal(model_revision)
    local_files_only_literal = _local_files_only_literal(model_revision)
    return f"""import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from trl import DPOTrainer, DPOConfig
from datasets import load_dataset

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    {model_literal},
    revision={revision_literal},
    local_files_only={local_files_only_literal},
    quantization_config=bnb_config,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(
    {model_literal},
    revision={revision_literal},
    local_files_only={local_files_only_literal},
)
tokenizer.pad_token = tokenizer.eos_token

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    task_type="CAUSAL_LM",
    use_dora=True,
)
model = get_peft_model(model, lora_config)

dataset = load_dataset("json", data_files={_json.dumps(str(dpo_path))}, split="train")

trainer = DPOTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    args=DPOConfig(
        output_dir={_json.dumps(str(output_dir))},
        loss_type="simpo",
        beta=2.0,
        num_train_epochs={int(epochs)},
        per_device_train_batch_size=1,
        learning_rate=5e-6,
        logging_steps=10,
        save_strategy="epoch",
    ),
)
trainer.train()
model.save_pretrained({_json.dumps(str(output_dir) + "/lora_adapter")})
"""


def build_dapo_reward_dpo_script(
    model: str,
    dapo_dataset_path: str,
    output_dir: str,
    epochs: int = 1,
    model_revision: str | None = None,
) -> str:
    """Return a Python script string for DAPO-reward-weighted DPO training.

    The generated script trains with ``trl.DPOTrainer`` (DPO loss, low beta=0.1)
    over preference pairs whose chosen/rejected ranking comes from DAPO rewards
    (see ``vetinari/training/dapo_rewards.py::compute_dapo_reward``). It is NOT
    a real DAPO trainer — the loss is DPO, the *ranking signal* is DAPO. All
    string parameters are JSON-encoded to prevent injection. 4-bit QLoRA via
    bitsandbytes for memory.

    Renamed 2026-04-25 from ``build_dapo_script`` to make the DAPO-reward / DPO-
    loss hybrid explicit at the call site (SHARD-01-revised).

    Args:
        model: Path or identifier of the base model to fine-tune.
        dapo_dataset_path: Path to the JSONL preference dataset produced by
            the DAPO reward scoring step (chosen/rejected pairs).
        output_dir: Directory for saving the LoRA adapter and training artefacts.
        epochs: Number of training epochs.
        model_revision: Optional immutable Hugging Face revision for remote base models.

    Returns:
        Python source code as a string, ready to be written to a ``.py`` file
        and executed with ``subprocess.run([sys.executable, script_path])``.
    """
    model_literal = _json.dumps(str(model))
    revision_literal = _revision_literal(model_revision)
    local_files_only_literal = _local_files_only_literal(model_revision)
    return f"""import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from trl import DPOTrainer, DPOConfig
from datasets import load_dataset

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    {model_literal},
    revision={revision_literal},
    local_files_only={local_files_only_literal},
    quantization_config=bnb_config,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(
    {model_literal},
    revision={revision_literal},
    local_files_only={local_files_only_literal},
)
tokenizer.pad_token = tokenizer.eos_token

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    task_type="CAUSAL_LM",
    use_dora=True,
)
model = get_peft_model(model, lora_config)

dataset = load_dataset("json", data_files={_json.dumps(str(dapo_dataset_path))}, split="train")

trainer = DPOTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    args=DPOConfig(
        output_dir={_json.dumps(str(output_dir))},
        beta=0.1,
        num_train_epochs={int(epochs)},
        per_device_train_batch_size=1,
        learning_rate=1e-6,
        logging_steps=10,
        save_strategy="epoch",
    ),
)
trainer.train()
model.save_pretrained({_json.dumps(str(output_dir) + "/lora_adapter")})
"""
