"""Vetinari Packaging CLI — thin facade over split command modules.

Delegates all implementation to:
- ``cli_packaging_data``    — init wizard, hardware detection, model tiers,
                              _print_header, _print_check
- ``cli_packaging_doctor``  — diagnostic suite (cmd_doctor)
- ``cli_packaging_models``  — model management + forget/config/resume/quick-action

All public symbols are re-exported so ``from vetinari.cli_packaging import cmd_init``
and similar patterns continue to work unchanged.
"""

from __future__ import annotations

from typing import Any

from vetinari.cli_packaging_data import (
    _CHECK_FAIL,
    _CHECK_INFO,
    _CHECK_PASS,
    _CHECK_WARN,
    _MODEL_TIERS,
    DEFAULT_USER_MODELS_DIR,
    _detect_hardware,
    _get_recommended_models,
    _print_check,
    _print_header,
    cmd_init,
)
from vetinari.cli_packaging_doctor import cmd_doctor
from vetinari.cli_packaging_models import (
    _download_with_progress,
    _find_models_dir,
    _guess_family,
    _guess_quantization,
    _models_download,
    _models_files,
    _models_info,
    _models_list,
    _models_recommend,
    _models_remove,
    _models_scan,
    _models_status,
    _verify_sha256,
    cmd_config_reload,
    cmd_forget,
    cmd_models,
    cmd_quick_action,
    cmd_resume,
)

__all__ = [
    # data
    "DEFAULT_USER_MODELS_DIR",
    "_CHECK_FAIL",
    "_CHECK_INFO",
    "_CHECK_PASS",
    "_CHECK_WARN",
    "_MODEL_TIERS",
    "_detect_hardware",
    # models
    "_download_with_progress",
    "_find_models_dir",
    "_get_recommended_models",
    "_guess_family",
    "_guess_quantization",
    "_models_download",
    "_models_files",
    "_models_info",
    "_models_list",
    "_models_recommend",
    "_models_remove",
    "_models_scan",
    "_models_status",
    "_print_check",
    "_print_header",
    # registration
    "_register_packaging_commands",
    "_verify_sha256",
    "cmd_config_reload",
    # doctor
    "cmd_doctor",
    "cmd_forget",
    "cmd_init",
    "cmd_models",
    "cmd_quick_action",
    "cmd_resume",
]


def _register_packaging_commands(subparsers: Any) -> None:
    """Register init, doctor, models, forget, config, resume, and quick-action subcommands.

    Adds the following subparsers to the CLI argument parser:

    * ``init``    — first-run setup wizard (``--skip-download`` flag).
    * ``doctor``  — diagnostic report (``--json`` flag).
    * ``models``  — model management (positional action + ``--repo``,
      ``--filename``, ``--name`` options).
    * ``forget``  — purge learned data for a project (``--project`` required).
    * ``config``  — hot-reload settings (``reload`` action).
    * ``resume``  — resume a plan from checkpoint (positional ``plan_id``).
    * ``explain``, ``test``, ``fix`` — quick single-file actions.

    Args:
        subparsers: The ``argparse`` subparsers action group returned by
            ``parser.add_subparsers()``.
    """
    # ── init ──────────────────────────────────────────────────────────────────
    p_init = subparsers.add_parser(
        "init",
        help="First-run setup wizard — detect hardware, select and download a model",
    )
    p_init.add_argument(
        "--skip-download",
        action="store_true",
        default=False,
        dest="skip_download",
        help="Skip the model download step (print download URL instead)",
    )

    # ── doctor ────────────────────────────────────────────────────────────────
    p_doctor = subparsers.add_parser(
        "doctor",
        help="Run diagnostic checks and report system health",
    )
    p_doctor.add_argument(
        "--json",
        action="store_true",
        default=False,
        dest="json",
        help="Emit machine-readable JSON output instead of formatted text",
    )

    # ── models ────────────────────────────────────────────────────────────────
    p_models = subparsers.add_parser(
        "models",
        help="Manage local and Hugging Face model files",
    )
    p_models.add_argument(
        "models_action",
        choices=["list", "files", "download", "status", "remove", "info", "recommend", "scan", "check"],
        help=(
            "list: show all local models | "
            "files: list downloadable repo artifacts | "
            "download: fetch a model from HuggingFace | "
            "status: inspect a persisted download | "
            "remove: delete a model | "
            "info: show model metadata | "
            "recommend: suggest models for detected VRAM | "
            "scan: discover .gguf/.awq on disk | "
            "check: check for newer, better models"
        ),
    )
    p_models.add_argument(
        "--repo",
        default=None,
        help="HuggingFace repo ID, e.g. TheBloke/Mistral-7B-Instruct-v0.2-GGUF (used with download)",
    )
    p_models.add_argument(
        "--filename",
        default=None,
        help="GGUF filename within the repo, e.g. mistral-7b-instruct-v0.2.Q4_K_M.gguf (llama.cpp downloads)",
    )
    p_models.add_argument(
        "--backend",
        default="auto",
        choices=["auto", "llama_cpp", "vllm", "nim"],
        help="Target backend; auto prefers native snapshots unless a GGUF filename/format is requested",
    )
    p_models.add_argument(
        "--format",
        dest="model_format",
        default=None,
        choices=["gguf", "safetensors", "awq", "gptq"],
        help="Model artifact format to list or download; native defaults to safetensors",
    )
    p_models.add_argument(
        "--revision",
        default=None,
        help="Immutable revision or tag to resolve before download",
    )
    p_models.add_argument("--download-id", default=None, help="Download id for models status")
    p_models.add_argument("--objective", default=None, help="Filter by objective/category, e.g. coding, chat, reasoning")
    p_models.add_argument("--family", default=None, help="Filter by model family, e.g. qwen, llama, mistral")
    p_models.add_argument("--quantization", default=None, help="Filter by quantization, e.g. Q4_K_M, AWQ, GPTQ")
    p_models.add_argument("--file-type", default=None, help="Filter by file type, e.g. gguf or safetensors")
    p_models.add_argument("--min-size-gb", type=float, default=None, help="Minimum artifact size in GB")
    p_models.add_argument("--max-size-gb", type=float, default=None, help="Maximum artifact size in GB")
    p_models.add_argument("--vram-gb", type=int, default=32, help="VRAM budget used when listing repo files")
    p_models.add_argument(
        "--name",
        default=None,
        help="Partial or full filename to match (used with remove and info)",
    )

    # ── forget ────────────────────────────────────────────────────────────────
    p_forget = subparsers.add_parser("forget", help="Purge all learned data for a project")
    p_forget.add_argument("--project", required=True, help="Project name to forget")

    # ── config ────────────────────────────────────────────────────────────────
    p_config = subparsers.add_parser("config", help="Configuration management (reload)")
    p_config.add_argument("config_action", choices=["reload"], help="reload: hot-reload settings")

    # ── resume ────────────────────────────────────────────────────────────────
    p_resume = subparsers.add_parser("resume", help="Resume interrupted plan execution")
    p_resume.add_argument("plan_id", help="Plan ID to resume from checkpoint")

    # ── Quick action commands (explain, test, fix) ─────────────────────────────
    for qaction in ("explain", "test", "fix"):
        p_qa = subparsers.add_parser(qaction, help=f"{qaction.capitalize()} a file")
        p_qa.add_argument("file", help="File path to operate on")
        p_qa.set_defaults(quick_action=qaction)
