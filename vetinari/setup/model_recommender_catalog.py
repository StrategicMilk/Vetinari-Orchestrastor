"""Static setup model recommendation catalogs."""

from __future__ import annotations

from typing import Any

from vetinari.setup.model_recommender_types import SetupModelRecommendation

_VRAM_TIERS: list[dict[str, Any]] = [
    {
        "min_vram_gb": 0.0,
        "max_vram_gb": 4.0,
        "label": "CPU-only / < 4 GB VRAM",
        "models": [
            SetupModelRecommendation(
                name="Qwen 2.5 1.5B Q4_K_M",
                repo_id="Qwen/Qwen2.5-1.5B-Instruct-GGUF",
                filename="qwen2.5-1.5b-instruct-q4_k_m.gguf",
                size_gb=1.1,
                quantization="Q4_K_M",
                parameter_count="1.5B",
                reason="Smallest capable model — fits in RAM for CPU inference",
                is_primary=True,
            ),
            SetupModelRecommendation(
                name="TinyLlama 1.1B Q4_K_M",
                repo_id="TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
                filename="tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
                size_gb=0.7,
                quantization="Q4_K_M",
                parameter_count="1.1B",
                reason="Ultra-lightweight fallback for very constrained systems",
            ),
        ],
    },
    {
        "min_vram_gb": 4.0,
        "max_vram_gb": 8.0,
        "label": "4-8 GB VRAM",
        "models": [
            SetupModelRecommendation(
                name="Qwen 2.5 7B Q4_K_M",
                repo_id="Qwen/Qwen2.5-7B-Instruct-GGUF",
                filename="qwen2.5-7b-instruct-q4_k_m.gguf",
                size_gb=4.4,
                quantization="Q4_K_M",
                parameter_count="7B",
                reason="Best quality-per-VRAM at 4-bit quantization for 8GB cards",
                is_primary=True,
            ),
            SetupModelRecommendation(
                name="Mistral 7B v0.3 Q4_K_M",
                repo_id="bartowski/Mistral-7B-Instruct-v0.3-GGUF",
                filename="Mistral-7B-Instruct-v0.3-Q4_K_M.gguf",
                size_gb=4.1,
                quantization="Q4_K_M",
                parameter_count="7B",
                reason="Strong instruction following, wide community support",
            ),
        ],
    },
    {
        "min_vram_gb": 8.0,
        "max_vram_gb": 16.0,
        "label": "8-16 GB VRAM",
        "models": [
            SetupModelRecommendation(
                name="Qwen 2.5 7B Q6_K",
                repo_id="Qwen/Qwen2.5-7B-Instruct-GGUF",
                filename="qwen2.5-7b-instruct-q6_k.gguf",
                size_gb=6.0,
                quantization="Q6_K",
                parameter_count="7B",
                reason="Higher quantization — better output quality with 12+ GB VRAM",
                is_primary=True,
            ),
            SetupModelRecommendation(
                name="Llama 3.1 8B Q6_K",
                repo_id="bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
                filename="Meta-Llama-3.1-8B-Instruct-Q6_K.gguf",
                size_gb=6.6,
                quantization="Q6_K",
                parameter_count="8B",
                reason="Excellent code generation, strong reasoning",
            ),
        ],
    },
    {
        "min_vram_gb": 16.0,
        "max_vram_gb": 24.0,
        "label": "16-24 GB VRAM",
        "models": [
            SetupModelRecommendation(
                name="Qwen 2.5 14B Q4_K_M",
                repo_id="Qwen/Qwen2.5-14B-Instruct-GGUF",
                filename="qwen2.5-14b-instruct-q4_k_m.gguf",
                size_gb=8.7,
                quantization="Q4_K_M",
                parameter_count="14B",
                reason="14B parameters at 4-bit — substantial quality jump over 7B",
                is_primary=True,
            ),
            SetupModelRecommendation(
                name="Codestral 22B Q4_K_M",
                repo_id="bartowski/Codestral-22B-v0.1-GGUF",
                filename="Codestral-22B-v0.1-Q4_K_M.gguf",
                size_gb=12.9,
                quantization="Q4_K_M",
                parameter_count="22B",
                reason="Specialist coding model, excellent for code-heavy workloads",
            ),
        ],
    },
    {
        "min_vram_gb": 24.0,
        "max_vram_gb": 999.0,
        "label": "24+ GB VRAM",
        "models": [
            SetupModelRecommendation(
                name="Qwen 2.5 14B Q6_K",
                repo_id="Qwen/Qwen2.5-14B-Instruct-GGUF",
                filename="qwen2.5-14b-instruct-q6_k.gguf",
                size_gb=12.0,
                quantization="Q6_K",
                parameter_count="14B",
                reason="14B at 6-bit — best quality for 24GB+ cards",
                is_primary=True,
            ),
            SetupModelRecommendation(
                name="Qwen 2.5 32B Q4_K_M",
                repo_id="Qwen/Qwen2.5-32B-Instruct-GGUF",
                filename="qwen2.5-32b-instruct-q4_k_m.gguf",
                size_gb=19.8,
                quantization="Q4_K_M",
                parameter_count="32B",
                reason="32B parameters — top-tier reasoning and instruction following",
            ),
        ],
    },
]


_VLLM_MODEL_TIERS: list[dict[str, Any]] = [
    {
        "min_vram_gb": 8.0,
        "max_vram_gb": 16.0,
        "label": "8-16 GB VRAM (vLLM)",
        "models": [
            SetupModelRecommendation(
                name="Qwen 2.5 7B AWQ",
                repo_id="Qwen/Qwen2.5-7B-Instruct-AWQ",
                filename="",  # vLLM loads from HF repo directly
                size_gb=4.5,
                quantization="AWQ",
                parameter_count="7B",
                reason="AWQ quantized — faster throughput on vLLM than GGUF",
                is_primary=True,
                model_format="awq",
                backend="vllm",
                gpu_only=True,
            ),
            SetupModelRecommendation(
                name="Qwen 2.5 Coder 7B SafeTensors",
                repo_id="Qwen/Qwen2.5-Coder-7B-Instruct",
                filename="",
                size_gb=14.0,
                quantization="BF16",
                parameter_count="7B",
                reason="Native SafeTensors snapshot for vLLM when VRAM allows full precision",
                model_format="safetensors",
                backend="vllm",
                gpu_only=True,
                best_for=("coding", "review", "documentation"),
            ),
        ],
    },
    {
        "min_vram_gb": 16.0,
        "max_vram_gb": 24.0,
        "label": "16-24 GB VRAM (vLLM)",
        "models": [
            SetupModelRecommendation(
                name="Qwen 2.5 14B AWQ",
                repo_id="Qwen/Qwen2.5-14B-Instruct-AWQ",
                filename="",
                size_gb=9.0,
                quantization="AWQ",
                parameter_count="14B",
                reason="14B AWQ — fits in 16GB+ VRAM with fast vLLM throughput",
                is_primary=True,
                model_format="awq",
                backend="vllm",
                gpu_only=True,
            ),
            SetupModelRecommendation(
                name="Codestral 22B GPTQ",
                repo_id="bartowski/Codestral-22B-v0.1-GPTQ",
                filename="",
                size_gb=13.0,
                quantization="GPTQ",
                parameter_count="22B",
                reason="Specialist coding model — GPTQ for vLLM",
                model_format="gptq",
                backend="vllm",
                gpu_only=True,
            ),
            SetupModelRecommendation(
                name="Qwen 2.5 14B SafeTensors",
                repo_id="Qwen/Qwen2.5-14B-Instruct",
                filename="",
                size_gb=28.0,
                quantization="BF16",
                parameter_count="14B",
                reason="Native SafeTensors snapshot for vLLM/NIM on larger GPUs",
                model_format="safetensors",
                backend="vllm",
                gpu_only=True,
                best_for=("reasoning", "planning", "research"),
            ),
        ],
    },
    {
        "min_vram_gb": 24.0,
        "max_vram_gb": 999.0,
        "label": "24+ GB VRAM (vLLM)",
        "models": [
            SetupModelRecommendation(
                name="Qwen 2.5 32B AWQ",
                repo_id="Qwen/Qwen2.5-32B-Instruct-AWQ",
                filename="",
                size_gb=18.0,
                quantization="AWQ",
                parameter_count="32B",
                reason="32B AWQ — top-tier reasoning, fits in 24GB+ VRAM on vLLM",
                is_primary=True,
                model_format="awq",
                backend="vllm",
                gpu_only=True,
            ),
            SetupModelRecommendation(
                name="Qwen 2.5 14B GPTQ",
                repo_id="Qwen/Qwen2.5-14B-Instruct-GPTQ-Int4",
                filename="",
                size_gb=9.0,
                quantization="GPTQ",
                parameter_count="14B",
                reason="14B GPTQ — alternative quantization for vLLM",
                model_format="gptq",
                backend="vllm",
                gpu_only=True,
            ),
            SetupModelRecommendation(
                name="Qwen 2.5 14B SafeTensors",
                repo_id="Qwen/Qwen2.5-14B-Instruct",
                filename="",
                size_gb=28.0,
                quantization="BF16",
                parameter_count="14B",
                reason="Native SafeTensors snapshot for vLLM/NIM full-precision serving",
                model_format="safetensors",
                backend="vllm",
                gpu_only=True,
                best_for=("reasoning", "planning", "research"),
            ),
        ],
    },
]


# ── CPU Offload Models (llama-cpp only, GGUF, larger than VRAM) ───────────────
# These models are too large for most GPUs but can run via llama-cpp's CPU
# offload (partial GPU layers + RAM).  Slower but dramatically more capable.

_CPU_OFFLOAD_MODELS: list[SetupModelRecommendation] = [
    SetupModelRecommendation(
        name="Qwen 2.5 72B Q4_K_M (CPU offload)",
        repo_id="bartowski/Qwen2.5-72B-Instruct-GGUF",
        filename="Qwen2.5-72B-Instruct-Q4_K_M.gguf",
        size_gb=42.0,
        quantization="Q4_K_M",
        parameter_count="72B",
        reason="72B model — requires VRAM+RAM split via llama-cpp CPU offload, slower but top-tier reasoning",
        model_format="gguf",
        backend="llama_cpp",
        gpu_only=False,
    ),
    SetupModelRecommendation(
        name="Llama 3.3 70B Q4_K_M (CPU offload)",
        repo_id="bartowski/Llama-3.3-70B-Instruct-GGUF",
        filename="Llama-3.3-70B-Instruct-Q4_K_M.gguf",
        size_gb=40.0,
        quantization="Q4_K_M",
        parameter_count="70B",
        reason="70B flagship — requires CPU offload, excellent for complex reasoning",
        model_format="gguf",
        backend="llama_cpp",
        gpu_only=False,
    ),
]
