"""Regression proof for Session 34I2 lifecycle, deploy, and health contracts."""

from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]


def test_litestar_lifespan_delegates_to_shared_lifecycle() -> None:
    """The mounted app lifespan uses vetinari.web.lifespan, not a local no-op."""
    from vetinari.web.litestar_app import _lifespan

    calls: list[tuple[str, object]] = []
    app = object()

    @asynccontextmanager
    async def fake_lifespan(received_app: object):
        calls.append(("start", received_app))
        yield
        calls.append(("stop", received_app))

    async def run() -> None:
        with patch("vetinari.web.lifespan.vetinari_lifespan", fake_lifespan):
            async with _lifespan(app):
                calls.append(("inside", app))

    asyncio.run(run())

    assert calls == [("start", app), ("inside", app), ("stop", app)]


def test_model_pool_warmups_are_observable_and_stoppable() -> None:
    """Model warm-up workers expose status and join on bounded shutdown."""
    from vetinari.models.model_pool import ModelPool, stop_all_model_warmups

    pool = ModelPool({"models": []})
    pool.models = [{"id": "model-a"}]

    def slow_load(_model_id: str) -> None:
        time.sleep(0.05)

    mock_adapter = MagicMock()
    mock_adapter.load_model.side_effect = slow_load

    with patch("vetinari.adapters.llama_cpp_adapter.LlamaCppProviderAdapter", return_value=mock_adapter):
        pool.warm_up_primary()
        status = pool.get_warmup_status()
        assert status["active"] is True
        results = stop_all_model_warmups(timeout=1)

    assert results
    assert pool.get_warmup_status()["active"] is False
    mock_adapter.unload_model.assert_called_once_with("model-a")


def test_lifespan_stops_model_warmups_before_unload() -> None:
    """Shutdown must stop warm-up workers before clearing loaded GGUF models."""
    source = (ROOT / "vetinari" / "web" / "lifespan.py").read_text(encoding="utf-8")

    assert source.index("stop_all_model_warmups") < source.index("LlamaCppProviderAdapter.unload_all")


def test_deploy_healthcheck_uses_readiness_endpoint() -> None:
    dockerfile = (ROOT / "deploy" / "Dockerfile").read_text(encoding="utf-8")

    assert "/ready" in dockerfile
    assert "/api/v1/health').raise_for_status()" not in dockerfile


def test_otel_ports_are_localhost_bound() -> None:
    compose = (ROOT / "deploy" / "docker-compose.otel.yaml").read_text(encoding="utf-8")

    assert '"127.0.0.1:4317:4317"' in compose
    assert '"127.0.0.1:16686:16686"' in compose
    assert '"4317:4317"' not in compose
    assert '"16686:16686"' not in compose


def test_vllm_helper_adopts_pid_and_defaults_localhost() -> None:
    script = (ROOT / "start-vllm-wsl.ps1").read_text(encoding="utf-8")

    assert '[string]$VllmHost = "127.0.0.1"' in script
    assert "Adopted existing vLLM process" in script
    assert "vllm-$Port.pid" in script
    assert "vllm-$Port.log" in script
    assert "if ($ForceRestart -or -not (Test-VllmEndpoint -Endpoint $endpoint))" in script
    assert 'pgrep -f -- "`$MARKER"' in script
    assert "stop-vllm-wsl.ps1" not in script
    assert "StartupTimeoutSeconds must be greater than zero" in script
    assert '[string]$PrefixCachingHashAlgo = "sha256"' in script
    assert "--enable-prefix-caching --prefix-caching-hash-algo" in script
    assert "--no-enable-prefix-caching" in script
    assert "VETINARI_VLLM_PREFIX_CACHING_ENABLED" in script
    assert "VETINARI_VLLM_CACHE_SALT" in script
