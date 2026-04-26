"""Tests for vetinari.config.settings — VetinariSettings validation.

Covers the KV cache quantization type validators added in Session 15.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from vetinari.config.settings import VetinariSettings


class TestKVCacheTypeValidator:
    """Tests for local_cache_type_k / local_cache_type_v validators."""

    def test_defaults_to_f16(self) -> None:
        s = VetinariSettings()
        assert s.local_cache_type_k == "f16"
        assert s.local_cache_type_v == "f16"

    @pytest.mark.parametrize("quant_type", ["f16", "q8_0", "q4_0"])
    def test_valid_k_types_accepted(self, quant_type: str) -> None:
        s = VetinariSettings(local_cache_type_k=quant_type)
        assert s.local_cache_type_k == quant_type

    @pytest.mark.parametrize("quant_type", ["f16", "q8_0", "q4_0"])
    def test_valid_v_types_accepted(self, quant_type: str) -> None:
        s = VetinariSettings(local_cache_type_v=quant_type)
        assert s.local_cache_type_v == quant_type

    @pytest.mark.parametrize("invalid", ["q4_1", "bf16", "q2_k", "Q8_0", ""])
    def test_invalid_k_type_raises(self, invalid: str) -> None:
        with pytest.raises(ValidationError):
            VetinariSettings(local_cache_type_k=invalid)

    @pytest.mark.parametrize("invalid", ["q4_1", "bf16", "q2_k", "Q8_0", ""])
    def test_invalid_v_type_raises(self, invalid: str) -> None:
        with pytest.raises(ValidationError):
            VetinariSettings(local_cache_type_v=invalid)

    def test_k_and_v_can_differ(self) -> None:
        s = VetinariSettings(local_cache_type_k="q4_0", local_cache_type_v="q8_0")
        assert s.local_cache_type_k == "q4_0"
        assert s.local_cache_type_v == "q8_0"
