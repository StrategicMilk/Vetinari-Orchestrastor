"""Runtime safety primitives — supported-matrix-driven preconditions.

Exposes the runtime doctor that reads config/runtime/supported_matrix.yaml and
fails closed when the detected runtime does not satisfy the matrix.
"""

from __future__ import annotations

from vetinari.runtime.runtime_doctor import (
    DoctorReport,
    RuntimeCheckResult,
    check_matrix_row,
    load_matrix,
    run_doctor,
    validate_runtime_version,
)

__all__ = [
    "DoctorReport",
    "RuntimeCheckResult",
    "check_matrix_row",
    "load_matrix",
    "run_doctor",
    "validate_runtime_version",
]
