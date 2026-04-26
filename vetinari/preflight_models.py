"""Data models for startup preflight checks."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class HardwareInfo:
    """Hardware detection results relevant to dependency recommendations.

    Attributes:
        has_nvidia_gpu: Whether an NVIDIA GPU was detected.
        gpu_name: Human-readable GPU name (empty if no GPU).
        vram_gb: Dedicated VRAM in gigabytes (0.0 if no GPU).
        has_cuda_toolkit: Whether ``nvcc`` is on PATH.
        torch_has_cuda: Whether installed torch was built with CUDA.
        llama_cpp_has_cuda: Whether llama-cpp-python was built with CUDA.
        has_docker: Whether ``docker`` is on PATH.
    """

    has_nvidia_gpu: bool = False
    gpu_name: str = ""
    vram_gb: float = 0.0
    has_cuda_toolkit: bool = False
    torch_has_cuda: bool = False
    llama_cpp_has_cuda: bool = False
    has_docker: bool = False

    def __repr__(self) -> str:
        return "HardwareInfo(...)"


@dataclass(frozen=True, slots=True)
class DependencyGroupStatus:
    """Status of a single optional dependency group.

    Attributes:
        name: Human-readable group name (e.g. "Cloud LLM Providers").
        extra: pip extra name (e.g. "cloud").
        description: What this group enables.
        import_names: Python module names to probe.
        installed: Modules that were found.
        missing: Modules that were not found.
        recommended: Whether this group is recommended for the detected hardware.
    """

    name: str
    extra: str
    description: str
    import_names: tuple[str, ...]
    installed: tuple[str, ...] = ()
    missing: tuple[str, ...] = ()
    recommended: bool = False

    @property
    def is_complete(self) -> bool:
        """All modules in the group are installed."""
        return len(self.missing) == 0

    @property
    def is_partial(self) -> bool:
        """Some but not all modules are installed."""
        return len(self.installed) > 0 and len(self.missing) > 0

    def __repr__(self) -> str:
        return f"DependencyGroupStatus(name={self.name!r}, extra={self.extra!r})"


@dataclass(frozen=True, slots=True)
class DependencyReadinessSpec:
    """Static definition for one package in the dependency readiness matrix."""

    package: str
    import_names: tuple[str, ...]
    channel: str
    expected: str
    description: str
    install_command: str
    distribution: str | None = None
    gpu_expected: str | None = None

    def __repr__(self) -> str:
        return f"DependencyReadinessSpec(package={self.package!r}, channel={self.channel!r})"


@dataclass(frozen=True, slots=True)
class DependencyReadiness:
    """Resolved readiness state for one package in the current environment."""

    package: str
    import_name: str | None
    channel: str
    expected: str
    description: str
    install_command: str
    distribution: str | None
    installed: bool
    version: str | None
    runtime_verified: bool | None
    status: str
    detail: str

    def __repr__(self) -> str:
        return f"DependencyReadiness(package={self.package!r}, status={self.status!r})"


@dataclass(slots=True)
class PreflightReport:
    """Complete preflight check results.

    Attributes:
        hardware: Detected hardware info.
        groups: Status of each dependency group.
        system_actions: Non-pip actions the user should take (e.g. install CUDA toolkit).
    """

    hardware: HardwareInfo
    groups: list[DependencyGroupStatus] = field(default_factory=list)
    dependency_matrix: list[DependencyReadiness] = field(default_factory=list)
    system_actions: list[str] = field(default_factory=list)

    def __repr__(self) -> str:
        return "PreflightReport(...)"
