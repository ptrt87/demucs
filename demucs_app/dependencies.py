from __future__ import annotations

import importlib.util
from typing import Iterable


RECOMMENDED_PYTHON = "3.10"
TORCH_PACKAGE = "torch==2.0.1"
TORCHAUDIO_PACKAGE = "torchaudio==2.0.2"
WINDOWS_CPU_INSTALL = (
    f"python -m pip install {TORCH_PACKAGE} {TORCHAUDIO_PACKAGE} "
    "--index-url https://download.pytorch.org/whl/cpu"
)
WINDOWS_CUDA118_INSTALL = (
    f"python -m pip install {TORCH_PACKAGE} {TORCHAUDIO_PACKAGE} "
    "--index-url https://download.pytorch.org/whl/cu118"
)


def missing_runtime_packages() -> list[str]:
    return [name for name in ("torch", "torchaudio") if importlib.util.find_spec(name) is None]


def format_missing_dependency_error(missing: str | Iterable[str]) -> str:
    if isinstance(missing, str):
        missing_packages = [missing]
    else:
        missing_packages = list(missing)
    if not missing_packages:
        missing_packages = ["torch", "torchaudio"]
    names = ", ".join(sorted(set(missing_packages)))
    return (
        f"Missing required audio dependency: {names}. "
        f"Use Python {RECOMMENDED_PYTHON} and install the tested PyTorch stack with: "
        f"{WINDOWS_CPU_INSTALL}. "
        f"For NVIDIA CUDA 11.8, use: {WINDOWS_CUDA118_INSTALL}."
    )
