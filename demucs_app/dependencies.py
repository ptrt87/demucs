from __future__ import annotations

import importlib.util
import shutil
import sys
from typing import Iterable


RECOMMENDED_PYTHON = "3.10"
UVR_CPU_INSTALL = 'python -m pip install "audio-separator[cpu]>=0.44.2" soundfile'
UVR_GPU_INSTALL = 'python -m pip install "audio-separator[gpu]>=0.44.2" soundfile'
WINDOWS_FFMPEG_NOTE = "Install FFmpeg and make sure ffmpeg.exe is available on PATH."
MODULE_PACKAGES = {
    "audio_separator": "audio-separator",
    "soundfile": "soundfile",
    "torch": "torch",
}


def missing_runtime_packages() -> list[str]:
    missing: list[str] = []
    if sys.version_info < (3, 10):
        missing.append("Python 3.10")
    for module_name in ("audio_separator", "soundfile", "torch"):
        if importlib.util.find_spec(module_name) is None:
            missing.append(module_name)
    if shutil.which("ffmpeg") is None:
        missing.append("ffmpeg")
    return missing


def format_missing_dependency_error(missing: str | Iterable[str]) -> str:
    if isinstance(missing, str):
        missing_packages = [missing]
    else:
        missing_packages = list(missing)
    if not missing_packages:
        missing_packages = ["audio_separator", "soundfile", "torch"]
    names = ", ".join(_friendly_name(name) for name in sorted(set(missing_packages)))
    return (
        f"Missing required Ultimate Vocal Remover dependency: {names}. "
        f"Use Python {RECOMMENDED_PYTHON} and install the UVR runtime with: {UVR_CPU_INSTALL}. "
        f"For NVIDIA GPU acceleration, use: {UVR_GPU_INSTALL}. "
        f"{WINDOWS_FFMPEG_NOTE}"
    )


def _friendly_name(name: str) -> str:
    return MODULE_PACKAGES.get(name, name)
