from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional

import torch

from demucs.api import LoadAudioError, Separator, save_audio

from .audio_cleanup import StemAnalysis, cleanup_stems


ProgressCallback = Callable[[str, float], None]


DEFAULT_MODEL = "htdemucs_ft"
DEFAULT_SHIFTS = 4
DEFAULT_OVERLAP = 0.50


@dataclass
class SeparationResult:
    vocals_path: Path
    instrumental_path: Path
    samplerate: int
    analysis: Dict[str, StemAnalysis]


def separate_and_enhance(
    input_path: Path,
    output_dir: Path,
    progress: Optional[ProgressCallback] = None,
) -> SeparationResult:
    """Run Demucs two-stem separation followed by conservative cleanup."""
    progress = progress or (lambda _stage, _value: None)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_name = os.environ.get("DEMUCS_WEB_MODEL", DEFAULT_MODEL)
    device = os.environ.get("DEMUCS_WEB_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
    shifts = _env_int("DEMUCS_WEB_SHIFTS", DEFAULT_SHIFTS)
    jobs = _env_int("DEMUCS_WEB_JOBS", 0)
    segment = _env_int("DEMUCS_WEB_SEGMENT", 0) or None
    overlap = _env_float("DEMUCS_WEB_OVERLAP", DEFAULT_OVERLAP)

    progress("Separating audio...", 0.08)
    separator = Separator(
        model=model_name,
        device=device,
        shifts=shifts,
        split=True,
        overlap=overlap,
        segment=segment,
        jobs=jobs,
        progress=False,
        callback=_separation_callback(progress),
        callback_arg={"shifts": shifts},
    )

    try:
        origin, stems = separator.separate_audio_file(input_path)
    except LoadAudioError:
        raise
    except Exception as exc:
        raise RuntimeError(f"Separation failed: {exc}") from exc

    if "vocals" not in stems:
        raise RuntimeError("The selected model did not return a vocals stem.")

    vocals = stems["vocals"].detach().cpu().float()
    instrumental = _build_instrumental(origin.detach().cpu().float(), stems)

    vocals, instrumental, analysis = cleanup_stems(
        vocals,
        instrumental,
        separator.samplerate,
        progress=progress,
    )

    progress("Finalizing files...", 0.94)
    vocals_path = output_dir / "vocals.wav"
    instrumental_path = output_dir / "instrumental.wav"
    save_audio(vocals, vocals_path, samplerate=separator.samplerate, clip="rescale", bits_per_sample=16)
    save_audio(
        instrumental,
        instrumental_path,
        samplerate=separator.samplerate,
        clip="rescale",
        bits_per_sample=16,
    )
    progress("Finalizing files...", 1.0)
    return SeparationResult(vocals_path, instrumental_path, separator.samplerate, analysis)


def _build_instrumental(origin: torch.Tensor, stems: Dict[str, torch.Tensor]) -> torch.Tensor:
    parts = [stem.detach().cpu().float() for name, stem in stems.items() if name != "vocals"]
    if parts:
        instrumental = torch.zeros_like(parts[0])
        for stem in parts:
            instrumental = instrumental + stem
        return instrumental
    return origin - stems["vocals"].detach().cpu().float()


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if not value:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _separation_callback(progress: ProgressCallback):
    def callback(info: dict) -> None:
        if info.get("state") != "end":
            return
        models = max(1, int(info.get("models") or 1))
        shifts = max(1, int(info.get("shifts") or 1))
        model_idx = max(0, int(info.get("model_idx_in_bag") or 0))
        shift_idx = max(0, int(info.get("shift_idx") or 0))
        audio_length = max(1, int(info.get("audio_length") or 1))
        offset = max(0, int(info.get("segment_offset") or 0))
        segment_fraction = min(1.0, offset / audio_length)
        unit = ((model_idx * shifts) + shift_idx + segment_fraction) / (models * shifts)
        progress("Separating audio...", min(0.60, 0.08 + 0.52 * unit))

    return callback
