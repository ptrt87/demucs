from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional

from .audio_cleanup import StemAnalysis, cleanup_stems
from .dependencies import format_missing_dependency_error


ProgressCallback = Callable[[str, float], None]


class AudioPostProcessError(RuntimeError):
    """Raised when separated stems cannot be cleaned or exported."""


@dataclass(frozen=True)
class PostProcessResult:
    vocals_path: Path
    instrumental_path: Path
    samplerate: int
    analysis: Dict[str, StemAnalysis]


def clean_and_export_stems(
    vocals_path: Path,
    instrumental_path: Path,
    output_dir: Path,
    progress: Optional[ProgressCallback] = None,
) -> PostProcessResult:
    """Read UVR stems, run guarded cleanup, and write final downloadable WAVs."""
    progress = progress or (lambda _stage, _value: None)
    try:
        import soundfile as sf
        import torch
    except ModuleNotFoundError as exc:
        raise AudioPostProcessError(format_missing_dependency_error(exc.name or "soundfile")) from exc

    vocals, vocal_samplerate = _read_audio(sf, torch, vocals_path)
    instrumental, instrumental_samplerate = _read_audio(sf, torch, instrumental_path)
    if vocal_samplerate != instrumental_samplerate:
        raise AudioPostProcessError(
            "Ultimate Vocal Remover returned stems with different sample rates. "
            "Try a different UVR model or re-export the source audio as WAV first."
        )

    vocals, instrumental, analysis = cleanup_stems(vocals, instrumental, vocal_samplerate, progress=progress)

    output_dir.mkdir(parents=True, exist_ok=True)
    final_vocals = output_dir / "vocals.wav"
    final_instrumental = output_dir / "instrumental.wav"
    progress("Finalizing files...", 0.96)
    _write_audio(sf, vocals, vocal_samplerate, final_vocals)
    _write_audio(sf, instrumental, vocal_samplerate, final_instrumental)
    progress("Finalizing files...", 1.0)
    return PostProcessResult(final_vocals, final_instrumental, vocal_samplerate, analysis)


def _read_audio(sf, torch, path: Path):
    try:
        data, samplerate = sf.read(path, dtype="float32", always_2d=True)
    except Exception as exc:
        raise AudioPostProcessError(f"Could not read separated stem {path.name}: {exc}") from exc
    tensor = torch.from_numpy(data.T).contiguous().float()
    return tensor, int(samplerate)


def _write_audio(sf, tensor, samplerate: int, path: Path) -> None:
    data = tensor.detach().cpu().float().clamp(-1.0, 1.0).T.numpy()
    sf.write(path, data, samplerate, subtype="PCM_16")

