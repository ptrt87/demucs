from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional

from .audio_cleanup import StemAnalysis
from .audio_postprocess import clean_and_export_stems
from .backends.uvr_backend import UVRBackend


ProgressCallback = Callable[[str, float], None]


@dataclass(frozen=True)
class SeparationResult:
    vocals_path: Path
    instrumental_path: Path
    samplerate: int
    analysis: Dict[str, StemAnalysis]
    model_name: str


def separate_and_enhance(
    input_path: Path,
    output_dir: Path,
    progress: Optional[ProgressCallback] = None,
) -> SeparationResult:
    """Run UVR separation followed by conservative cleanup and enhancement."""
    progress = progress or (lambda _stage, _value: None)
    output_dir.mkdir(parents=True, exist_ok=True)

    backend = UVRBackend.from_env()
    raw_separation = backend.separate(input_path, output_dir, progress=progress)
    try:
        processed = clean_and_export_stems(
            raw_separation.vocals_path,
            raw_separation.instrumental_path,
            output_dir,
            progress=progress,
        )
    finally:
        shutil.rmtree(output_dir / "uvr_raw", ignore_errors=True)

    return SeparationResult(
        vocals_path=processed.vocals_path,
        instrumental_path=processed.instrumental_path,
        samplerate=processed.samplerate,
        analysis=processed.analysis,
        model_name=raw_separation.model_name,
    )

