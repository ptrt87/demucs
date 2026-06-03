from __future__ import annotations

import os
import shutil
from inspect import Parameter, signature
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional

from ..dependencies import format_missing_dependency_error


ProgressCallback = Callable[[str, float], None]

DEFAULT_UVR_MODEL = "model_bs_roformer_ep_317_sdr_12.9755.ckpt"
DEFAULT_CHUNK_DURATION = 600


class UVRBackendError(RuntimeError):
    """Raised when Ultimate Vocal Remover processing cannot complete."""


@dataclass(frozen=True)
class UVRSeparation:
    vocals_path: Path
    instrumental_path: Path
    model_name: str


class UVRBackend:
    """Thin adapter around audio-separator's UVR model API."""

    def __init__(
        self,
        model_filename: str = DEFAULT_UVR_MODEL,
        model_dir: Path | None = None,
        ensemble_preset: str | None = None,
        chunk_duration: int = DEFAULT_CHUNK_DURATION,
    ) -> None:
        self.model_filename = model_filename
        self.model_dir = model_dir
        self.ensemble_preset = ensemble_preset
        self.chunk_duration = chunk_duration

    @classmethod
    def from_env(cls) -> "UVRBackend":
        model_dir = os.environ.get("UVR_MODEL_DIR")
        return cls(
            model_filename=os.environ.get("UVR_MODEL_FILENAME", DEFAULT_UVR_MODEL),
            model_dir=Path(model_dir).expanduser() if model_dir else None,
            ensemble_preset=os.environ.get("UVR_ENSEMBLE_PRESET") or None,
            chunk_duration=_env_int("UVR_CHUNK_DURATION", DEFAULT_CHUNK_DURATION),
        )

    def separate(
        self,
        input_path: Path,
        work_dir: Path,
        progress: Optional[ProgressCallback] = None,
    ) -> UVRSeparation:
        progress = progress or (lambda _stage, _value: None)
        raw_dir = work_dir / "uvr_raw"
        raw_dir.mkdir(parents=True, exist_ok=True)

        progress("Separating with Ultimate Vocal Remover...", 0.05)
        try:
            from audio_separator.separator import Separator
        except ModuleNotFoundError as exc:
            raise UVRBackendError(format_missing_dependency_error(exc.name or "audio_separator")) from exc
        except Exception as exc:
            raise UVRBackendError(_friendly_uvr_error(exc)) from exc

        separator = self._create_separator(Separator, raw_dir)
        try:
            if self.ensemble_preset:
                separator.load_model()
                model_label = self.ensemble_preset
            else:
                separator.load_model(model_filename=self.model_filename)
                model_label = self.model_filename

            output_names = {
                "Vocals": "uvr_raw_vocals",
                "Instrumental": "uvr_raw_instrumental",
                "Other": "uvr_raw_instrumental",
            }
            output_files = separator.separate(str(input_path), output_names) or []
        except Exception as exc:
            raise UVRBackendError(_friendly_uvr_error(exc)) from exc

        vocals_path, instrumental_path = _pick_uvr_stems(output_files, raw_dir)
        normalized_vocals = raw_dir / "vocals.wav"
        normalized_instrumental = raw_dir / "instrumental.wav"
        _copy_stem(vocals_path, normalized_vocals)
        _copy_stem(instrumental_path, normalized_instrumental)
        progress("Separating with Ultimate Vocal Remover...", 0.60)
        return UVRSeparation(
            vocals_path=normalized_vocals,
            instrumental_path=normalized_instrumental,
            model_name=model_label,
        )

    def _create_separator(self, separator_cls, output_dir: Path):
        kwargs = {
            "output_dir": str(output_dir),
            "output_format": "WAV",
            "normalization_threshold": 0.9,
            "amplification_threshold": 0.0,
            "use_soundfile": True,
            "chunk_duration": self.chunk_duration,
            "mdxc_params": {
                "overlap": _env_int("UVR_MDXC_OVERLAP", 16),
                "batch_size": max(1, _env_int("UVR_MDXC_BATCH_SIZE", 1)),
            },
        }
        if self.model_dir is not None:
            kwargs["model_file_dir"] = str(self.model_dir)
        if self.ensemble_preset:
            kwargs["ensemble_preset"] = self.ensemble_preset
        return separator_cls(**_filter_supported_kwargs(separator_cls, kwargs))


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if not value:
        return default
    try:
        return max(0, int(value))
    except ValueError:
        return default


def _filter_supported_kwargs(callable_obj, kwargs: dict) -> dict:
    try:
        parameters = signature(callable_obj).parameters.values()
    except (TypeError, ValueError):
        return kwargs
    names = set()
    accepts_any = False
    for parameter in parameters:
        if parameter.kind == Parameter.VAR_KEYWORD:
            accepts_any = True
        elif parameter.kind in (Parameter.POSITIONAL_OR_KEYWORD, Parameter.KEYWORD_ONLY):
            names.add(parameter.name)
    if accepts_any:
        return kwargs
    return {key: value for key, value in kwargs.items() if key in names}


def _friendly_uvr_error(exc: Exception) -> str:
    message = str(exc) or exc.__class__.__name__
    lowered = message.lower()
    if "model" in lowered and ("not found" in lowered or "no such" in lowered):
        return (
            "Ultimate Vocal Remover could not find or download the selected model. "
            "Check your internet connection for the first run, or set UVR_MODEL_DIR "
            "to a folder that already contains the UVR model files."
        )
    if "ffmpeg" in lowered:
        return (
            "Ultimate Vocal Remover needs FFmpeg to decode this audio file. "
            "Install FFmpeg and make sure ffmpeg.exe is available on PATH."
        )
    if "cuda" in lowered or "onnxruntime" in lowered:
        return (
            "Ultimate Vocal Remover could not start the selected acceleration backend. "
            "Install the CPU extra with python -m pip install \"audio-separator[cpu]\" "
            "or reinstall the matching GPU/CUDA runtime for audio-separator."
        )
    return f"Ultimate Vocal Remover processing failed: {message}"


def _pick_uvr_stems(output_files: Iterable[str | Path], output_dir: Path) -> tuple[Path, Path]:
    paths = [_resolve_output_path(value, output_dir) for value in output_files]
    existing = [path for path in paths if path.exists()]
    if not existing:
        existing = list(output_dir.glob("*.wav"))

    vocals = _find_stem(existing, _VOCAL_MARKERS, _INSTRUMENTAL_MARKERS)
    instrumental = _find_stem(existing, _INSTRUMENTAL_MARKERS, _VOCAL_MARKERS)
    if vocals is None and len(existing) == 2 and instrumental is not None:
        vocals = next(path for path in existing if path != instrumental)
    if instrumental is None and len(existing) == 2 and vocals is not None:
        instrumental = next(path for path in existing if path != vocals)

    if vocals is None or instrumental is None:
        names = ", ".join(path.name for path in existing) or "no output files"
        raise UVRBackendError(
            "Ultimate Vocal Remover finished, but the app could not identify separate "
            f"vocals and instrumental stems. Outputs seen: {names}."
        )
    return vocals, instrumental


_VOCAL_MARKERS = ("vocals", "vocal", "voice", "sing")
_INSTRUMENTAL_MARKERS = (
    "instrumental",
    "instrument",
    "inst",
    "accompaniment",
    "karaoke",
    "no_vocals",
    "no-vocals",
    "novocals",
    "other",
)


def _resolve_output_path(value: str | Path, output_dir: Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = output_dir / path
    return path


def _find_stem(paths: list[Path], include: tuple[str, ...], exclude: tuple[str, ...]) -> Path | None:
    for path in paths:
        name = path.stem.lower()
        if any(marker in name for marker in include) and not any(marker in name for marker in exclude):
            return path
    for path in paths:
        name = path.stem.lower()
        if any(marker in name for marker in include):
            return path
    return None


def _copy_stem(source: Path, destination: Path) -> None:
    if source.resolve() == destination.resolve():
        return
    shutil.copy2(source, destination)
