from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch


EPSILON = 1e-8


@dataclass
class StemAnalysis:
    rms: float
    peak: float
    noise_ratio: float
    hiss_ratio: float
    hum_ratio: float
    spectral_centroid: float
    muffled_score: float
    artifact_score: float

    @property
    def cleanup_score(self) -> float:
        clipping_penalty = max(0.0, self.peak - 0.99) * 8.0
        return (
            self.noise_ratio * 0.55
            + self.hiss_ratio * 0.25
            + self.hum_ratio * 0.30
            + self.artifact_score * 0.20
            + clipping_penalty
        )


def analyze_stem(wav: torch.Tensor, samplerate: int) -> StemAnalysis:
    wav = _as_float_audio(wav)
    mono = wav.mean(dim=0)
    rms = float(torch.sqrt(torch.mean(mono.square()) + EPSILON))
    peak = float(wav.abs().max())

    frame = max(512, int(samplerate * 0.046))
    hop = max(128, frame // 2)
    if mono.numel() < frame:
        frame_rms = torch.sqrt(torch.mean(mono.square()).reshape(1) + EPSILON)
    else:
        frames = mono.unfold(0, frame, hop)
        frame_rms = torch.sqrt(torch.mean(frames.square(), dim=1) + EPSILON)

    noise_floor = float(torch.quantile(frame_rms, 0.12))
    noise_ratio = noise_floor / (rms + EPSILON)

    n_fft = 4096
    hop_length = 1024
    if mono.numel() < n_fft:
        padded = torch.nn.functional.pad(mono, (0, n_fft - mono.numel()))
    else:
        padded = mono
    window = torch.hann_window(n_fft, device=padded.device)
    spec = torch.stft(
        padded,
        n_fft=n_fft,
        hop_length=hop_length,
        window=window,
        return_complex=True,
    )
    mag = spec.abs().mean(dim=1)
    freqs = torch.linspace(0, samplerate / 2, mag.numel(), device=mag.device)

    hiss_ratio = _band_ratio(mag, freqs, 8000.0, min(18000.0, samplerate / 2))
    hum_ratio = (
        _band_ratio(mag, freqs, 47.0, 63.0)
        + _band_ratio(mag, freqs, 94.0, 126.0)
        + _band_ratio(mag, freqs, 140.0, 190.0)
    )
    spectral_centroid = float((freqs * mag).sum() / (mag.sum() + EPSILON))
    presence_ratio = _band_ratio(mag, freqs, 1800.0, min(6000.0, samplerate / 2))
    muffled_score = max(0.0, min(1.0, (2200.0 - spectral_centroid) / 2200.0))
    if presence_ratio > 0.12:
        muffled_score *= 0.5

    high_energy = _band_ratio(mag, freqs, 10000.0, min(20000.0, samplerate / 2))
    artifact_score = max(0.0, high_energy - 0.16) + max(0.0, noise_ratio - 0.28)

    return StemAnalysis(
        rms=rms,
        peak=peak,
        noise_ratio=float(noise_ratio),
        hiss_ratio=float(hiss_ratio),
        hum_ratio=float(hum_ratio),
        spectral_centroid=spectral_centroid,
        muffled_score=float(muffled_score),
        artifact_score=float(artifact_score),
    )


def cleanup_stems(
    vocals: torch.Tensor,
    instrumental: torch.Tensor,
    samplerate: int,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, StemAnalysis]]:
    vocal_analysis = analyze_stem(vocals, samplerate)
    inst_analysis = analyze_stem(instrumental, samplerate)

    vocals = _maybe_accept(
        vocals,
        _denoise(vocals, strength=1.75, mix=0.42, floor=0.30),
        samplerate,
        vocal_analysis,
        max_change=0.18,
    )
    instrumental = _maybe_accept(
        instrumental,
        _denoise(instrumental, strength=1.45, mix=0.28, floor=0.42),
        samplerate,
        inst_analysis,
        max_change=0.14,
    )

    vocal_analysis = analyze_stem(vocals, samplerate)
    inst_analysis = analyze_stem(instrumental, samplerate)

    vocals = _maybe_accept(
        vocals,
        _enhance_vocals(vocals, samplerate, vocal_analysis),
        samplerate,
        vocal_analysis,
        max_change=0.12,
        allow_equal=True,
    )
    instrumental = _maybe_accept(
        instrumental,
        _enhance_instrumental(instrumental, samplerate, inst_analysis),
        samplerate,
        inst_analysis,
        max_change=0.10,
        allow_equal=True,
    )

    vocals = _safe_normalize(vocals)
    instrumental = _safe_normalize(instrumental)

    return vocals, instrumental, {
        "vocals": analyze_stem(vocals, samplerate),
        "instrumental": analyze_stem(instrumental, samplerate),
    }


def _as_float_audio(wav: torch.Tensor) -> torch.Tensor:
    wav = wav.detach().cpu().float()
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    return torch.nan_to_num(wav)


def _band_ratio(mag: torch.Tensor, freqs: torch.Tensor, low: float, high: float) -> float:
    mask = (freqs >= low) & (freqs <= high)
    if not bool(mask.any()):
        return 0.0
    return float(mag[mask].sum() / (mag.sum() + EPSILON))


def _denoise(wav: torch.Tensor, strength: float, mix: float, floor: float) -> torch.Tensor:
    wav = _as_float_audio(wav)
    n_fft = 2048
    hop_length = 512
    length = wav.shape[-1]
    window = torch.hann_window(n_fft)
    cleaned = []
    for channel in wav:
        if channel.numel() < n_fft:
            channel = torch.nn.functional.pad(channel, (0, n_fft - channel.numel()))
        spec = torch.stft(
            channel,
            n_fft=n_fft,
            hop_length=hop_length,
            window=window,
            return_complex=True,
        )
        mag = spec.abs()
        frame_energy = mag.mean(dim=0)
        quiet_count = max(3, int(frame_energy.numel() * 0.15))
        quiet_idx = torch.topk(frame_energy, quiet_count, largest=False).indices
        noise_profile = torch.median(mag[:, quiet_idx], dim=1).values[:, None]
        threshold = noise_profile * strength
        mask = ((mag - threshold) / (mag + EPSILON)).clamp(0.0, 1.0)
        mask = mask * (1.0 - floor) + floor
        denoised = torch.istft(
            spec * mask,
            n_fft=n_fft,
            hop_length=hop_length,
            window=window,
            length=channel.numel(),
        )[:length]
        cleaned.append(denoised)
    cleaned_wav = torch.stack(cleaned, dim=0)
    return wav * (1.0 - mix) + cleaned_wav * mix


def _enhance_vocals(wav: torch.Tensor, samplerate: int, analysis: StemAnalysis) -> torch.Tensor:
    wav = _fft_filter(wav, samplerate, highpass=65.0, presence_gain=1.05)
    if analysis.hiss_ratio > 0.08:
        wav = _fft_filter(wav, samplerate, lowpass=15500.0)
    return wav


def _enhance_instrumental(wav: torch.Tensor, samplerate: int, analysis: StemAnalysis) -> torch.Tensor:
    wav = _fft_filter(wav, samplerate, highpass=35.0)
    if analysis.hum_ratio > 0.006:
        wav = _notch_hum(wav, samplerate)
    return wav


def _fft_filter(
    wav: torch.Tensor,
    samplerate: int,
    highpass: float | None = None,
    lowpass: float | None = None,
    presence_gain: float = 1.0,
) -> torch.Tensor:
    wav = _as_float_audio(wav)
    spec = torch.fft.rfft(wav, dim=-1)
    freqs = torch.fft.rfftfreq(wav.shape[-1], d=1.0 / samplerate)
    gain = torch.ones_like(freqs)
    if highpass is not None:
        ramp = ((freqs - highpass) / max(highpass, 1.0)).clamp(0.0, 1.0)
        gain *= ramp
    if lowpass is not None:
        ramp = ((lowpass - freqs) / max(lowpass * 0.15, 1.0)).clamp(0.0, 1.0)
        gain *= ramp
    if presence_gain != 1.0:
        presence = (freqs >= 1800.0) & (freqs <= min(5200.0, samplerate / 2))
        gain[presence] *= presence_gain
    out = torch.fft.irfft(spec * gain, n=wav.shape[-1], dim=-1)
    return out


def _notch_hum(wav: torch.Tensor, samplerate: int) -> torch.Tensor:
    wav = _as_float_audio(wav)
    spec = torch.fft.rfft(wav, dim=-1)
    freqs = torch.fft.rfftfreq(wav.shape[-1], d=1.0 / samplerate)
    gain = torch.ones_like(freqs)
    for base in (50.0, 60.0):
        for multiple in (1, 2, 3):
            center = base * multiple
            width = 1.5 * multiple
            notch = torch.exp(-0.5 * ((freqs - center) / width).square())
            gain *= 1.0 - notch * 0.35
    return torch.fft.irfft(spec * gain, n=wav.shape[-1], dim=-1)


def _safe_normalize(wav: torch.Tensor) -> torch.Tensor:
    wav = _as_float_audio(wav)
    peak = wav.abs().max()
    if peak > 0.98:
        return wav / peak * 0.96
    if peak < 0.35:
        gain = min(float(0.75 / (peak + EPSILON)), 1.6)
        return wav * gain
    return wav


def _maybe_accept(
    original: torch.Tensor,
    candidate: torch.Tensor,
    samplerate: int,
    original_analysis: StemAnalysis,
    max_change: float,
    allow_equal: bool = False,
) -> torch.Tensor:
    candidate = torch.nan_to_num(candidate.detach().cpu().float())
    if candidate.shape != original.shape:
        return original
    candidate_analysis = analyze_stem(candidate, samplerate)
    if candidate_analysis.peak > 1.05:
        return original
    change = _relative_change(original, candidate)
    if change > max_change:
        return original
    old_score = original_analysis.cleanup_score
    new_score = candidate_analysis.cleanup_score
    if new_score < old_score * 0.995:
        return candidate
    if allow_equal and new_score <= old_score * 1.01 and candidate_analysis.peak <= original_analysis.peak * 1.03:
        return candidate
    return original


def _relative_change(original: torch.Tensor, candidate: torch.Tensor) -> float:
    original = _as_float_audio(original)
    candidate = _as_float_audio(candidate)
    delta = torch.sqrt(torch.mean((candidate - original).square()) + EPSILON)
    base = torch.sqrt(torch.mean(original.square()) + EPSILON)
    return float(delta / (base + EPSILON))
