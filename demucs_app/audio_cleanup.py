from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Tuple

import torch


EPSILON = 1e-8
ProgressCallback = Callable[[str, float], None]


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
    progress: ProgressCallback | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, StemAnalysis]]:
    progress = progress or (lambda _stage, _value: None)
    progress("Checking background noise...", 0.62)
    vocal_analysis = analyze_stem(vocals, samplerate)
    inst_analysis = analyze_stem(instrumental, samplerate)

    progress("Removing noise...", 0.70)
    vocals = _maybe_accept(
        vocals,
        _adaptive_denoise(vocals, samplerate, vocal_analysis, is_vocal=True),
        samplerate,
        vocal_analysis,
        max_change=0.24,
        preserve_envelope=True,
    )
    instrumental = _maybe_accept(
        instrumental,
        _adaptive_denoise(instrumental, samplerate, inst_analysis, is_vocal=False),
        samplerate,
        inst_analysis,
        max_change=0.18,
        preserve_envelope=True,
    )

    progress("Checking bleed and artifacts...", 0.78)
    vocal_analysis = analyze_stem(vocals, samplerate)
    inst_analysis = analyze_stem(instrumental, samplerate)
    bleed_reduced_vocals = _reduce_cross_bleed(
        vocals,
        instrumental,
        strength=1.60,
        mix=0.30,
        floor=0.42,
        protect_low=105.0,
        samplerate=samplerate,
    )
    vocals = _maybe_accept(
        vocals,
        bleed_reduced_vocals,
        samplerate,
        vocal_analysis,
        max_change=0.16,
        allow_equal=True,
        preserve_envelope=True,
    )
    bleed_reduced_instrumental = _reduce_cross_bleed(
        instrumental,
        vocals,
        strength=1.35,
        mix=0.18,
        floor=0.58,
        protect_low=45.0,
        samplerate=samplerate,
    )
    instrumental = _maybe_accept(
        instrumental,
        bleed_reduced_instrumental,
        samplerate,
        inst_analysis,
        max_change=0.12,
        allow_equal=True,
        preserve_envelope=True,
    )

    progress("Enhancing audio...", 0.86)
    vocal_analysis = analyze_stem(vocals, samplerate)
    inst_analysis = analyze_stem(instrumental, samplerate)
    vocals = _maybe_accept(
        vocals,
        _enhance_vocals(vocals, samplerate, vocal_analysis),
        samplerate,
        vocal_analysis,
        max_change=0.15,
        allow_equal=True,
        preserve_envelope=True,
    )
    instrumental = _maybe_accept(
        instrumental,
        _enhance_instrumental(instrumental, samplerate, inst_analysis),
        samplerate,
        inst_analysis,
        max_change=0.10,
        allow_equal=True,
        preserve_envelope=True,
    )

    progress("Finalizing files...", 0.92)
    vocals = _maybe_accept(
        vocals,
        _safe_normalize(vocals, target_peak=0.94),
        samplerate,
        analyze_stem(vocals, samplerate),
        max_change=0.28,
        allow_equal=True,
        preserve_envelope=False,
    )
    instrumental = _maybe_accept(
        instrumental,
        _safe_normalize(instrumental, target_peak=0.92),
        samplerate,
        analyze_stem(instrumental, samplerate),
        max_change=0.22,
        allow_equal=True,
        preserve_envelope=False,
    )

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


def _adaptive_denoise(
    wav: torch.Tensor,
    samplerate: int,
    analysis: StemAnalysis,
    is_vocal: bool,
) -> torch.Tensor:
    if is_vocal:
        strength = 1.95 if analysis.noise_ratio > 0.18 else 1.65
        mix = 0.54 if analysis.noise_ratio > 0.22 else 0.42
        floor = 0.27 if analysis.hiss_ratio > 0.10 else 0.33
    else:
        strength = 1.52 if analysis.noise_ratio > 0.20 else 1.32
        mix = 0.32 if analysis.noise_ratio > 0.20 else 0.24
        floor = 0.42

    cleaned = _denoise(wav, strength=strength, mix=mix, floor=floor, n_fft=2048)
    if samplerate >= 32000:
        # A second wider-window pass catches slow hiss beds without clamping vocal consonants.
        wide_mix = 0.20 if is_vocal else 0.12
        cleaned = _denoise(cleaned, strength=max(1.18, strength - 0.34), mix=wide_mix, floor=0.55, n_fft=4096)
    return cleaned


def _denoise(
    wav: torch.Tensor,
    strength: float,
    mix: float,
    floor: float,
    n_fft: int,
) -> torch.Tensor:
    wav = _as_float_audio(wav)
    hop_length = max(128, n_fft // 4)
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
        mask = _smooth_mask(mask)
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
    wav = _fft_filter(wav, samplerate, highpass=70.0, presence_gain=1.08, air_gain=1.02)
    if analysis.muffled_score > 0.20:
        wav = _fft_filter(wav, samplerate, presence_gain=1.06)
    if analysis.hiss_ratio > 0.08:
        wav = _fft_filter(wav, samplerate, lowpass=15500.0)
    return wav


def _enhance_instrumental(wav: torch.Tensor, samplerate: int, analysis: StemAnalysis) -> torch.Tensor:
    wav = _fft_filter(wav, samplerate, highpass=32.0, air_gain=1.01)
    if analysis.hum_ratio > 0.006:
        wav = _notch_hum(wav, samplerate)
    return wav


def _fft_filter(
    wav: torch.Tensor,
    samplerate: int,
    highpass: float | None = None,
    lowpass: float | None = None,
    presence_gain: float = 1.0,
    air_gain: float = 1.0,
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
    if air_gain != 1.0:
        air = (freqs >= 6500.0) & (freqs <= min(12000.0, samplerate / 2))
        gain[air] *= air_gain
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


def _safe_normalize(wav: torch.Tensor, target_peak: float) -> torch.Tensor:
    wav = _as_float_audio(wav)
    peak = wav.abs().max()
    if peak > target_peak:
        return wav / peak * target_peak
    if peak < 0.35:
        gain = min(float((target_peak * 0.78) / (peak + EPSILON)), 1.8)
        return wav * gain
    return wav


def _reduce_cross_bleed(
    target: torch.Tensor,
    reference: torch.Tensor,
    strength: float,
    mix: float,
    floor: float,
    protect_low: float,
    samplerate: int,
) -> torch.Tensor:
    target = _as_float_audio(target)
    reference = _match_shape(_as_float_audio(reference), target)
    n_fft = 4096
    hop_length = 1024
    length = target.shape[-1]
    window = torch.hann_window(n_fft)
    freqs = torch.linspace(0, samplerate / 2, n_fft // 2 + 1)
    cleaned = []
    for target_channel, reference_channel in zip(target, reference):
        padded_target = target_channel
        padded_reference = reference_channel
        if padded_target.numel() < n_fft:
            pad = n_fft - padded_target.numel()
            padded_target = torch.nn.functional.pad(padded_target, (0, pad))
            padded_reference = torch.nn.functional.pad(padded_reference, (0, pad))
        target_spec = torch.stft(
            padded_target,
            n_fft=n_fft,
            hop_length=hop_length,
            window=window,
            return_complex=True,
        )
        reference_spec = torch.stft(
            padded_reference,
            n_fft=n_fft,
            hop_length=hop_length,
            window=window,
            return_complex=True,
        )
        target_mag = target_spec.abs()
        reference_mag = reference_spec.abs()
        dominance = target_mag / (target_mag + reference_mag + EPSILON)
        mask = floor + (1.0 - floor) * dominance.pow(strength)
        low_mask = freqs <= protect_low
        mask[low_mask, :] = torch.maximum(mask[low_mask, :], torch.full_like(mask[low_mask, :], 0.92))
        mask = _smooth_mask(mask)
        cleaned_channel = torch.istft(
            target_spec * mask,
            n_fft=n_fft,
            hop_length=hop_length,
            window=window,
            length=padded_target.numel(),
        )[:length]
        cleaned.append(cleaned_channel)
    cleaned_wav = torch.stack(cleaned, dim=0)
    return target * (1.0 - mix) + cleaned_wav * mix


def _smooth_mask(mask: torch.Tensor) -> torch.Tensor:
    # Light smoothing reduces musical-noise speckle from spectral gating.
    smoothed = mask[None, None]
    smoothed = torch.nn.functional.avg_pool2d(smoothed, kernel_size=(3, 3), stride=1, padding=1)
    return smoothed[0, 0].clamp(0.0, 1.0)


def _preserve_loudness_envelope(
    original: torch.Tensor,
    candidate: torch.Tensor,
    samplerate: int,
    max_gain_db: float = 2.0,
) -> torch.Tensor:
    original = _as_float_audio(original)
    candidate = _match_shape(_as_float_audio(candidate), original)
    original_env = _rms_envelope(original, samplerate)
    candidate_env = _rms_envelope(candidate, samplerate)
    gain = (original_env / (candidate_env + EPSILON)).clamp(
        _db_to_gain(-max_gain_db),
        _db_to_gain(max_gain_db),
    )
    gain = _smooth_gain_curve(gain, samplerate)
    leveled = candidate * gain
    original_peak = original.abs().max()
    leveled_peak = leveled.abs().max()
    peak_limit = max(float(original_peak * 1.08), 0.98)
    if leveled_peak > peak_limit:
        leveled = leveled / (leveled_peak + EPSILON) * peak_limit
    return leveled


def _rms_envelope(wav: torch.Tensor, samplerate: int) -> torch.Tensor:
    mono = wav.mean(dim=0, keepdim=True)
    window = max(512, int(samplerate * 0.30))
    if window % 2 == 0:
        window += 1
    padded = _pad_envelope_input(mono.square()[None], window)
    env = torch.nn.functional.avg_pool1d(padded, kernel_size=window, stride=1)[0].sqrt()
    return env.expand_as(wav)


def _smooth_gain_curve(gain: torch.Tensor, samplerate: int) -> torch.Tensor:
    window = max(256, int(samplerate * 0.70))
    if window % 2 == 0:
        window += 1
    mono_gain = gain.mean(dim=0, keepdim=True)
    padded = _pad_envelope_input(mono_gain[None], window)
    smoothed = torch.nn.functional.avg_pool1d(padded, kernel_size=window, stride=1)[0]
    return smoothed.expand_as(gain)


def _pad_envelope_input(wav: torch.Tensor, window: int) -> torch.Tensor:
    pad = window // 2
    mode = "reflect" if wav.shape[-1] > pad else "replicate"
    return torch.nn.functional.pad(wav, (pad, pad), mode=mode)


def _db_to_gain(db: float) -> float:
    return 10 ** (db / 20)


def _match_shape(wav: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if wav.shape[0] < target.shape[0]:
        wav = wav.expand(target.shape[0], wav.shape[-1])
    elif wav.shape[0] > target.shape[0]:
        wav = wav[:target.shape[0]]
    if wav.shape[-1] < target.shape[-1]:
        wav = torch.nn.functional.pad(wav, (0, target.shape[-1] - wav.shape[-1]))
    elif wav.shape[-1] > target.shape[-1]:
        wav = wav[..., :target.shape[-1]]
    return wav


def _maybe_accept(
    original: torch.Tensor,
    candidate: torch.Tensor,
    samplerate: int,
    original_analysis: StemAnalysis,
    max_change: float,
    allow_equal: bool = False,
    preserve_envelope: bool = False,
) -> torch.Tensor:
    candidate = torch.nan_to_num(candidate.detach().cpu().float())
    if candidate.shape != original.shape:
        return original
    if preserve_envelope:
        candidate = _preserve_loudness_envelope(original, candidate, samplerate)
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
