"""Shared 16 kHz -> 40-bin Log-Mel Spectrogram DSP & C++ table exporter.

Mathematical contract shared identically with `firmware_esp32s3/include/dsp_mel.h`:
  - Sample rate: 16000 Hz, 1.0s clip = 16000 float32 samples in [-1.0, 1.0]
  - Causal valid framing (no center padding):
      WIN_LEN = 480 (30 ms), HOP_LEN = 320 (20 ms), N_FFT = 512 (zero-padded)
      NUM_FRAMES = 1 + (16000 - 480) // 320 = 49
  - Window: periodic Hann: 0.5 * (1 - cos(2*pi*n / 480)), n = 0..479
  - Power spectrum: |RFFT(windowed_frame, 512)|^2 over 257 bins (0..256)
  - Mel filterbank: 40 HTK triangular filters, unit peak height, 80.0 Hz .. 7600.0 Hz
  - Log & fixed scaling: clip((log(mel_energy + 1e-6) + 6.0) / 6.0, -3.0, 3.0)
  - Output shape: (49, 40) float32
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

SAMPLE_RATE: int = 16000
CLIP_SAMPLES: int = 16000
WIN_LEN: int = 480
HOP_LEN: int = 320
N_FFT: int = 512
N_FFT_BINS: int = N_FFT // 2 + 1  # 257
NUM_MEL_BINS: int = 40
NUM_FRAMES: int = 1 + (CLIP_SAMPLES - WIN_LEN) // HOP_LEN  # 49
F_MIN_HZ: float = 80.0
F_MAX_HZ: float = 7600.0
LOG_EPS: float = 1e-6
LOG_OFFSET: float = -6.0
LOG_SCALE: float = 6.0
FEAT_CLIP_MIN: float = -3.0
FEAT_CLIP_MAX: float = 3.0


def hz_to_htk_mel(hz: np.ndarray | float) -> np.ndarray | float:
    """Convert frequency in Hz to HTK Mel scale."""
    return 2595.0 * np.log10(1.0 + np.asarray(hz, dtype=np.float64) / 700.0)


def htk_mel_to_hz(mel: np.ndarray | float) -> np.ndarray | float:
    """Convert HTK Mel scale back to Hz."""
    return 700.0 * (10.0 ** (np.asarray(mel, dtype=np.float64) / 2595.0) - 1.0)


def build_hann_window(win_len: int = WIN_LEN) -> np.ndarray:
    """Periodic Hann window matching C++ `g_hann_window[480]`."""
    n = np.arange(win_len, dtype=np.float64)
    win = 0.5 * (1.0 - np.cos(2.0 * np.pi * n / float(win_len)))
    return win.astype(np.float32)


def build_mel_filterbank(
    sample_rate: int = SAMPLE_RATE,
    n_fft: int = N_FFT,
    n_mels: int = NUM_MEL_BINS,
    f_min: float = F_MIN_HZ,
    f_max: float = F_MAX_HZ,
) -> np.ndarray:
    """Build (40, 257) HTK triangular Mel filterbank with unit peak height."""
    n_bins = n_fft // 2 + 1
    mel_min = hz_to_htk_mel(f_min)
    mel_max = hz_to_htk_mel(f_max)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2, dtype=np.float64)
    hz_points = htk_mel_to_hz(mel_points)

    # Exact fractional bin positions mapped to integer FFT bins
    fft_freqs = np.linspace(0.0, sample_rate / 2.0, n_bins, dtype=np.float64)
    fb = np.zeros((n_mels, n_bins), dtype=np.float64)

    for m in range(n_mels):
        f_left = hz_points[m]
        f_center = hz_points[m + 1]
        f_right = hz_points[m + 2]

        up_denom = max(f_center - f_left, 1e-12)
        down_denom = max(f_right - f_center, 1e-12)

        for k in range(n_bins):
            f = fft_freqs[k]
            if f_left < f <= f_center:
                fb[m, k] = (f - f_left) / up_denom
            elif f_center < f < f_right:
                fb[m, k] = (f_right - f) / down_denom

    return fb.astype(np.float32)


HANN_WINDOW: np.ndarray = build_hann_window()
MEL_FILTERBANK: np.ndarray = build_mel_filterbank()


def normalize_audio_length(
    audio: np.ndarray,
    target_len: int = CLIP_SAMPLES,
    pad_mode: str = "end",
) -> np.ndarray:
    """Normalize a 1D float32 audio array to exact `target_len` samples in [-1, 1]."""
    x = np.asarray(audio, dtype=np.float32).reshape(-1)
    if x.size == target_len:
        return np.clip(x, -1.0, 1.0)
    if x.size > target_len:
        return np.clip(x[-target_len:] if pad_mode == "recent" else x[:target_len], -1.0, 1.0)
    out = np.zeros(target_len, dtype=np.float32)
    if pad_mode == "center":
        start = (target_len - x.size) // 2
        out[start : start + x.size] = x
    else:
        out[: x.size] = x
    return np.clip(out, -1.0, 1.0)


def compute_frame_log_mel(frame_480: np.ndarray) -> np.ndarray:
    """Compute one 40-bin Log-Mel feature vector from a 480-sample frame."""
    frame = np.asarray(frame_480, dtype=np.float32)
    if frame.shape[0] != WIN_LEN:
        raise ValueError(f"Expected {WIN_LEN} samples per frame, got {frame.shape[0]}")

    fft_in = np.zeros(N_FFT, dtype=np.float32)
    fft_in[:WIN_LEN] = frame * HANN_WINDOW
    spec = np.fft.rfft(fft_in, n=N_FFT)
    power = (spec.real * spec.real + spec.imag * spec.imag).astype(np.float32)
    mel_energy = MEL_FILTERBANK @ power
    log_mel = (np.log(mel_energy + LOG_EPS) - LOG_OFFSET) / LOG_SCALE
    return np.clip(log_mel, FEAT_CLIP_MIN, FEAT_CLIP_MAX).astype(np.float32)


def compute_log_mel_spectrogram(audio_1s: np.ndarray) -> np.ndarray:
    """Compute exact (49, 40) float32 Log-Mel spectrogram from a 1.0s (16000-sample) clip."""
    audio = normalize_audio_length(audio_1s, CLIP_SAMPLES)
    # Strided causal valid framing: shape (49, 480)
    frames = np.lib.stride_tricks.sliding_window_view(audio, WIN_LEN)[::HOP_LEN]
    if frames.shape[0] != NUM_FRAMES:
        raise RuntimeError(f"Expected {NUM_FRAMES} frames, got {frames.shape[0]}")

    windowed = frames * HANN_WINDOW[None, :]
    fft_buf = np.zeros((NUM_FRAMES, N_FFT), dtype=np.float32)
    fft_buf[:, :WIN_LEN] = windowed
    spec = np.fft.rfft(fft_buf, n=N_FFT, axis=1)
    power = (spec.real * spec.real + spec.imag * spec.imag).astype(np.float32)
    mel_energy = power @ MEL_FILTERBANK.T
    log_mel = (np.log(mel_energy + LOG_EPS) - LOG_OFFSET) / LOG_SCALE
    return np.clip(log_mel, FEAT_CLIP_MIN, FEAT_CLIP_MAX).astype(np.float32)


class StreamingLogMelExtractor:
    """Causal frame-by-frame Log-Mel extractor mirroring ESP32-S3 Core 0 ring buffer."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._buf = np.empty(0, dtype=np.float32)
        self._spec_ring = np.full((NUM_FRAMES, NUM_MEL_BINS), FEAT_CLIP_MIN, dtype=np.float32)
        self._frames_computed = 0

    def push_hop(self, hop_320: np.ndarray) -> np.ndarray | None:
        """Push one 320-sample (20 ms) chunk; computes a frame once >=480 samples exist."""
        hop = np.asarray(hop_320, dtype=np.float32).reshape(-1)
        if hop.size != HOP_LEN:
            raise ValueError(f"Expected hop of {HOP_LEN} samples, got {hop.size}")

        self._buf = np.concatenate([self._buf, hop]) if self._buf.size else hop.copy()
        last_feat: np.ndarray | None = None
        while self._buf.size >= WIN_LEN:
            last_feat = compute_frame_log_mel(self._buf[:WIN_LEN])
            self._buf = self._buf[HOP_LEN:]
            self._spec_ring[:-1] = self._spec_ring[1:]
            self._spec_ring[-1] = last_feat
            self._frames_computed += 1
        return last_feat

    def push_audio_chunk(self, samples: np.ndarray) -> int:
        """Push arbitrary-length audio; returns number of new frames computed."""
        x = np.asarray(samples, dtype=np.float32).reshape(-1)
        if x.size == 0:
            return 0
        self._buf = np.concatenate([self._buf, x]) if self._buf.size else x.copy()
        new_frames = 0
        while self._buf.size >= WIN_LEN:
            feat = compute_frame_log_mel(self._buf[:WIN_LEN])
            self._buf = self._buf[HOP_LEN:]
            self._spec_ring[:-1] = self._spec_ring[1:]
            self._spec_ring[-1] = feat
            self._frames_computed += 1
            new_frames += 1
        return new_frames

    def get_spectrogram(self) -> np.ndarray:
        """Return copy of current rolling (49, 40) Log-Mel spectrogram."""
        return self._spec_ring.copy()


def load_wav_16k_mono(
    path: str | Path,
    target_len: int | None = CLIP_SAMPLES,
    pad_mode: str = "end",
) -> np.ndarray:
    """Read a WAV file, convert to mono float32 in [-1, 1], resample to 16 kHz."""
    sr, data = wavfile.read(str(path))
    if data.ndim > 1:
        data = data.mean(axis=1)

    if np.issubdtype(data.dtype, np.integer):
        max_val = float(np.iinfo(data.dtype).max)
        # int16 normalization matches ESP32-S3 `sample / 32768.0f`
        denom = 32768.0 if data.dtype == np.int16 else max_val
        audio = data.astype(np.float32) / denom
    else:
        audio = data.astype(np.float32)

    if sr != SAMPLE_RATE and audio.size > 0:
        gcd = int(np.gcd(sr, SAMPLE_RATE))
        up = SAMPLE_RATE // gcd
        down = sr // gcd
        audio = resample_poly(audio, up, down).astype(np.float32)

    audio = np.clip(audio, -1.0, 1.0)
    if target_len is not None:
        audio = normalize_audio_length(audio, target_len, pad_mode=pad_mode)
    return audio


def save_wav_16k_mono(path: str | Path, audio: np.ndarray) -> None:
    """Save float32 audio in [-1, 1] as 16 kHz 16-bit mono PCM WAV."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    x = np.clip(np.asarray(audio, dtype=np.float32).reshape(-1), -1.0, 1.0)
    pcm = np.round(x * 32767.0).astype(np.int16)
    wavfile.write(str(out_path), SAMPLE_RATE, pcm)


def _format_float_array(arr: np.ndarray, per_line: int = 8) -> str:
    lines: List[str] = []
    flat = arr.reshape(-1)
    for i in range(0, flat.size, per_line):
        chunk = ", ".join(f"{float(v):.8e}f" for v in flat[i : i + per_line])
        lines.append(f"    {chunk}")
    return ",\n".join(lines)


def _format_uint16_array(arr: List[int], per_line: int = 16) -> str:
    lines: List[str] = []
    for i in range(0, len(arr), per_line):
        chunk = ", ".join(f"{int(v)}" for v in arr[i : i + per_line])
        lines.append(f"    {chunk}")
    return ",\n".join(lines)


def export_dsp_tables_header(output_path: str | Path) -> Tuple[int, int]:
    """Emit C++ header `dsp_tables.h` with Hann window and sparse Mel filterbank."""
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    starts: List[int] = []
    lengths: List[int] = []
    offsets: List[int] = []
    weights: List[float] = []

    for m in range(NUM_MEL_BINS):
        row = MEL_FILTERBANK[m]
        nz = np.where(row > 0.0)[0]
        if nz.size == 0:
            starts.append(0)
            lengths.append(0)
            offsets.append(len(weights))
        else:
            s = int(nz[0])
            e = int(nz[-1]) + 1
            starts.append(s)
            lengths.append(e - s)
            offsets.append(len(weights))
            weights.extend(float(v) for v in row[s:e])

    weights_arr = np.asarray(weights, dtype=np.float32)

    content = f"""#pragma once
// Auto-generated by ml_pipeline/dsp.py — DO NOT EDIT MANUALLY.
// Guarantees bit-for-bit parameter parity between Python training and ESP32-S3 C++.

#include <stdint.h>

namespace kws_dsp {{

constexpr int kSampleRate = {SAMPLE_RATE};
constexpr int kClipSamples = {CLIP_SAMPLES};
constexpr int kWinLen = {WIN_LEN};
constexpr int kHopLen = {HOP_LEN};
constexpr int kNfft = {N_FFT};
constexpr int kNumFftBins = {N_FFT_BINS};
constexpr int kNumMelBins = {NUM_MEL_BINS};
constexpr int kNumFrames = {NUM_FRAMES};
constexpr float kLogEps = {LOG_EPS:.8e}f;
constexpr float kLogOffset = {LOG_OFFSET:.8e}f;
constexpr float kLogScale = {LOG_SCALE:.8e}f;
constexpr float kFeatClipMin = {FEAT_CLIP_MIN:.8e}f;
constexpr float kFeatClipMax = {FEAT_CLIP_MAX:.8e}f;
constexpr int kMelWeightsTotal = {weights_arr.size};

alignas(16) static const float kHannWindow[kWinLen] = {{
{_format_float_array(HANN_WINDOW)}
}};

static const uint16_t kMelBinStart[kNumMelBins] = {{
{_format_uint16_array(starts)}
}};

static const uint16_t kMelBinLen[kNumMelBins] = {{
{_format_uint16_array(lengths)}
}};

static const uint16_t kMelWeightOffset[kNumMelBins] = {{
{_format_uint16_array(offsets)}
}};

alignas(16) static const float kMelWeights[kMelWeightsTotal] = {{
{_format_float_array(weights_arr)}
}};

}}  // namespace kws_dsp
"""
    out_path.write_text(content, encoding="utf-8")
    return HANN_WINDOW.size, int(weights_arr.size)


def verify_streaming_parity(seed: int = 123) -> float:
    """Verify StreamingLogMelExtractor matches batch compute_log_mel_spectrogram."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, CLIP_SAMPLES, endpoint=False, dtype=np.float32)
    sig = (
        0.4 * np.sin(2.0 * np.pi * 440.0 * t)
        + 0.2 * np.sin(2.0 * np.pi * 1320.0 * t)
        + 0.05 * rng.standard_normal(CLIP_SAMPLES).astype(np.float32)
    )
    sig = np.clip(sig, -1.0, 1.0)

    batch = compute_log_mel_spectrogram(sig)
    stream = StreamingLogMelExtractor()
    for i in range(0, CLIP_SAMPLES, HOP_LEN):
        stream.push_hop(sig[i : i + HOP_LEN])
    streaming_spec = stream.get_spectrogram()
    max_diff = float(np.max(np.abs(batch - streaming_spec)))
    if max_diff >= 1e-5:
        raise AssertionError(f"Streaming vs batch parity failed: max_diff={max_diff}")
    return max_diff


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify DSP parity & export ESP32-S3 dsp_tables.h")
    parser.add_argument(
        "--export-header",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "firmware_esp32s3" / "include" / "dsp_tables.h",
        help="Output path for generated C++ dsp_tables.h",
    )
    args = parser.parse_args()

    max_diff = verify_streaming_parity()
    win_len, mel_weights = export_dsp_tables_header(args.export_header)
    print(f"[DSP] Streaming vs batch parity OK (max_abs_diff = {max_diff:.2e})")
    print(
        f"[DSP] Exported {args.export_header} "
        f"(hann={win_len}, sparse_mel_weights={mel_weights}, shape=({NUM_FRAMES}, {NUM_MEL_BINS}))"
    )


if __name__ == "__main__":
    main()
