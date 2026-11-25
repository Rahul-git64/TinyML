"""Live laptop microphone capture & offline local speaker WAV prompt engine."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from ml_pipeline.dsp import (
    CLIP_SAMPLES,
    HOP_LEN,
    SAMPLE_RATE,
    compute_log_mel_spectrogram,
    load_wav_16k_mono,
    save_wav_16k_mono,
)

ASSETS_DIR = Path(__file__).resolve().parent / "assets"
DEFAULT_PROMPT_WAV = ASSETS_DIR / "verify_prompt.wav"
PROMPT_TEXT = "Emergency triggered. Say Confirm or Cancel."


def list_input_devices() -> List[Dict]:
    """Return available audio input devices via sounddevice."""
    try:
        import sounddevice as sd

        devices = []
        for idx, dev in enumerate(sd.query_devices()):
            if dev.get("max_input_channels", 0) > 0:
                devices.append({"index": idx, "name": dev["name"], "sr": dev["default_samplerate"]})
        return devices
    except Exception:
        return []


def _synthesize_fallback_chime(duration_sec: float = 1.6) -> np.ndarray:
    """Offline fallback alert chime (16 kHz mono) if OS speech synthesizer is unavailable."""
    n = int(SAMPLE_RATE * duration_sec)
    t = np.linspace(0.0, duration_sec, n, endpoint=False, dtype=np.float32)
    sig = np.zeros(n, dtype=np.float32)
    tones = [(0.0, 0.28, 880.0), (0.34, 0.62, 660.0), (0.70, 1.05, 987.77)]
    for t0, t1, freq in tones:
        mask = (t >= t0) & (t < t1)
        env = np.sin(np.pi * (t[mask] - t0) / (t1 - t0))
        sig[mask] = 0.45 * env * np.sin(2.0 * np.pi * freq * (t[mask] - t0))
    return np.clip(sig, -1.0, 1.0)


def ensure_verify_prompt_wav(output_path: Path = DEFAULT_PROMPT_WAV) -> Path:
    """Create local 16 kHz mono WAV prompt ('Emergency triggered. Say Confirm or Cancel.')."""
    out = Path(output_path)
    if out.exists() and out.stat().st_size > 1000:
        return out

    out.parent.mkdir(parents=True, exist_ok=True)
    generated = False

    if sys.platform.startswith("win"):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            ps_script = (
                "Add-Type -AssemblyName System.Speech; "
                "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                "$s.Rate = 0; "
                f"$s.SetOutputToWaveFile('{str(tmp_path).replace(chr(39), chr(39)*2)}'); "
                f"$s.Speak('{PROMPT_TEXT}'); "
                "$s.Dispose();"
            )
            proc = subprocess.run(
                ["powershell", "-NoProfile", "-NonInteractive", "-Command", ps_script],
                capture_output=True,
                text=True,
                timeout=15,
            )
            if proc.returncode == 0 and tmp_path.exists() and tmp_path.stat().st_size > 1000:
                audio = load_wav_16k_mono(tmp_path, target_len=None)
                save_wav_16k_mono(out, audio)
                generated = True
        except Exception:
            generated = False
        finally:
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass

    if not generated:
        save_wav_16k_mono(out, _synthesize_fallback_chime())
    return out


def play_verify_prompt_async(wav_path: Path = DEFAULT_PROMPT_WAV) -> None:
    """Play local WAV prompt asynchronously over laptop speakers."""
    path = ensure_verify_prompt_wav(wav_path)

    def _worker() -> None:
        try:
            import sounddevice as sd

            audio = load_wav_16k_mono(path, target_len=None)
            sd.play(audio, SAMPLE_RATE, blocking=True)
            return
        except Exception:
            pass

        if sys.platform.startswith("win"):
            try:
                import winsound

                winsound.PlaySound(str(path), winsound.SND_FILENAME)
            except Exception:
                pass

    threading.Thread(target=_worker, daemon=True).start()


class AudioCaptureEngine:
    """Thread-safe 16 kHz mono microphone ring buffer (1.0s rolling window)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._ring = np.zeros(CLIP_SAMPLES, dtype=np.float32)
        self._stream = None
        self._running = False
        self._device_index: Optional[int] = None
        self.last_error: Optional[str] = None
        self.total_samples: int = 0

    @property
    def is_running(self) -> bool:
        return self._running

    def start(self, device_index: Optional[int] = None) -> bool:
        self.stop()
        self.last_error = None
        try:
            import sounddevice as sd

            def _callback(indata, frames, time_info, status) -> None:  # noqa: ARG001
                mono = np.asarray(indata[:, 0], dtype=np.float32)
                n = mono.size
                with self._lock:
                    if n >= CLIP_SAMPLES:
                        self._ring[:] = mono[-CLIP_SAMPLES:]
                    else:
                        self._ring[:-n] = self._ring[n:]
                        self._ring[-n:] = mono
                    self.total_samples += n

            self._stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="float32",
                blocksize=HOP_LEN,
                device=device_index,
                callback=_callback,
            )
            self._stream.start()
            self._device_index = device_index
            self._running = True
            return True
        except Exception as exc:
            self.last_error = str(exc)
            self._stream = None
            self._running = False
            return False

    def stop(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
        self._stream = None
        self._running = False

    def clear(self) -> None:
        with self._lock:
            self._ring.fill(0.0)

    def push_external_audio(self, audio_1s: np.ndarray) -> None:
        """Inject a 1s clip (e.g. uploaded WAV) into the rolling buffer for testing."""
        x = np.asarray(audio_1s, dtype=np.float32).reshape(-1)
        with self._lock:
            if x.size >= CLIP_SAMPLES:
                self._ring[:] = x[-CLIP_SAMPLES:]
            else:
                self._ring[:-x.size] = self._ring[x.size :]
                self._ring[-x.size :] = x
            self.total_samples += x.size

    def get_window_and_spec(self) -> tuple[np.ndarray, np.ndarray, float]:
        """Snapshot rolling 1.0s audio, compute (49, 40) Log-Mel spec & RMS."""
        with self._lock:
            win = self._ring.copy()
        rms = float(np.sqrt(np.mean(win * win) + 1e-12))
        spec = compute_log_mel_spectrogram(win)
        return win, spec, rms
