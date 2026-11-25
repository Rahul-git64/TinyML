"""CLI tool to record 1-second 16 kHz 16-bit mono WAV clips from the laptop microphone.

Supports real human voice capture for accent fine-tuning across:
  help, confirm, cancel, unknown, background
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml_pipeline.dsp import CLIP_SAMPLES, SAMPLE_RATE, save_wav_16k_mono  # noqa: E402

VALID_LABELS = ("help", "confirm", "cancel", "unknown", "background")
DEFAULT_OUT_DIR = REPO_ROOT / "data" / "custom_recordings"


def list_devices() -> None:
    import sounddevice as sd

    print("Available Audio Input Devices:")
    for idx, dev in enumerate(sd.query_devices()):
        if dev.get("max_input_channels", 0) > 0:
            print(f"  [{idx}] {dev['name']} (default_sr={int(dev['default_samplerate'])})")


def print_dataset_counts(out_dir: Path) -> None:
    print(f"\nCurrent custom recordings in {out_dir}:")
    for lbl in VALID_LABELS:
        d = out_dir / lbl
        count = len(list(d.glob("*.wav"))) if d.exists() else 0
        print(f"  - {lbl:<12}: {count} clips")


def record_one_clip(device: int | None) -> np.ndarray:
    import sounddevice as sd

    rec = sd.rec(
        CLIP_SAMPLES,
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        device=device,
    )
    sd.wait()
    return np.asarray(rec[:, 0], dtype=np.float32)


def replay_clip(audio: np.ndarray) -> None:
    import sounddevice as sd

    sd.play(audio, SAMPLE_RATE)
    sd.wait()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Record 1s 16 kHz 16-bit mono WAV clips of real human voices."
    )
    parser.add_argument("--label", choices=VALID_LABELS, help="Target keyword/class label")
    parser.add_argument("--count", type=int, default=10, help="Number of 1s clips to record")
    parser.add_argument("--speaker", type=str, default="speaker01", help="Speaker ID prefix")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="Output directory")
    parser.add_argument("--device", type=int, default=None, help="Input device index")
    parser.add_argument("--list-devices", action="store_true", help="List mic input devices and exit")
    parser.add_argument("--auto-accept", action="store_true", help="Save without per-clip prompt")
    parser.add_argument("--replay", action="store_true", help="Play back each clip after recording")
    parser.add_argument("--countdown", type=int, default=3, help="Seconds countdown before each clip")
    args = parser.parse_args()

    if args.list_devices:
        list_devices()
        return

    if not args.label:
        parser.error("--label is required unless --list-devices is specified.")

    out_label_dir = args.out_dir / args.label
    out_label_dir.mkdir(parents=True, exist_ok=True)
    speaker = "".join(c if c.isalnum() or c in "-_" else "_" for c in args.speaker)

    print(
        f"Recording {args.count} x 1.0s clips (16 kHz, 16-bit mono) for label '{args.label}' "
        f"(speaker='{speaker}')"
    )
    if args.label == "background":
        print("Tip: stay silent or capture normal room/TV/music ambience.")
    else:
        print(f"Tip: clearly speak '{args.label}' once during each 1.0s capture window.")

    saved = 0
    attempt = 0
    while saved < args.count:
        attempt += 1
        print(f"\nClip {saved + 1}/{args.count} — get ready...")
        for sec in range(args.countdown, 0, -1):
            print(f"  {sec}...", flush=True)
            time.sleep(0.65)
        print("  [RECORDING 1.0s NOW]", flush=True)

        audio = record_one_clip(args.device)
        peak = float(np.max(np.abs(audio)))
        rms = float(np.sqrt(np.mean(audio * audio) + 1e-12))
        print(f"  Captured: peak={peak:.3f}, rms={rms:.4f}")

        if peak >= 0.98:
            print("  [WARN] Audio clipped! Move slightly farther from the mic.")
        if args.label != "background" and rms < 0.005:
            print("  [WARN] Very low energy — did you speak during the 1.0s window?")

        if args.replay:
            replay_clip(audio)

        accept = args.auto_accept
        if not accept:
            ans = input("  Save clip? [Y/n/r(eplay)/q(uit)]: ").strip().lower()
            if ans == "r":
                replay_clip(audio)
                ans = input("  Save clip? [Y/n/q]: ").strip().lower()
            if ans == "q":
                break
            accept = ans in ("", "y", "yes")

        if accept:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            fname = f"{speaker}_{args.label}_{ts}_{saved + 1:03d}.wav"
            out_path = out_label_dir / fname
            save_wav_16k_mono(out_path, audio)
            print(f"  Saved -> {out_path}")
            saved += 1
        else:
            print("  Discarded, retrying...")

    print_dataset_counts(args.out_dir)


if __name__ == "__main__":
    main()
