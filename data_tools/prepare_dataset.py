"""Dataset preparation combining Google Speech Commands v2 + MUSAN + real human clips.

Strictly real audio only (NO synthetic TTS data).
Outputs:
  - data/processed/stage1_features.npz  (classes: background, unknown, help)
  - data/processed/stage2_features.npz  (classes: silence_unknown, confirm, cancel)
  - data/processed/dataset_manifest.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tarfile
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml_pipeline.dsp import (  # noqa: E402
    CLIP_SAMPLES,
    SAMPLE_RATE,
    compute_log_mel_spectrogram,
    load_wav_16k_mono,
    normalize_audio_length,
)

GSC_URL = "http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz"
MUSAN_URL = "https://www.openslr.org/resources/17/musan.tar.gz"

STAGE1_LABELS = ("background", "unknown", "help")
STAGE2_LABELS = ("silence_unknown", "confirm", "cancel")

DEFAULT_CUSTOM_DIR = REPO_ROOT / "data" / "custom_recordings"
DEFAULT_GSC_DIR = REPO_ROOT / "data" / "raw" / "speech_commands_v0.02"
DEFAULT_MUSAN_DIR = REPO_ROOT / "data" / "raw" / "musan"
DEFAULT_OUT_DIR = REPO_ROOT / "data" / "processed"


def _download_and_extract(url: str, dest_dir: Path, archive_name: str) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    archive_path = dest_dir.parent / archive_name
    if not archive_path.exists():
        print(f"[DOWNLOAD] Fetching {url} -> {archive_path} ...")
        urllib.request.urlretrieve(url, archive_path)
    print(f"[EXTRACT] Extracting {archive_path} -> {dest_dir} ...")
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(dest_dir)


def _speaker_id_from_path(wav_path: Path, source: str) -> str:
    stem = wav_path.stem
    if source == "gsc":
        # GSC filenames look like `0a2b400e_nohash_0.wav`
        return stem.split("_nohash_")[0]
    if source == "custom":
        # Custom filenames look like `speaker01_help_20261125_160000_001.wav`
        return stem.split("_")[0]
    return hashlib.md5(str(wav_path.parent).encode("utf-8")).hexdigest()[:10]


def _split_bucket(speaker_id: str) -> str:
    """Deterministic speaker-aware 70/15/15 split."""
    h = int(hashlib.sha1(speaker_id.encode("utf-8")).hexdigest()[:8], 16) % 100
    if h < 70:
        return "train"
    if h < 85:
        return "val"
    return "test"


def _slice_long_wav_to_1s(wav_path: Path, max_chunks: int = 20) -> List[np.ndarray]:
    """Load a long background/music/noise WAV and slice into non-overlapping 1.0s clips."""
    try:
        audio = load_wav_16k_mono(wav_path, target_len=None)
    except Exception:
        return []
    if audio.size < CLIP_SAMPLES:
        return [normalize_audio_length(audio, CLIP_SAMPLES)]
    chunks: List[np.ndarray] = []
    for start in range(0, audio.size - CLIP_SAMPLES + 1, CLIP_SAMPLES):
        chunks.append(audio[start : start + CLIP_SAMPLES])
        if len(chunks) >= max_chunks:
            break
    return chunks


def _augment_real_clip(
    audio: np.ndarray,
    noise_pool: Sequence[np.ndarray],
    rng: np.random.Generator,
) -> np.ndarray:
    """Real-audio-only augmentation: time shift, gain jitter, and real background mixing."""
    x = audio.copy()
    # Time shift +/- 100 ms (1600 samples)
    shift = int(rng.integers(-1600, 1601))
    if shift > 0:
        x = np.concatenate([np.zeros(shift, dtype=np.float32), x[:-shift]])
    elif shift < 0:
        x = np.concatenate([x[-shift:], np.zeros(-shift, dtype=np.float32)])

    # Gain jitter
    gain = float(rng.uniform(0.7, 1.25))
    x = x * gain

    # Mix real MUSAN / background noise at 5..20 dB SNR
    if noise_pool and rng.random() < 0.75:
        noise = noise_pool[int(rng.integers(0, len(noise_pool)))]
        sig_rms = float(np.sqrt(np.mean(x * x) + 1e-9))
        noise_rms = float(np.sqrt(np.mean(noise * noise) + 1e-9))
        if noise_rms > 1e-6 and sig_rms > 1e-5:
            snr_db = float(rng.uniform(5.0, 20.0))
            target_noise_rms = sig_rms / (10.0 ** (snr_db / 20.0))
            x = x + noise * (target_noise_rms / noise_rms)

    return np.clip(x, -1.0, 1.0).astype(np.float32)


def _collect_custom_clips(custom_dir: Path) -> Dict[str, List[Tuple[np.ndarray, str]]]:
    clips: Dict[str, List[Tuple[np.ndarray, str]]] = defaultdict(list)
    for label in ("help", "confirm", "cancel", "unknown", "background"):
        ldir = custom_dir / label
        if not ldir.exists():
            continue
        for wav in sorted(ldir.glob("*.wav")):
            try:
                audio = load_wav_16k_mono(wav, target_len=CLIP_SAMPLES)
                spk = _speaker_id_from_path(wav, "custom")
                clips[label].append((audio, f"custom_{spk}"))
            except Exception as exc:
                print(f"[WARN] Skipping {wav}: {exc}")
    return clips


def _collect_gsc_clips(
    gsc_dir: Path,
    max_unknown: int = 1200,
    seed: int = 42,
) -> Tuple[List[Tuple[np.ndarray, str]], List[np.ndarray]]:
    """Collect unknown speech clips + real _background_noise_ chunks from GSC v2."""
    unknown_clips: List[Tuple[np.ndarray, str]] = []
    bg_chunks: List[np.ndarray] = []

    if not gsc_dir.exists():
        return unknown_clips, bg_chunks

    bg_dir = gsc_dir / "_background_noise_"
    if bg_dir.exists():
        for wav in sorted(bg_dir.glob("*.wav")):
            bg_chunks.extend(_slice_long_wav_to_1s(wav, max_chunks=40))

    rng = np.random.default_rng(seed)
    candidate_paths: List[Path] = []
    for sub in sorted(gsc_dir.iterdir()):
        if not sub.is_dir() or sub.name.startswith("_"):
            continue
        candidate_paths.extend(sorted(sub.glob("*.wav")))

    if candidate_paths:
        rng.shuffle(candidate_paths)
        for wav in candidate_paths[:max_unknown]:
            try:
                audio = load_wav_16k_mono(wav, target_len=CLIP_SAMPLES)
                spk = _speaker_id_from_path(wav, "gsc")
                unknown_clips.append((audio, f"gsc_{spk}"))
            except Exception:
                continue

    return unknown_clips, bg_chunks


def _collect_musan_clips(
    musan_dir: Path,
    max_chunks_per_cat: int = 400,
) -> Tuple[List[np.ndarray], List[Tuple[np.ndarray, str]]]:
    """Collect MUSAN music/noise as negative background + MUSAN speech as unknown babble."""
    neg_bg: List[np.ndarray] = []
    speech_unknown: List[Tuple[np.ndarray, str]] = []

    if not musan_dir.exists():
        return neg_bg, speech_unknown

    for cat in ("music", "noise"):
        cdir = musan_dir / cat
        if not cdir.exists():
            continue
        collected = 0
        for wav in sorted(cdir.rglob("*.wav")):
            chunks = _slice_long_wav_to_1s(wav, max_chunks=5)
            for ch in chunks:
                neg_bg.append(ch)
                collected += 1
                if collected >= max_chunks_per_cat:
                    break
            if collected >= max_chunks_per_cat:
                break

    sdir = musan_dir / "speech"
    if sdir.exists():
        collected = 0
        for wav in sorted(sdir.rglob("*.wav")):
            spk = _speaker_id_from_path(wav, "musan")
            chunks = _slice_long_wav_to_1s(wav, max_chunks=3)
            for ch in chunks:
                speech_unknown.append((ch, f"musan_{spk}"))
                collected += 1
                if collected >= max_chunks_per_cat:
                    break
            if collected >= max_chunks_per_cat:
                break

    return neg_bg, speech_unknown


def _build_stage_npz(
    stage_name: str,
    class_names: Tuple[str, ...],
    raw_by_class: Dict[str, List[Tuple[np.ndarray, str]]],
    noise_pool: Sequence[np.ndarray],
    augments_per_target: int,
    augments_per_neg: int,
    target_classes: Sequence[str],
    out_npz: Path,
    rng: np.random.Generator,
) -> Dict:
    splits: Dict[str, Dict[str, List]] = {
        "train": {"X": [], "y": []},
        "val": {"X": [], "y": []},
        "test": {"X": [], "y": []},
    }
    counts: Dict[str, Dict[str, int]] = {s: {c: 0 for c in class_names} for s in splits}

    # Ensure every speaker group has at least train/val/test representation if only 1 speaker exists
    for cls_idx, cls_name in enumerate(class_names):
        items = list(raw_by_class.get(cls_name, []))
        if not items:
            continue

        unique_speakers = {spk for _, spk in items}
        n_aug = augments_per_target if cls_name in target_classes else augments_per_neg

        for idx, (audio, spk) in enumerate(items):
            if len(unique_speakers) <= 2:
                # Single/dual speaker custom dataset: deterministic clip-level stratified 70/15/15
                mod = idx % 20
                split = "train" if mod < 14 else ("val" if mod < 17 else "test")
            else:
                split = _split_bucket(spk)

            spec = compute_log_mel_spectrogram(audio)
            splits[split]["X"].append(spec)
            splits[split]["y"].append(cls_idx)
            counts[split][cls_name] += 1

            if split == "train" and n_aug > 0:
                for _ in range(n_aug):
                    aug_audio = _augment_real_clip(audio, noise_pool, rng)
                    splits["train"]["X"].append(compute_log_mel_spectrogram(aug_audio))
                    splits["train"]["y"].append(cls_idx)
                    counts["train"][cls_name] += 1

    arrays: Dict[str, np.ndarray] = {}
    for split_name, bucket in splits.items():
        if bucket["X"]:
            x_arr = np.stack(bucket["X"], axis=0).astype(np.float32)
            y_arr = np.asarray(bucket["y"], dtype=np.int64)
            perm = rng.permutation(len(y_arr))
            arrays[f"X_{split_name}"] = x_arr[perm]
            arrays[f"y_{split_name}"] = y_arr[perm]
        else:
            arrays[f"X_{split_name}"] = np.zeros((0, 49, 40), dtype=np.float32)
            arrays[f"y_{split_name}"] = np.zeros((0,), dtype=np.int64)

    arrays["labels"] = np.asarray(class_names)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_npz, **arrays)
    print(f"[{stage_name}] Saved {out_npz} -> splits={counts}")
    return counts


def prepare_datasets(
    custom_dir: Path,
    gsc_dir: Path,
    musan_dir: Path,
    out_dir: Path,
    augments_per_target: int = 8,
    augments_per_neg: int = 1,
    max_gsc_unknown: int = 1200,
    seed: int = 42,
    allow_partial: bool = False,
) -> Dict:
    rng = np.random.default_rng(seed)

    custom = _collect_custom_clips(custom_dir)
    gsc_unknown, gsc_bg = _collect_gsc_clips(gsc_dir, max_unknown=max_gsc_unknown, seed=seed)
    musan_bg, musan_speech = _collect_musan_clips(musan_dir)

    missing_targets = [lbl for lbl in ("help", "confirm", "cancel") if len(custom.get(lbl, [])) == 0]
    if missing_targets and not allow_partial:
        cmds = "\n".join(
            f"  python data_tools/record_voice.py --label {lbl} --count 25 --speaker speaker01"
            for lbl in missing_targets
        )
        raise SystemExit(
            "Missing real human recordings for required emergency keywords: "
            f"{missing_targets}.\nRecord them first (no synthetic TTS allowed):\n{cmds}"
        )

    # Combine real background noise clips from custom + GSC _background_noise_ + MUSAN music/noise
    noise_pool: List[np.ndarray] = [a for a, _ in custom.get("background", [])] + gsc_bg + musan_bg
    if not noise_pool:
        # Pure quiet room floor fallback if no noise dataset was provided
        noise_pool = [np.zeros(CLIP_SAMPLES, dtype=np.float32)]

    bg_items: List[Tuple[np.ndarray, str]] = list(custom.get("background", []))
    for idx, ch in enumerate(gsc_bg):
        bg_items.append((ch, f"gscbg_{idx}"))
    for idx, ch in enumerate(musan_bg):
        bg_items.append((ch, f"musanbg_{idx}"))

    unknown_items: List[Tuple[np.ndarray, str]] = list(custom.get("unknown", [])) + gsc_unknown + musan_speech

    stage1_raw = {
        "background": bg_items,
        "unknown": unknown_items,
        "help": list(custom.get("help", [])),
    }
    stage2_raw = {
        "silence_unknown": bg_items[:400] + unknown_items[:600],
        "confirm": list(custom.get("confirm", [])),
        "cancel": list(custom.get("cancel", [])),
    }

    s1_counts = _build_stage_npz(
        stage_name="STAGE1",
        class_names=STAGE1_LABELS,
        raw_by_class=stage1_raw,
        noise_pool=noise_pool,
        augments_per_target=augments_per_target,
        augments_per_neg=augments_per_neg,
        target_classes=("help",),
        out_npz=out_dir / "stage1_features.npz",
        rng=rng,
    )
    s2_counts = _build_stage_npz(
        stage_name="STAGE2",
        class_names=STAGE2_LABELS,
        raw_by_class=stage2_raw,
        noise_pool=noise_pool,
        augments_per_target=augments_per_target,
        augments_per_neg=augments_per_neg,
        target_classes=("confirm", "cancel"),
        out_npz=out_dir / "stage2_features.npz",
        rng=rng,
    )

    manifest = {
        "sources": {
            "custom_recordings": {k: len(v) for k, v in custom.items()},
            "gsc_unknown_clips": len(gsc_unknown),
            "gsc_background_1s_chunks": len(gsc_bg),
            "musan_music_noise_1s_chunks": len(musan_bg),
            "musan_speech_1s_chunks": len(musan_speech),
        },
        "stage1_splits": s1_counts,
        "stage2_splits": s2_counts,
        "synthetic_tts_used": False,
    }
    manifest_path = out_dir / "dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[DATASET] Wrote manifest -> {manifest_path}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Combine Google Speech Commands v2 + MUSAN + real human voice recordings."
    )
    parser.add_argument("--custom-dir", type=Path, default=DEFAULT_CUSTOM_DIR)
    parser.add_argument("--gsc-dir", type=Path, default=DEFAULT_GSC_DIR)
    parser.add_argument("--musan-dir", type=Path, default=DEFAULT_MUSAN_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--download-gsc", action="store_true", help="Download GSC v0.02 (~2.4 GB)")
    parser.add_argument("--download-musan", action="store_true", help="Download MUSAN (~11 GB)")
    parser.add_argument("--augments-per-clip", type=int, default=8, help="Real-audio augments for target keywords")
    parser.add_argument("--max-gsc-unknown", type=int, default=1200)
    parser.add_argument("--allow-partial", action="store_true", help="Allow building even if a keyword is empty")
    args = parser.parse_args()

    if args.download_gsc and not args.gsc_dir.exists():
        _download_and_extract(GSC_URL, args.gsc_dir, "speech_commands_v0.02.tar.gz")
    if args.download_musan and not args.musan_dir.exists():
        _download_and_extract(MUSAN_URL, args.musan_dir.parent, "musan.tar.gz")

    prepare_datasets(
        custom_dir=args.custom_dir,
        gsc_dir=args.gsc_dir,
        musan_dir=args.musan_dir,
        out_dir=args.out_dir,
        augments_per_target=args.augments_per_clip,
        max_gsc_unknown=args.max_gsc_unknown,
        allow_partial=args.allow_partial,
    )


if __name__ == "__main__":
    main()
