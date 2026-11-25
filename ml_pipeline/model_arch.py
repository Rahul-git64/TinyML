"""DS-CNN architecture, full INT8 TFLite conversion, C-array export, and smoke-test acoustic data."""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

from ml_pipeline.dsp import CLIP_SAMPLES, NUM_FRAMES, NUM_MEL_BINS, SAMPLE_RATE, compute_log_mel_spectrogram

REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = REPO_ROOT / "models"
FIRMWARE_INCLUDE_DIR = REPO_ROOT / "firmware_esp32s3" / "include"


def build_dscnn_model(
    input_shape: Tuple[int, int, int] = (NUM_FRAMES, NUM_MEL_BINS, 1),
    num_classes: int = 3,
    name: str = "kws_dscnn",
) -> tf.keras.Model:
    """Lightweight Depthwise-Separable CNN using strictly ESP-NN accelerated ops (< 30 KB INT8)."""
    inp = tf.keras.Input(shape=input_shape, name="log_mel_input")

    # Initial Conv2D: (49, 40, 1) -> (25, 20, 20)
    x = tf.keras.layers.Conv2D(20, (3, 3), strides=(2, 2), padding="same", use_bias=False)(inp)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    # DS Block 1: (25, 20, 20) -> (13, 10, 32)
    x = tf.keras.layers.DepthwiseConv2D((3, 3), strides=(2, 2), padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(32, (1, 1), padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    # DS Block 2: (13, 10, 32) -> (7, 5, 48)
    x = tf.keras.layers.DepthwiseConv2D((3, 3), strides=(2, 2), padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(48, (1, 1), padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    # DS Block 3: (7, 5, 48) -> (7, 5, 48)
    x = tf.keras.layers.DepthwiseConv2D((3, 3), strides=(1, 1), padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(48, (1, 1), padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    # DS Block 4: (7, 5, 48) -> (7, 5, 48)
    x = tf.keras.layers.DepthwiseConv2D((3, 3), strides=(1, 1), padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(48, (1, 1), padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    # Explicit AveragePooling2D (avoids MEAN op in TFLM/ESP-NN)
    x = tf.keras.layers.AveragePooling2D(pool_size=(7, 5), strides=(1, 1), padding="valid")(x)
    x = tf.keras.layers.Reshape((48,))(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    x = tf.keras.layers.Dense(num_classes)(x)
    out = tf.keras.layers.Softmax(name="class_probs")(x)

    return tf.keras.Model(inputs=inp, outputs=out, name=name)


def spec_augment(
    x: np.ndarray,
    freq_mask_max: int = 5,
    time_mask_max: int = 6,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Apply SpecAugment frequency & time masking on a (N, 49, 40, 1) batch."""
    if rng is None:
        rng = np.random.default_rng()
    out = x.copy()
    n, t_len, f_len = out.shape[0], out.shape[1], out.shape[2]
    fill = float(np.min(out))
    for i in range(n):
        if rng.random() < 0.7:
            f = int(rng.integers(1, freq_mask_max + 1))
            f0 = int(rng.integers(0, max(1, f_len - f + 1)))
            out[i, :, f0 : f0 + f, ...] = fill
        if rng.random() < 0.7:
            t = int(rng.integers(1, time_mask_max + 1))
            t0 = int(rng.integers(0, max(1, t_len - t + 1)))
            out[i, t0 : t0 + t, :, ...] = fill
    return out


def compute_balanced_class_weights(y: np.ndarray, num_classes: int) -> Dict[int, float]:
    classes = np.arange(num_classes)
    present = np.unique(y)
    if len(present) == num_classes:
        weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
        return {int(c): float(w) for c, w in zip(classes, weights)}
    return {int(c): 1.0 for c in classes}


def estimate_tensor_arena_kb(tflite_path: Path) -> float:
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    total_activation_bytes = 0
    for td in interpreter.get_tensor_details():
        name = td.get("name", "")
        shape = td.get("shape", ())
        dtype = np.dtype(td.get("dtype", np.int8))
        if any(k in name.lower() for k in ("kernel", "bias", "const", "weight", "arith.constant")):
            continue
        n_elem = int(np.prod(shape)) if len(shape) > 0 else 1
        total_activation_bytes += n_elem * dtype.itemsize
    return round((total_activation_bytes * 0.45 + 6144.0) / 1024.0, 2)


def convert_to_int8_tflite(
    model: tf.keras.Model,
    rep_data: np.ndarray,
    output_tflite_path: Path,
    max_kb: float = 30.0,
) -> Dict:
    """Apply full INT8 quantization, verify int8 I/O dtypes, and enforce < max_kb Flash budget."""
    x_rep = np.asarray(rep_data, dtype=np.float32)
    if x_rep.ndim == 3:
        x_rep = x_rep[..., np.newaxis]
    n_rep = min(150, len(x_rep))

    def representative_dataset():
        for i in range(n_rep):
            yield [x_rep[i : i + 1]]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_bytes = converter.convert()
    out_path = Path(output_tflite_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(tflite_bytes)

    size_bytes = len(tflite_bytes)
    size_kb = round(size_bytes / 1024.0, 2)
    if size_kb >= max_kb:
        raise AssertionError(
            f"Quantized model {out_path.name} is {size_kb} KB, exceeding < {max_kb} KB budget!"
        )

    interpreter = tf.lite.Interpreter(model_path=str(out_path))
    interpreter.allocate_tensors()
    in_det = interpreter.get_input_details()[0]
    out_det = interpreter.get_output_details()[0]
    if np.dtype(in_det["dtype"]) != np.int8 or np.dtype(out_det["dtype"]) != np.int8:
        raise AssertionError(
            f"Expected int8 I/O tensors, got input={in_det['dtype']} output={out_det['dtype']}"
        )

    arena_kb = estimate_tensor_arena_kb(out_path)
    return {
        "tflite_path": str(out_path),
        "flash_bytes": size_bytes,
        "flash_kb": size_kb,
        "arena_kb": arena_kb,
        "input_dtype": "int8",
        "output_dtype": "int8",
        "input_quant": [float(in_det["quantization"][0]), int(in_det["quantization"][1])],
        "output_quant": [float(out_det["quantization"][0]), int(out_det["quantization"][1])],
    }


def export_c_header(
    tflite_path: Path,
    header_path: Path,
    var_name: str,
    labels: Sequence[str],
    firmware_copy_path: Path | None = None,
) -> Path:
    """Export `.tflite` binary as an `xxd`-compatible `alignas(16)` C array header."""
    data = Path(tflite_path).read_bytes()
    lines: List[str] = []
    for i in range(0, len(data), 12):
        chunk = ", ".join(f"0x{b:02x}" for b in data[i : i + 12])
        lines.append(f"  {chunk}")

    labels_csv = ", ".join(f'"{lbl}"' for lbl in labels)
    content = (
        "#pragma once\n"
        "// Auto-generated INT8 TFLite FlatBuffer C-array (xxd -i compatible)\n"
        "#include <stdint.h>\n\n"
        f"// Labels: [{labels_csv}]\n"
        f"// Size: {len(data)} bytes ({len(data) / 1024.0:.2f} KB)\n\n"
        f"alignas(16) const unsigned char {var_name}[] = {{\n"
        + ",\n".join(lines)
        + "\n};\n"
        f"const unsigned int {var_name}_len = {len(data)};\n"
    )

    out = Path(header_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(content, encoding="utf-8")

    if firmware_copy_path is not None:
        fw_out = Path(firmware_copy_path)
        fw_out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(out, fw_out)
    return out


def update_model_metadata(stage_key: str, info: Dict) -> Path:
    meta_path = MODELS_DIR / "model_metadata.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta = {}
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            meta = {}
    info["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    meta[stage_key] = info
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta_path


def make_smoke_test_dataset(
    stage: int,
    n_per_class: int = 100,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Non-speech acoustic test signals strictly for `--smoke-test` pipeline verification."""
    rng = np.random.default_rng(seed + stage * 100)
    t = np.linspace(0.0, 1.0, CLIP_SAMPLES, endpoint=False, dtype=np.float32)

    X_all, y_all = [], []
    for cls_idx in range(3):
        for _ in range(n_per_class):
            noise = 0.03 * rng.standard_normal(CLIP_SAMPLES).astype(np.float32)
            if cls_idx == 0:
                # Background / silence-like low-amplitude noise
                sig = 0.02 * rng.standard_normal(CLIP_SAMPLES).astype(np.float32)
            elif cls_idx == 1:
                # Class 1: rising multi-tone / unknown pattern
                f0 = float(rng.uniform(350.0, 520.0))
                sig = 0.35 * np.sin(2.0 * np.pi * f0 * t) + 0.2 * np.sin(2.0 * np.pi * (f0 * 2.1) * t) + noise
            else:
                # Class 2: modulated dual-harmonic burst (distinct spectral signature)
                f0 = float(rng.uniform(900.0, 1400.0))
                env = np.exp(-((t - 0.5) ** 2) / 0.04).astype(np.float32)
                sig = 0.45 * env * np.sin(2.0 * np.pi * f0 * t) + noise
            spec = compute_log_mel_spectrogram(np.clip(sig, -1.0, 1.0))
            X_all.append(spec)
            y_all.append(cls_idx)

    X = np.stack(X_all, axis=0).astype(np.float32)
    y = np.asarray(y_all, dtype=np.int64)
    perm = rng.permutation(len(y))
    X, y = X[perm], y[perm]

    n_total = len(y)
    n_train = int(0.70 * n_total)
    n_val = int(0.15 * n_total)
    return (
        X[:n_train],
        y[:n_train],
        X[n_train : n_train + n_val],
        y[n_train : n_train + n_val],
        X[n_train + n_val :],
        y[n_train + n_val :],
    )


def train_and_export_stage(
    stage: int,
    labels: Tuple[str, ...],
    dataset_npz: Path,
    tflite_out: Path,
    header_out: Path,
    firmware_header_out: Path,
    c_var_name: str,
    epochs: int = 35,
    batch_size: int = 32,
    smoke_test: bool = False,
) -> Dict:
    """Shared training + INT8 quantization + C-header export pipeline for Stage 1 & 2."""
    tf.keras.utils.set_random_seed(42 + stage)

    if smoke_test:
        print(f"[STAGE {stage}] Running --smoke-test acoustic pipeline verification (non-speech).")
        X_train, y_train, X_val, y_val, X_test, y_test = make_smoke_test_dataset(stage=stage)
    else:
        if not dataset_npz.exists():
            raise SystemExit(
                f"Dataset file not found: {dataset_npz}\n"
                "Run `python data_tools/prepare_dataset.py` first (or pass `--smoke-test` to verify the pipeline)."
            )
        data = np.load(dataset_npz)
        X_train, y_train = data["X_train"], data["y_train"]
        X_val, y_val = data["X_val"], data["y_val"]
        X_test, y_test = data["X_test"], data["y_test"]

    if len(X_train) == 0:
        raise SystemExit(f"Training split in {dataset_npz} is empty.")
    if len(X_val) == 0:
        X_val, y_val = X_train[: min(32, len(X_train))], y_train[: min(32, len(y_train))]

    X_train_4d = X_train[..., np.newaxis].astype(np.float32)
    X_val_4d = X_val[..., np.newaxis].astype(np.float32)

    # Apply SpecAugment copy to training set for noise robustness
    rng = np.random.default_rng(42 + stage)
    X_aug = spec_augment(X_train_4d, rng=rng)
    X_fit = np.concatenate([X_train_4d, X_aug], axis=0)
    y_fit = np.concatenate([y_train, y_train], axis=0)

    model = build_dscnn_model(num_classes=len(labels), name=f"stage{stage}_dscnn")
    steps_per_epoch = max(1, len(X_fit) // batch_size)
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=2.5e-3,
        decay_steps=max(10, epochs * steps_per_epoch),
        alpha=0.05,
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    class_weights = compute_balanced_class_weights(y_train, len(labels))
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=6, restore_best_weights=True
        )
    ]

    history = model.fit(
        X_fit,
        y_fit,
        validation_data=(X_val_4d, y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=2,
    )

    quant_stats = convert_to_int8_tflite(
        model=model,
        rep_data=X_train_4d,
        output_tflite_path=tflite_out,
        max_kb=30.0,
    )
    export_c_header(
        tflite_path=tflite_out,
        header_path=header_out,
        var_name=c_var_name,
        labels=labels,
        firmware_copy_path=firmware_header_out,
    )

    best_val_acc = float(max(history.history.get("val_accuracy", [0.0])))
    info = {
        "stage": stage,
        "labels": list(labels),
        "trained_on_real_data": not smoke_test,
        "best_val_accuracy": round(best_val_acc, 4),
        "c_var_name": c_var_name,
        "header_path": str(header_out),
        "firmware_header_path": str(firmware_header_out),
        **quant_stats,
    }
    update_model_metadata(f"stage{stage}", info)
    print(
        f"[STAGE {stage}] Exported {tflite_out.name}: {quant_stats['flash_kb']} KB Flash "
        f"(budget < 30 KB), ~{quant_stats['arena_kb']} KB SRAM arena, val_acc={best_val_acc:.3f}"
    )
    return info
