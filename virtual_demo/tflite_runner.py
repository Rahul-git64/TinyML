"""INT8 TFLite inference runner + Mock Fallback for the Streamlit Digital Twin."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import tensorflow as tf

STAGE1_LABELS: Tuple[str, ...] = ("background", "unknown", "help")
STAGE2_LABELS: Tuple[str, ...] = ("silence_unknown", "confirm", "cancel")

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STAGE1_TFLITE = REPO_ROOT / "models" / "stage1_wake_int8.tflite"
DEFAULT_STAGE2_TFLITE = REPO_ROOT / "models" / "stage2_verify_int8.tflite"
DEFAULT_METADATA_JSON = REPO_ROOT / "models" / "model_metadata.json"


@dataclass
class InferenceOutput:
    probs: Dict[str, float]
    latency_ms: float
    is_mock: bool
    input_dtype: str
    output_dtype: str
    flash_kb: float
    arena_kb: float


def estimate_interpreter_arena_kb(interpreter: tf.lite.Interpreter) -> float:
    """Estimate peak tensor arena SRAM in KB from TFLite tensor allocation details."""
    tensor_details = interpreter.get_tensor_details()
    total_activation_bytes = 0
    for td in tensor_details:
        name = td.get("name", "")
        shape = td.get("shape", ())
        dtype = np.dtype(td.get("dtype", np.int8))
        # Constant weights reside in Flash; activations are intermediate tensors
        if any(k in name.lower() for k in ("kernel", "bias", "const", "weight", "arith.constant")):
            continue
        n_elem = int(np.prod(shape)) if len(shape) > 0 else 1
        total_activation_bytes += n_elem * dtype.itemsize
    # Conservative 40% peak liveness + 6 KB TFLM runtime overhead
    return round((total_activation_bytes * 0.45 + 6144.0) / 1024.0, 2)


class Int8TFLiteModel:
    """Runs an actual exported INT8 `.tflite` file via `tf.lite.Interpreter`."""

    def __init__(self, tflite_path: Path, labels: Sequence[str]) -> None:
        self.tflite_path = Path(tflite_path)
        self.labels = tuple(labels)
        if not self.tflite_path.exists():
            raise FileNotFoundError(f"INT8 TFLite model not found: {self.tflite_path}")

        self.flash_bytes = self.tflite_path.stat().st_size
        self.flash_kb = round(self.flash_bytes / 1024.0, 2)

        self.interpreter = tf.lite.Interpreter(model_path=str(self.tflite_path))
        self.interpreter.allocate_tensors()

        in_det = self.interpreter.get_input_details()[0]
        out_det = self.interpreter.get_output_details()[0]

        if np.dtype(in_det["dtype"]) != np.int8:
            raise ValueError(f"Expected int8 input tensor, got {in_det['dtype']}")
        if np.dtype(out_det["dtype"]) != np.int8:
            raise ValueError(f"Expected int8 output tensor, got {out_det['dtype']}")

        self.in_index = int(in_det["index"])
        self.out_index = int(out_det["index"])
        self.in_shape = tuple(int(d) for d in in_det["shape"])
        self.in_scale, self.in_zp = in_det["quantization"]
        self.out_scale, self.out_zp = out_det["quantization"]
        self.arena_kb = estimate_interpreter_arena_kb(self.interpreter)

    def predict_batch(self, specs: np.ndarray) -> np.ndarray:
        """Run INT8 inference over a batch `(N, 49, 40)` or `(N, 49, 40, 1)`."""
        arr = np.asarray(specs, dtype=np.float32)
        if arr.ndim == 3:
            arr = arr[..., np.newaxis]

        n = arr.shape[0]
        out = np.zeros((n, len(self.labels)), dtype=np.float32)
        for i in range(n):
            q_in = np.round(arr[i : i + 1] / float(self.in_scale) + float(self.in_zp))
            q_in = np.clip(q_in, -128, 127).astype(np.int8)
            self.interpreter.set_tensor(self.in_index, q_in)
            self.interpreter.invoke()
            q_out = self.interpreter.get_tensor(self.out_index)[0].astype(np.float32)
            probs = (q_out - float(self.out_zp)) * float(self.out_scale)
            probs = np.clip(probs, 0.0, 1.0)
            s = float(probs.sum())
            if s > 1e-6:
                probs /= s
            out[i] = probs
        return out

    def predict(self, spec_49x40: np.ndarray) -> InferenceOutput:
        """Run single-window INT8 inference and measure latency."""
        t0 = time.perf_counter()
        probs_arr = self.predict_batch(spec_49x40[np.newaxis, ...])[0]
        latency_ms = (time.perf_counter() - t0) * 1000.0
        probs = {lbl: float(probs_arr[idx]) for idx, lbl in enumerate(self.labels)}
        return InferenceOutput(
            probs=probs,
            latency_ms=round(latency_ms, 2),
            is_mock=False,
            input_dtype="int8",
            output_dtype="int8",
            flash_kb=self.flash_kb,
            arena_kb=self.arena_kb,
        )


class MockKWSModel:
    """Mock fallback model for immediate UI & FSM testing before real training."""

    def __init__(self, labels: Sequence[str], flash_kb: float = 21.4, arena_kb: float = 34.0) -> None:
        self.labels = tuple(labels)
        self.flash_kb = flash_kb
        self.arena_kb = arena_kb
        self._queued_label: Optional[str] = None
        self._queued_remaining: int = 0

    def trigger(self, label: str, repeat_ticks: int = 2) -> None:
        if label not in self.labels:
            raise ValueError(f"Unknown mock label '{label}' for {self.labels}")
        self._queued_label = label
        self._queued_remaining = max(1, int(repeat_ticks))

    def predict(self, spec_49x40: np.ndarray, rms: float = 0.01) -> InferenceOutput:
        t0 = time.perf_counter()
        probs = {lbl: 0.02 for lbl in self.labels}

        if self._queued_label is not None and self._queued_remaining > 0:
            target = self._queued_label
            self._queued_remaining -= 1
            if self._queued_remaining == 0:
                self._queued_label = None
            for lbl in self.labels:
                probs[lbl] = 0.93 if lbl == target else 0.035
        else:
            # RMS-reactive realistic idle output (never falsely triggers help/confirm/cancel)
            energy = float(np.clip(rms * 8.0, 0.0, 0.45))
            if self.labels == STAGE1_LABELS:
                probs["background"] = round(0.92 - energy * 0.7, 4)
                probs["unknown"] = round(0.06 + energy * 0.65, 4)
                probs["help"] = round(0.02 + energy * 0.05, 4)
            else:
                probs["silence_unknown"] = round(0.94 - energy * 0.15, 4)
                probs["confirm"] = round(0.03 + energy * 0.075, 4)
                probs["cancel"] = round(0.03 + energy * 0.075, 4)

        total = sum(probs.values())
        probs = {k: v / total for k, v in probs.items()}
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return InferenceOutput(
            probs=probs,
            latency_ms=round(latency_ms, 2),
            is_mock=True,
            input_dtype="mock_int8",
            output_dtype="mock_int8",
            flash_kb=self.flash_kb,
            arena_kb=self.arena_kb,
        )


class KWSModelManager:
    """Manages Stage 1 and Stage 2 INT8 TFLite models with Mock fallback."""

    def __init__(
        self,
        stage1_path: Path = DEFAULT_STAGE1_TFLITE,
        stage2_path: Path = DEFAULT_STAGE2_TFLITE,
        metadata_path: Path = DEFAULT_METADATA_JSON,
    ) -> None:
        self.stage1_path = Path(stage1_path)
        self.stage2_path = Path(stage2_path)
        self.metadata_path = Path(metadata_path)
        self.stage1_tflite: Optional[Int8TFLiteModel] = None
        self.stage2_tflite: Optional[Int8TFLiteModel] = None
        self.mock_stage1 = MockKWSModel(STAGE1_LABELS)
        self.mock_stage2 = MockKWSModel(STAGE2_LABELS)
        self.load_errors: Dict[str, str] = {}
        self.metadata: Dict = {}
        self.reload()

    def reload(self) -> None:
        self.load_errors.clear()
        self.stage1_tflite = None
        self.stage2_tflite = None

        if self.metadata_path.exists():
            try:
                self.metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
            except Exception:
                self.metadata = {}

        for key, path, labels in (
            ("stage1", self.stage1_path, STAGE1_LABELS),
            ("stage2", self.stage2_path, STAGE2_LABELS),
        ):
            try:
                model = Int8TFLiteModel(path, labels)
                if key == "stage1":
                    self.stage1_tflite = model
                else:
                    self.stage2_tflite = model
            except Exception as exc:
                self.load_errors[key] = str(exc)

    @property
    def real_models_ready(self) -> bool:
        return self.stage1_tflite is not None and self.stage2_tflite is not None

    @property
    def trained_on_real_data(self) -> bool:
        s1 = self.metadata.get("stage1", {}).get("trained_on_real_data", False)
        s2 = self.metadata.get("stage2", {}).get("trained_on_real_data", False)
        return bool(s1 and s2)

    def predict(
        self,
        stage: int,
        spec_49x40: np.ndarray,
        use_mock: bool = False,
        rms: float = 0.01,
    ) -> InferenceOutput:
        if stage not in (1, 2):
            raise ValueError(f"Invalid stage: {stage}")

        if not use_mock:
            model = self.stage1_tflite if stage == 1 else self.stage2_tflite
            if model is not None:
                return model.predict(spec_49x40)

        mock = self.mock_stage1 if stage == 1 else self.mock_stage2
        return mock.predict(spec_49x40, rms=rms)
