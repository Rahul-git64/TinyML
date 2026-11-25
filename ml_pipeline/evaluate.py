"""Evaluate exported INT8 `.tflite` models: Confusion Matrix, FAR/FRR, Flash KB, SRAM Arena KB."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml_pipeline.model_arch import MODELS_DIR, make_smoke_test_dataset  # noqa: E402
from virtual_demo.tflite_runner import STAGE1_LABELS, STAGE2_LABELS, Int8TFLiteModel  # noqa: E402


def _save_cm_plot(cm: np.ndarray, labels: Sequence[str], title: str, out_png: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 4.4), dpi=130)
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        title=title,
        ylabel="True Label",
        xlabel="Predicted Label",
    )
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(int(cm[i, j]), "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png)
    plt.close(fig)


def evaluate_stage(
    stage: int,
    tflite_path: Path,
    labels: Tuple[str, ...],
    dataset_npz: Path,
    target_positive_label: str,
    default_threshold: float,
    cm_out_png: Path,
    smoke_test: bool = False,
) -> Dict:
    model = Int8TFLiteModel(tflite_path, labels)

    if smoke_test:
        _, _, _, _, X_test, y_test = make_smoke_test_dataset(stage=stage)
    else:
        data = np.load(dataset_npz)
        X_test = data["X_test"]
        y_test = data["y_test"]
        if len(X_test) == 0:
            X_test, y_test = data["X_val"], data["y_val"]

    probs = model.predict_batch(X_test)
    y_pred = np.argmax(probs, axis=1)

    acc = float(accuracy_score(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred, labels=list(range(len(labels))))
    cls_report = classification_report(
        y_test,
        y_pred,
        labels=list(range(len(labels))),
        target_names=list(labels),
        output_dict=True,
        zero_division=0,
    )

    pos_idx = labels.index(target_positive_label)
    is_pos = y_test == pos_idx
    is_neg = ~is_pos

    threshold_sweep = {}
    for thr in (0.50, 0.65, default_threshold, 0.85, 0.90):
        pred_pos = probs[:, pos_idx] >= thr
        far = float(np.mean(pred_pos[is_neg])) if np.any(is_neg) else 0.0
        frr = float(np.mean(~pred_pos[is_pos])) if np.any(is_pos) else 0.0
        threshold_sweep[f"{thr:.2f}"] = {"FAR": round(far, 4), "FRR": round(frr, 4)}

    default_key = f"{default_threshold:.2f}"
    far_default = threshold_sweep[default_key]["FAR"]
    frr_default = threshold_sweep[default_key]["FRR"]

    _save_cm_plot(
        cm,
        labels,
        f"Stage {stage} INT8 Confusion Matrix (Acc={acc:.2%}, FAR={far_default:.2%})",
        cm_out_png,
    )

    print(f"\n=== Stage {stage} INT8 Evaluation ({tflite_path.name}) ===")
    print(f"Flash Size          : {model.flash_kb:.2f} KB ({model.flash_bytes} bytes)")
    print(f"Estimated SRAM Arena: {model.arena_kb:.2f} KB")
    print(f"Accuracy (Test)     : {acc:.2%}")
    print(f"FAR @ {default_threshold:.2f} ('{target_positive_label}'): {far_default:.2%}")
    print(f"FRR @ {default_threshold:.2f} ('{target_positive_label}'): {frr_default:.2%}")
    print("Confusion Matrix (rows=true, cols=pred):", labels)
    print(cm)

    return {
        "stage": stage,
        "tflite_path": str(tflite_path),
        "labels": list(labels),
        "flash_bytes": model.flash_bytes,
        "flash_kb": model.flash_kb,
        "estimated_sram_arena_kb": model.arena_kb,
        "test_samples": int(len(y_test)),
        "accuracy": round(acc, 4),
        "default_threshold": default_threshold,
        "far_at_default": far_default,
        "frr_at_default": frr_default,
        "threshold_sweep": threshold_sweep,
        "confusion_matrix": cm.tolist(),
        "classification_report": cls_report,
        "confusion_matrix_png": str(cm_out_png),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate INT8 Stage 1 & Stage 2 TFLite models")
    parser.add_argument("--smoke-test", action="store_true", help="Evaluate on smoke-test data")
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data" / "processed")
    args = parser.parse_args()

    s1 = evaluate_stage(
        stage=1,
        tflite_path=MODELS_DIR / "stage1_wake_int8.tflite",
        labels=STAGE1_LABELS,
        dataset_npz=args.data_dir / "stage1_features.npz",
        target_positive_label="help",
        default_threshold=0.80,
        cm_out_png=MODELS_DIR / "eval_stage1_cm.png",
        smoke_test=args.smoke_test,
    )
    s2 = evaluate_stage(
        stage=2,
        tflite_path=MODELS_DIR / "stage2_verify_int8.tflite",
        labels=STAGE2_LABELS,
        dataset_npz=args.data_dir / "stage2_features.npz",
        target_positive_label="confirm",
        default_threshold=0.75,
        cm_out_png=MODELS_DIR / "eval_stage2_cm.png",
        smoke_test=args.smoke_test,
    )

    report = {"stage1": s1, "stage2": s2, "smoke_test": args.smoke_test}
    report_path = MODELS_DIR / "eval_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n[EVAL] Full report written -> {report_path}")


if __name__ == "__main__":
    main()
