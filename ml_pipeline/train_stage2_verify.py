"""Train Stage 2 Verification DS-CNN ('silence_unknown', 'confirm', 'cancel') and export INT8 TFLite + C header."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml_pipeline.model_arch import FIRMWARE_INCLUDE_DIR, MODELS_DIR, train_and_export_stage  # noqa: E402

STAGE2_LABELS = ("silence_unknown", "confirm", "cancel")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Stage 2 Verification INT8 DS-CNN (< 30 KB)")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=REPO_ROOT / "data" / "processed" / "stage2_features.npz",
    )
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run pipeline verification with non-speech acoustic test signals",
    )
    args = parser.parse_args()

    train_and_export_stage(
        stage=2,
        labels=STAGE2_LABELS,
        dataset_npz=args.dataset,
        tflite_out=MODELS_DIR / "stage2_verify_int8.tflite",
        header_out=MODELS_DIR / "stage2_verify_model.h",
        firmware_header_out=FIRMWARE_INCLUDE_DIR / "stage2_verify_model.h",
        c_var_name="g_stage2_verify_model_data",
        epochs=args.epochs,
        batch_size=args.batch_size,
        smoke_test=args.smoke_test,
    )


if __name__ == "__main__":
    main()
