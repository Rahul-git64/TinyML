"""End-to-end verification tests for DSP, FSM, INT8 TFLite runner, data tools, and firmware."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_tools.prepare_dataset import prepare_datasets  # noqa: E402
from ml_pipeline.dsp import (  # noqa: E402
    CLIP_SAMPLES,
    NUM_FRAMES,
    NUM_MEL_BINS,
    compute_log_mel_spectrogram,
    load_wav_16k_mono,
    save_wav_16k_mono,
    verify_streaming_parity,
)
from virtual_demo.audio_engine import ensure_verify_prompt_wav  # noqa: E402
from virtual_demo.fsm import FSMConfig, FSMState, LEDColor, TwoStageFSM  # noqa: E402
from virtual_demo.tflite_runner import (  # noqa: E402
    STAGE1_LABELS,
    STAGE2_LABELS,
    Int8TFLiteModel,
    KWSModelManager,
    MockKWSModel,
)


class TestDSPAndFSM(unittest.TestCase):
    def test_dsp_streaming_parity_and_shape(self) -> None:
        max_diff = verify_streaming_parity()
        self.assertLess(max_diff, 1e-5)
        spec = compute_log_mel_spectrogram(np.zeros(CLIP_SAMPLES, dtype=np.float32))
        self.assertEqual(spec.shape, (NUM_FRAMES, NUM_MEL_BINS))
        self.assertEqual(spec.dtype, np.float32)

    def test_fsm_confirm_cancel_timeout_and_d3(self) -> None:
        cfg = FSMConfig(
            stage1_threshold=0.80,
            stage1_consecutive_hits=2,
            stage2_confirm_threshold=0.75,
            stage2_cancel_threshold=0.75,
            verify_timeout_sec=10.0,
            prompt_grace_sec=2.2,
            post_cancel_cooldown_sec=1.0,
        )
        fsm = TwoStageFSM(cfg)
        self.assertEqual(fsm.state, FSMState.IDLE_MONITORING)
        self.assertEqual(fsm.led_color, LEDColor.GREEN)

        # Single hit should not trigger VOICE_VERIFY yet
        fsm.step({"background": 0.05, "unknown": 0.05, "help": 0.90}, now=1.0)
        self.assertEqual(fsm.state, FSMState.IDLE_MONITORING)

        # Second consecutive hit triggers VOICE_VERIFY
        tr = fsm.step({"background": 0.05, "unknown": 0.05, "help": 0.92}, now=1.2)
        self.assertTrue(tr.entered_voice_verify)
        self.assertEqual(fsm.state, FSMState.VOICE_VERIFY)
        self.assertEqual(fsm.led_color, LEDColor.YELLOW)

        # Confirm during grace window (1.2 + 1.0 < 3.4) should be ignored
        fsm.step({"silence_unknown": 0.05, "confirm": 0.95, "cancel": 0.0}, now=2.2)
        self.assertEqual(fsm.state, FSMState.VOICE_VERIFY)

        # Confirm after grace window triggers ALERT_DISPATCHED
        tr2 = fsm.step({"silence_unknown": 0.05, "confirm": 0.95, "cancel": 0.0}, now=4.0)
        self.assertTrue(tr2.entered_alert)
        self.assertEqual(fsm.state, FSMState.ALERT_DISPATCHED)
        self.assertEqual(fsm.led_color, LEDColor.RED)
        self.assertEqual(fsm.alert_reason, "Voice Confirmed")

        # Caregiver acknowledge resets to IDLE_MONITORING
        tr3 = fsm.acknowledge_alert(now=5.0)
        self.assertTrue(tr3.returned_idle)
        self.assertEqual(fsm.state, FSMState.IDLE_MONITORING)
        self.assertEqual(fsm.led_color, LEDColor.GREEN)

        # Re-enter VOICE_VERIFY after cooldown, then test voice cancel
        fsm.step({"background": 0.0, "unknown": 0.0, "help": 0.95}, now=7.0)
        fsm.step({"background": 0.0, "unknown": 0.0, "help": 0.95}, now=7.2)
        self.assertEqual(fsm.state, FSMState.VOICE_VERIFY)
        tr_cancel = fsm.step({"silence_unknown": 0.05, "confirm": 0.05, "cancel": 0.90}, now=10.0)
        self.assertTrue(tr_cancel.returned_idle)
        self.assertEqual(fsm.state, FSMState.IDLE_MONITORING)

        # Re-enter VOICE_VERIFY, then test 10s silence fail-safe timeout
        fsm.step({"background": 0.0, "unknown": 0.0, "help": 0.95}, now=12.0)
        fsm.step({"background": 0.0, "unknown": 0.0, "help": 0.95}, now=12.2)
        self.assertEqual(fsm.state, FSMState.VOICE_VERIFY)
        tr_timeout = fsm.step({"silence_unknown": 0.95, "confirm": 0.03, "cancel": 0.02}, now=22.3)
        self.assertTrue(tr_timeout.entered_alert)
        self.assertEqual(fsm.state, FSMState.ALERT_DISPATCHED)
        self.assertIn("Timeout", fsm.alert_reason)

        # Physical Cancel Button (D3) resets from ALERT_DISPATCHED
        tr_d3 = fsm.press_cancel_button(now=23.0)
        self.assertTrue(tr_d3.returned_idle)
        self.assertEqual(fsm.state, FSMState.IDLE_MONITORING)


class TestInt8TFLiteAndAudio(unittest.TestCase):
    def test_exported_int8_models_run_via_interpreter(self) -> None:
        s1_path = REPO_ROOT / "models" / "stage1_wake_int8.tflite"
        s2_path = REPO_ROOT / "models" / "stage2_verify_int8.tflite"
        self.assertTrue(s1_path.exists(), f"Missing {s1_path}")
        self.assertTrue(s2_path.exists(), f"Missing {s2_path}")
        self.assertLess(s1_path.stat().st_size, 30 * 1024)
        self.assertLess(s2_path.stat().st_size, 30 * 1024)

        m1 = Int8TFLiteModel(s1_path, STAGE1_LABELS)
        m2 = Int8TFLiteModel(s2_path, STAGE2_LABELS)
        dummy_spec = np.zeros((NUM_FRAMES, NUM_MEL_BINS), dtype=np.float32)

        out1 = m1.predict(dummy_spec)
        out2 = m2.predict(dummy_spec)
        self.assertEqual(out1.input_dtype, "int8")
        self.assertEqual(out1.output_dtype, "int8")
        self.assertEqual(out2.input_dtype, "int8")
        self.assertAlmostEqual(sum(out1.probs.values()), 1.0, places=3)
        self.assertAlmostEqual(sum(out2.probs.values()), 1.0, places=3)

        mgr = KWSModelManager()
        self.assertTrue(mgr.real_models_ready)

    def test_mock_model_fallback(self) -> None:
        mock = MockKWSModel(STAGE1_LABELS)
        dummy_spec = np.zeros((NUM_FRAMES, NUM_MEL_BINS), dtype=np.float32)
        idle_out = mock.predict(dummy_spec, rms=0.01)
        self.assertTrue(idle_out.is_mock)
        self.assertLess(idle_out.probs["help"], 0.2)

        mock.trigger("help", repeat_ticks=1)
        trig_out = mock.predict(dummy_spec, rms=0.01)
        self.assertGreater(trig_out.probs["help"], 0.85)

    def test_verify_prompt_wav_generation(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            wav_path = Path(td) / "verify_prompt.wav"
            out = ensure_verify_prompt_wav(wav_path)
            self.assertTrue(out.exists())
            audio = load_wav_16k_mono(out, target_len=None)
            self.assertGreater(audio.size, 8000)


class TestDataToolsAndFirmware(unittest.TestCase):
    def test_record_voice_cli_help(self) -> None:
        proc = subprocess.run(
            [sys.executable, str(REPO_ROOT / "data_tools" / "record_voice.py"), "--help"],
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertIn("--label", proc.stdout)

    def test_prepare_dataset_dry_run(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            custom_dir = root / "custom"
            out_dir = root / "processed"
            rng = np.random.default_rng(7)
            for lbl in ("help", "confirm", "cancel", "unknown", "background"):
                ldir = custom_dir / lbl
                ldir.mkdir(parents=True, exist_ok=True)
                for idx in range(6):
                    sig = 0.1 * rng.standard_normal(CLIP_SAMPLES).astype(np.float32)
                    save_wav_16k_mono(ldir / f"spk01_{lbl}_{idx:02d}.wav", sig)

            manifest = prepare_datasets(
                custom_dir=custom_dir,
                gsc_dir=root / "no_gsc",
                musan_dir=root / "no_musan",
                out_dir=out_dir,
                augments_per_target=1,
                augments_per_neg=0,
            )
            self.assertFalse(manifest["synthetic_tts_used"])
            self.assertTrue((out_dir / "stage1_features.npz").exists())
            self.assertTrue((out_dir / "stage2_features.npz").exists())

    def test_firmware_files_and_headers_consistent(self) -> None:
        fw = REPO_ROOT / "firmware_esp32s3"
        required = [
            fw / "platformio.ini",
            fw / "src" / "main.cpp",
            fw / "include" / "config.h",
            fw / "include" / "dsp_tables.h",
            fw / "include" / "dsp_mel.h",
            fw / "include" / "fsm.h",
            fw / "include" / "kws_engine.h",
            fw / "include" / "audio_prompt.h",
            fw / "include" / "caregiver_ap.h",
            fw / "include" / "stage1_wake_model.h",
            fw / "include" / "stage2_verify_model.h",
        ]
        for path in required:
            self.assertTrue(path.exists(), f"Missing firmware file: {path}")

        s1_h = (fw / "include" / "stage1_wake_model.h").read_text(encoding="utf-8")
        s2_h = (fw / "include" / "stage2_verify_model.h").read_text(encoding="utf-8")
        self.assertIn("g_stage1_wake_model_data", s1_h)
        self.assertIn("g_stage2_verify_model_data", s2_h)


if __name__ == "__main__":
    unittest.main(verbosity=2)
