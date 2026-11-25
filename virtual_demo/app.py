"""ESP32-S3 Virtual Hardware Simulator ("Digital Twin") on localhost.

Runs the exact exported INT8 `.tflite` models via `tf.lite.Interpreter`
on live laptop microphone audio (with an instant Mock-Model Fallback toggle),
executes the Two-Stage Emergency FSM, plays local speaker voice prompts,
and renders simulated ESP32-S3 telemetry + Caregiver Alert Patient ID Card.
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml_pipeline.dsp import FEAT_CLIP_MAX, FEAT_CLIP_MIN, load_wav_16k_mono  # noqa: E402
from virtual_demo.audio_engine import (  # noqa: E402
    DEFAULT_PROMPT_WAV,
    AudioCaptureEngine,
    ensure_verify_prompt_wav,
    list_input_devices,
    play_verify_prompt_async,
)
from virtual_demo.fsm import FSMConfig, FSMState, LEDColor, TwoStageFSM  # noqa: E402
from virtual_demo.tflite_runner import (  # noqa: E402
    STAGE1_LABELS,
    STAGE2_LABELS,
    KWSModelManager,
)

PATIENT_PROFILE_PATH = Path(__file__).resolve().parent / "patient_profile.json"

DEFAULT_PATIENT_PROFILE: Dict[str, str] = {
    "patient_name": "Eleanor Vance",
    "age": "74",
    "blood_group": "O+",
    "conditions": "Atrial Fibrillation, Mild Asthma, Fall Risk",
    "allergies": "Penicillin, Sulfa drugs",
    "medications": "Apixaban 5mg BID, Albuterol PRN",
    "emergency_contact": "Marcus Vance (Son) — +1-555-0194",
    "address": "42 Maplewood Terrace, Apt 3B",
}


def load_patient_profile() -> Dict[str, str]:
    if PATIENT_PROFILE_PATH.exists():
        try:
            data = json.loads(PATIENT_PROFILE_PATH.read_text(encoding="utf-8"))
            merged = dict(DEFAULT_PATIENT_PROFILE)
            merged.update({k: str(v) for k, v in data.items()})
            return merged
        except Exception:
            pass
    PATIENT_PROFILE_PATH.write_text(json.dumps(DEFAULT_PATIENT_PROFILE, indent=2), encoding="utf-8")
    return dict(DEFAULT_PATIENT_PROFILE)


def save_patient_profile(profile: Dict[str, str]) -> None:
    PATIENT_PROFILE_PATH.write_text(json.dumps(profile, indent=2), encoding="utf-8")


def _inject_custom_css() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #0b0f17;
            color: #e6edf3;
        }
        .hw-card {
            background: #131a26;
            border: 1px solid #263248;
            border-radius: 12px;
            padding: 14px 18px;
            margin-bottom: 12px;
        }
        .led-wrap {
            display: flex;
            align-items: center;
            gap: 12px;
        }
        .led-dot {
            width: 22px;
            height: 22px;
            border-radius: 50%;
            display: inline-block;
        }
        .led-GREEN { background: #22c55e; box-shadow: 0 0 16px #22c55e; }
        .led-YELLOW { background: #facc15; box-shadow: 0 0 18px #facc15; }
        .led-RED { background: #ef4444; box-shadow: 0 0 22px #ef4444; }
        .state-pill {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 999px;
            font-weight: 700;
            font-size: 0.9rem;
            letter-spacing: 0.04em;
        }
        .pill-IDLE_MONITORING { background: #14532d; color: #86efac; border: 1px solid #22c55e; }
        .pill-VOICE_VERIFY { background: #713f12; color: #fde047; border: 1px solid #facc15; }
        .pill-ALERT_DISPATCHED { background: #7f1d1d; color: #fca5a5; border: 1px solid #ef4444; }
        .alert-banner {
            background: linear-gradient(135deg, #450a0a 0%, #7f1d1d 100%);
            border: 2px solid #ef4444;
            border-radius: 12px;
            padding: 18px 22px;
            margin-bottom: 14px;
        }
        .verify-banner {
            background: linear-gradient(135deg, #422006 0%, #713f12 100%);
            border: 2px solid #facc15;
            border-radius: 12px;
            padding: 16px 20px;
            margin-bottom: 14px;
        }
        .medical-card {
            background: #101722;
            border: 1px solid #334155;
            border-radius: 10px;
            padding: 14px 16px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _init_session_state() -> None:
    if "fsm" not in st.session_state:
        st.session_state.fsm = TwoStageFSM(FSMConfig())
    if "audio_engine" not in st.session_state:
        st.session_state.audio_engine = AudioCaptureEngine()
    if "model_manager" not in st.session_state:
        st.session_state.model_manager = KWSModelManager()
    if "patient_profile" not in st.session_state:
        st.session_state.patient_profile = load_patient_profile()
    if "use_mock" not in st.session_state:
        # Default to Real INT8 when models exist, otherwise Mock Fallback automatically
        st.session_state.use_mock = not st.session_state.model_manager.real_models_ready
    if "prompt_wav_path" not in st.session_state:
        st.session_state.prompt_wav_path = ensure_verify_prompt_wav(DEFAULT_PROMPT_WAV)
    if "last_inference" not in st.session_state:
        st.session_state.last_inference = None
    if "ws_alert_payload" not in st.session_state:
        st.session_state.ws_alert_payload = None


def _render_spectrogram_figure(spec: np.ndarray):
    fig, ax = plt.subplots(figsize=(5.4, 2.7), dpi=110)
    fig.patch.set_facecolor("#131a26")
    ax.set_facecolor("#0b0f17")
    ax.imshow(
        spec.T,
        origin="lower",
        aspect="auto",
        cmap="magma",
        vmin=FEAT_CLIP_MIN,
        vmax=FEAT_CLIP_MAX,
    )
    ax.set_title("Live 40-Bin Log-Mel Spectrogram (49 x 40)", color="#e6edf3", fontsize=9.5)
    ax.set_xlabel("Frame (20 ms hop)", color="#94a3b8", fontsize=8)
    ax.set_ylabel("Mel Bin", color="#94a3b8", fontsize=8)
    ax.tick_params(colors="#94a3b8", labelsize=7.5)
    plt.tight_layout(pad=0.6)
    return fig


def _render_patient_medical_card(profile: Dict[str, str]) -> None:
    st.markdown(
        f"""
        <div class="medical-card">
            <div style="font-size:1.05rem; font-weight:700; color:#f8fafc; margin-bottom:8px;">
                Patient Medical ID Card
            </div>
            <div><b>Name:</b> {profile['patient_name']} &nbsp;|&nbsp; <b>Age:</b> {profile['age']} &nbsp;|&nbsp; <b>Blood:</b> {profile['blood_group']}</div>
            <div><b>Conditions:</b> {profile['conditions']}</div>
            <div><b>Allergies:</b> {profile['allergies']}</div>
            <div><b>Medications:</b> {profile['medications']}</div>
            <div><b>Emergency Contact:</b> {profile['emergency_contact']}</div>
            <div><b>Location:</b> {profile['address']}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _handle_fsm_side_effects(transition, fsm: TwoStageFSM, profile: Dict[str, str]) -> None:
    if transition.entered_voice_verify:
        play_verify_prompt_async(st.session_state.prompt_wav_path)
    if transition.entered_alert:
        st.session_state.ws_alert_payload = {
            "event": "ALERT_DISPATCHED",
            "node_ip": "192.168.4.1",
            "ws_endpoint": "ws://192.168.4.1/ws",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "reason": fsm.alert_reason,
            "confidence": round(fsm.alert_confidence, 3),
            "patient": profile,
        }
    if transition.returned_idle:
        st.session_state.ws_alert_payload = None


def main() -> None:
    st.set_page_config(
        page_title="ESP32-S3 Two-Stage Emergency KWS Digital Twin",
        page_icon="🩺",
        layout="wide",
    )
    _inject_custom_css()
    _init_session_state()

    fsm: TwoStageFSM = st.session_state.fsm
    audio: AudioCaptureEngine = st.session_state.audio_engine
    models: KWSModelManager = st.session_state.model_manager

    # ---------------- Sidebar: Hardware & FSM Controls ----------------
    with st.sidebar:
        st.header("ESP32-S3 Twin Controls")

        mode_options = ["Real INT8 .tflite Models", "Mock-Model Fallback"]
        default_idx = 1 if st.session_state.use_mock else 0
        selected_mode = st.radio("Inference Engine", mode_options, index=default_idx)
        want_mock = selected_mode == "Mock-Model Fallback"

        if not want_mock and not models.real_models_ready:
            st.warning(
                "Real INT8 `.tflite` models not found yet. Falling back to Mock Mode until "
                "you run `train_stage1_wake.py` and `train_stage2_verify.py`."
            )
            st.session_state.use_mock = True
        else:
            st.session_state.use_mock = want_mock

        if models.real_models_ready and not models.trained_on_real_data:
            st.info(
                "INT8 `.tflite` models loaded (`--smoke-test` weights). "
                "Record real voice clips via `data_tools/record_voice.py` for speech accuracy."
            )

        if st.button("Reload .tflite Models", use_container_width=True):
            models.reload()
            st.rerun()

        st.divider()
        st.subheader("Microphone Input (16 kHz Mono)")
        devices = list_input_devices()
        dev_labels = ["Default System Mic"] + [f"[{d['index']}] {d['name']}" for d in devices]
        dev_choice = st.selectbox("Input Device", dev_labels)
        selected_dev_idx = None
        if dev_choice != "Default System Mic":
            selected_dev_idx = int(dev_choice.split("]")[0].lstrip("["))

        c1, c2 = st.columns(2)
        if c1.button("Start Mic", type="primary", use_container_width=True, disabled=audio.is_running):
            if not audio.start(selected_dev_idx):
                st.error(f"Mic start failed: {audio.last_error}")
            st.rerun()
        if c2.button("Stop Mic", use_container_width=True, disabled=not audio.is_running):
            audio.stop()
            st.rerun()

        if audio.last_error:
            st.error(f"Audio error: {audio.last_error}")

        uploaded_wav = st.file_uploader("Test 1s WAV Clip via DSP + Model", type=["wav"])
        if uploaded_wav is not None and st.button("Inject Uploaded WAV", use_container_width=True):
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(uploaded_wav.read())
                tmp_path = Path(tmp.name)
            try:
                wav_audio = load_wav_16k_mono(tmp_path)
                audio.push_external_audio(wav_audio)
                st.success("Injected WAV into rolling 1.0s buffer.")
            finally:
                tmp_path.unlink(missing_ok=True)

        st.divider()
        st.subheader("FSM Thresholds")
        fsm.config.stage1_threshold = st.slider(
            "Stage 1 'help' Threshold", 0.30, 0.98, float(fsm.config.stage1_threshold), 0.02
        )
        fsm.config.stage1_consecutive_hits = st.slider(
            "Stage 1 Consecutive Hits", 1, 5, int(fsm.config.stage1_consecutive_hits), 1
        )
        fsm.config.stage2_confirm_threshold = st.slider(
            "Stage 2 'confirm' Threshold", 0.30, 0.98, float(fsm.config.stage2_confirm_threshold), 0.02
        )
        fsm.config.stage2_cancel_threshold = st.slider(
            "Stage 2 'cancel' Threshold", 0.30, 0.98, float(fsm.config.stage2_cancel_threshold), 0.02
        )
        fsm.config.verify_timeout_sec = st.slider(
            "Verify Countdown (sec)", 3.0, 20.0, float(fsm.config.verify_timeout_sec), 1.0
        )
        fsm.config.prompt_grace_sec = st.slider(
            "Prompt Acoustic Grace (sec)", 0.0, 4.0, float(fsm.config.prompt_grace_sec), 0.2
        )

        st.divider()
        with st.expander("Patient Medical ID Profile"):
            p = dict(st.session_state.patient_profile)
            p["patient_name"] = st.text_input("Patient Name", p["patient_name"])
            p["age"] = st.text_input("Age", p["age"])
            p["blood_group"] = st.text_input("Blood Group", p["blood_group"])
            p["conditions"] = st.text_area("Conditions", p["conditions"], height=68)
            p["allergies"] = st.text_input("Allergies", p["allergies"])
            p["medications"] = st.text_input("Medications", p["medications"])
            p["emergency_contact"] = st.text_input("Emergency Contact", p["emergency_contact"])
            p["address"] = st.text_input("Address", p["address"])
            if st.button("Save Patient Profile", use_container_width=True):
                st.session_state.patient_profile = p
                save_patient_profile(p)
                st.success("Saved locally.")

    # ---------------- Header ----------------
    st.title("ESP32-S3 Offline Two-Stage Emergency KWS — Digital Twin")
    st.caption(
        "100% Localhost Simulation • Exact 16 kHz → 40-Bin Log-Mel DSP • "
        "Actual INT8 `.tflite` Execution (`tf.lite.Interpreter`) • Zero Cloud / Zero Login"
    )

    # ---------------- Live Periodic Hardware Fragment ----------------
    @st.fragment(run_every=0.25)
    def live_digital_twin_cockpit() -> None:
        now = time.monotonic()
        _, spec, rms = audio.get_window_and_spec()

        stage = fsm.active_stage
        inf = models.predict(stage=stage, spec_49x40=spec, use_mock=st.session_state.use_mock, rms=rms)
        st.session_state.last_inference = inf

        transition = fsm.step(inf.probs, now=now)
        _handle_fsm_side_effects(transition, fsm, st.session_state.patient_profile)

        # Top Hardware Status Bar
        led = fsm.led_color.value
        col_state, col_led, col_stage, col_btn = st.columns([2.2, 1.3, 1.5, 1.8])
        with col_state:
            st.markdown(
                f"""
                <div class="hw-card">
                    <div style="font-size:0.75rem; color:#94a3b8;">ACTIVE FSM STATE</div>
                    <div style="margin-top:6px;">
                        <span class="state-pill pill-{fsm.state.value}">{fsm.state.value}</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_led:
            st.markdown(
                f"""
                <div class="hw-card">
                    <div style="font-size:0.75rem; color:#94a3b8;">STATUS LED (RGB)</div>
                    <div class="led-wrap" style="margin-top:6px;">
                        <span class="led-dot led-{led}"></span>
                        <b>{led}</b>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_stage:
            stage_name = "Stage 1: Wake ('help')" if fsm.active_stage == 1 else "Stage 2: Verify"
            mode_tag = "MOCK" if inf.is_mock else "INT8 TFLITE"
            st.markdown(
                f"""
                <div class="hw-card">
                    <div style="font-size:0.75rem; color:#94a3b8;">ACTIVE MODEL ({mode_tag})</div>
                    <div style="margin-top:6px; font-weight:700;">{stage_name}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_btn:
            st.markdown("<div style='height:6px;'></div>", unsafe_allow_html=True)
            if st.button(
                "Physical Cancel Button (D3)",
                type="secondary",
                use_container_width=True,
                key="d3_cancel_btn",
            ):
                tr = fsm.press_cancel_button(time.monotonic())
                _handle_fsm_side_effects(tr, fsm, st.session_state.patient_profile)
                st.rerun(scope="fragment")

        # Voice Verification Countdown Banner
        if fsm.state == FSMState.VOICE_VERIFY:
            rem = fsm.remaining_verify_seconds(now)
            frac = min(1.0, max(0.0, rem / max(0.1, fsm.config.verify_timeout_sec)))
            grace_note = (
                "Speaker Prompt Playing (acoustic grace window)..."
                if fsm.in_prompt_grace_period(now)
                else "Listening for 'CONFIRM' or 'CANCEL' (silence triggers alert at 0s)"
            )
            st.markdown(
                f"""
                <div class="verify-banner">
                    <div style="font-size:1.1rem; font-weight:700; color:#fde047;">
                        VOICE VERIFICATION ACTIVE — {rem:0.1f}s remaining
                    </div>
                    <div style="color:#fef08a; margin-top:4px;">{grace_note}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.progress(frac)

        # Caregiver Alert Banner & Patient Medical ID Card
        if fsm.state == FSMState.ALERT_DISPATCHED:
            st.markdown(
                f"""
                <div class="alert-banner">
                    <div style="font-size:1.25rem; font-weight:800; color:#fecaca;">
                        EMERGENCY ALERT DISPATCHED
                    </div>
                    <div style="color:#fee2e2; margin-top:4px;">
                        Trigger Reason: <b>{fsm.alert_reason}</b> &nbsp;|&nbsp;
                        Offline SoftAP Broadcast: <code>ws://192.168.4.1/ws</code>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            al_left, al_right = st.columns([1.4, 1.0])
            with al_left:
                _render_patient_medical_card(st.session_state.patient_profile)
                if st.button("Acknowledge & Reset Alert", type="primary", use_container_width=True):
                    tr = fsm.acknowledge_alert(time.monotonic())
                    _handle_fsm_side_effects(tr, fsm, st.session_state.patient_profile)
                    st.rerun(scope="fragment")
            with al_right:
                st.caption("Simulated ESP32-S3 SoftAP (`192.168.4.1`) WebSocket Alert JSON")
                st.json(st.session_state.ws_alert_payload or {})

        # Mock Trigger Controls (when Mock Mode is enabled)
        if st.session_state.use_mock:
            with st.container(border=True):
                st.markdown("**Mock-Model Trigger Panel** (instant FSM path verification without trained speech weights)")
                mc1, mc2, mc3, mc4, mc5 = st.columns(5)
                if mc1.button("Simulate 'HELP' (Stage 1)", use_container_width=True):
                    models.mock_stage1.trigger("help", repeat_ticks=max(2, fsm.config.stage1_consecutive_hits))
                if mc2.button("Simulate 'CONFIRM' (Stage 2)", use_container_width=True):
                    models.mock_stage2.trigger("confirm", repeat_ticks=1)
                if mc3.button("Simulate 'CANCEL' (Stage 2)", use_container_width=True):
                    models.mock_stage2.trigger("cancel", repeat_ticks=1)
                if mc4.button("Replay Prompt WAV", use_container_width=True):
                    play_verify_prompt_async(st.session_state.prompt_wav_path)
                if mc5.button("Fast-Forward Silence Timeout", use_container_width=True):
                    if fsm.state == FSMState.VOICE_VERIFY:
                        fsm.verify_started_at = now - fsm.config.verify_timeout_sec - 0.1

        # Main Live Telemetry Columns
        col_spec, col_probs = st.columns([1.25, 1.0])
        with col_spec:
            fig = _render_spectrogram_figure(spec)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
            mic_status = "LIVE MIC STREAMING" if audio.is_running else "MIC STOPPED (Click 'Start Mic' in sidebar)"
            st.caption(f"{mic_status} • Window RMS: `{rms:.4f}` • Samples captured: `{audio.total_samples:,}`")
            st.progress(min(1.0, rms * 5.0))

        with col_probs:
            st.subheader("INT8 Confidence Output")
            labels = STAGE1_LABELS if fsm.active_stage == 1 else STAGE2_LABELS
            for lbl in labels:
                p_val = float(inf.probs.get(lbl, 0.0))
                st.write(f"**{lbl}** — `{p_val * 100:.1f}%`")
                st.progress(min(1.0, max(0.0, p_val)))

            if fsm.state == FSMState.IDLE_MONITORING:
                st.caption(
                    f"Wake debounce hits: `{fsm.consecutive_help_hits} / {fsm.config.stage1_consecutive_hits}` "
                    f"(threshold `{fsm.config.stage1_threshold:.2f}`)"
                )

            st.divider()
            st.subheader("Simulated ESP32-S3 Hardware Telemetry")
            s1_kb = models.stage1_tflite.flash_kb if models.stage1_tflite else inf.flash_kb
            s2_kb = models.stage2_tflite.flash_kb if models.stage2_tflite else inf.flash_kb
            m1, m2, m3 = st.columns(3)
            m1.metric("Stage 1 Flash", f"{s1_kb:.1f} KB")
            m2.metric("Stage 2 Flash", f"{s2_kb:.1f} KB")
            m3.metric("SRAM Tensor Arena", f"{inf.arena_kb:.1f} KB")

            m4, m5, m6 = st.columns(3)
            m4.metric("Inference Latency", f"{inf.latency_ms:.2f} ms")
            m5.metric("Tensor Dtype", f"{inf.input_dtype} / {inf.output_dtype}")
            m6.metric("Dual-Core Load", "C0: DSP | C1: TFLM")

        # Bottom Row: Patient Card Preview + Event Log
        b_left, b_right = st.columns([1.0, 1.3])
        with b_left:
            if fsm.state != FSMState.ALERT_DISPATCHED:
                _render_patient_medical_card(st.session_state.patient_profile)
        with b_right:
            st.subheader("FSM Transition Log")
            if fsm.events:
                rows = [
                    {
                        "Time (s)": f"{e.timestamp:.1f}",
                        "From": e.from_state.value,
                        "To": e.to_state.value,
                        "Reason": e.reason,
                    }
                    for e in reversed(fsm.events[-10:])
                ]
                st.dataframe(rows, use_container_width=True, hide_index=True)
            else:
                st.info("No state transitions yet. System is monitoring in IDLE_MONITORING (Green LED).")

    live_digital_twin_cockpit()


if __name__ == "__main__":
    main()
