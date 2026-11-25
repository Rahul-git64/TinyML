# Offline Two-Stage Emergency Keyword Spotting (KWS) & Voice-Triggered Alert System

A 100% offline, privacy-first TinyML emergency alert system targeting the **ESP32-S3** (**Seeed Studio XIAO ESP32S3 Sense** or **ESP32-S3 + INMP441 I2S Mic + MAX98357A Speaker**), paired with a **localhost Streamlit Digital Twin** so you can live-demo the exact quantized **INT8 `.tflite`** models on your laptop microphone and speakers before flashing hardware.

---

## Architecture Overview

1. **Shared DSP Contract (`ml_pipeline/dsp.py` ↔ `firmware_esp32s3/include/dsp_mel.h`)**
   - `16 kHz` mono PCM → `480`-sample (30 ms) Hann window, `320`-sample (20 ms) stride, `512`-point RFFT power spectrum → `40`-bin HTK Mel filterbank (`80–7600 Hz`) → log & fixed scaling `clip((ln(E + 1e-6) + 6.0) / 6.0, -3.0, 3.0)` → `(49, 40)` features per 1.0 s clip.
   - `ml_pipeline/dsp.py` exports `firmware_esp32s3/include/dsp_tables.h` so Python and C++ (`ESP-DSP`) share bit-exact window & sparse Mel filterbank tables.

2. **Two-Stage INT8 DS-CNN Models (< 30 KB each)**
   - **Stage 1 Wake-Word (`stage1_wake_int8.tflite`)**: `["background", "unknown", "help"]`
   - **Stage 2 Verification (`stage2_verify_int8.tflite`)**: `["silence_unknown", "confirm", "cancel"]`
   - Built strictly with ESP-NN accelerated ops (`Conv2D`, `DepthwiseConv2D`, `BatchNormalization`, `ReLU`, `AveragePooling2D`, `Dense`, `Softmax`) and full INT8 I/O quantization.

3. **Two-Stage Finite State Machine (`virtual_demo/fsm.py` ↔ `firmware_esp32s3/include/fsm.h`)**
   - **`IDLE_MONITORING` (Green LED)**: Runs Stage 1 INT8 model on rolling 1 s windows. Requires consecutive `"help"` detections above threshold.
   - **`VOICE_VERIFY` (Yellow LED)**: Plays local audio prompt (*"Emergency triggered. Say Confirm or Cancel."*), starts a 10 s countdown, and swaps to the Stage 2 INT8 model.
     - Say **`"cancel"`** or press **Physical Cancel Button (D3)** → returns to `IDLE_MONITORING`.
     - Say **`"confirm"`** OR let the **10 s countdown expire in silence (fail-safe)** → transitions to `ALERT_DISPATCHED`.
   - **`ALERT_DISPATCHED` (Red LED)**: Displays the Caregiver Alert panel + Patient Medical ID card (and pushes JSON alerts over the ESP32-S3 offline Wi-Fi SoftAP `192.168.4.1` WebSocket).

---

## Quickstart

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Verify DSP & Export C++ Tables
```bash
python ml_pipeline/dsp.py
```

### 3. Launch the Localhost Digital Twin Immediately (Mock Fallback or Real INT8)
```bash
streamlit run virtual_demo/app.py
```
- Enable **Mock-Model Fallback** in the sidebar to test the full FSM, microphone spectrogram, speaker prompt, countdown, D3 cancel button, and Caregiver Alert card immediately before training.
- Or generate pipeline-verified INT8 `.tflite` models right away with `--smoke-test` (for pipeline/UI verification before recording your real voice dataset):
```bash
python ml_pipeline/train_stage1_wake.py --smoke-test --epochs 4
python ml_pipeline/train_stage2_verify.py --smoke-test --epochs 4
python ml_pipeline/evaluate.py --smoke-test
```

### 4. Record Real Human Voice Clips & Prepare Real Dataset (No Synthetic TTS)
Record 1-second 16 kHz mono WAV clips from your microphone (`help`, `confirm`, `cancel`, `unknown`, `background`):
```bash
python data_tools/record_voice.py --label help --count 25 --speaker speaker01
python data_tools/record_voice.py --label confirm --count 25 --speaker speaker01
python data_tools/record_voice.py --label cancel --count 25 --speaker speaker01
python data_tools/record_voice.py --label unknown --count 25 --speaker speaker01
python data_tools/record_voice.py --label background --count 15 --speaker room01
```
Combine **Google Speech Commands v2** + **MUSAN** + your real human recordings:
```bash
python data_tools/prepare_dataset.py --download-gsc --download-musan
```
Train & evaluate the real INT8 models:
```bash
python ml_pipeline/train_stage1_wake.py
python ml_pipeline/train_stage2_verify.py
python ml_pipeline/evaluate.py
```

### 5. Flash ESP32-S3 Firmware (PlatformIO)
```bash
cd firmware_esp32s3
# For Seeed Studio XIAO ESP32S3 Sense (onboard PDM mic + MAX98357A speaker):
pio run -e xiao_esp32s3_sense -t upload

# Or for generic ESP32-S3 + INMP441 I2S mic + MAX98357A speaker:
pio run -e esp32s3_inmp441 -t upload
```
Connect to the ESP32-S3 offline Wi-Fi SoftAP (`EmergencyKWS-AP`, password `caregiver123`) and open `http://192.168.4.1` for the live Caregiver Dashboard.
