#include <Arduino.h>
#include <driver/i2s.h>
#include <freertos/FreeRTOS.h>
#include <freertos/task.h>

#include "audio_prompt.h"
#include "caregiver_ap.h"
#include "config.h"
#include "dsp_mel.h"
#include "dsp_tables.h"
#include "fsm.h"
#include "kws_engine.h"

namespace {

kws_dsp::StreamingLogMelDSP g_dsp;
kws_ml::DualStageTflmEngine g_kws;
kws_fsm::TwoStageFSM g_fsm;
kws_audio::SpeakerPrompt g_speaker;
kws_net::CaregiverSoftAP g_ap;

portMUX_TYPE g_spec_mux = portMUX_INITIALIZER_UNLOCKED;
float g_shared_spec[kws_dsp::kNumFrames * kws_dsp::kNumMelBins] = {0};
TaskHandle_t g_inference_task_handle = nullptr;

bool init_mic_i2s() {
#if USE_XIAO_PDM_MIC
  i2s_config_t cfg = {
      .mode = static_cast<i2s_mode_t>(I2S_MODE_MASTER | I2S_MODE_RX | I2S_MODE_PDM),
      .sample_rate = kws_dsp::kSampleRate,
      .bits_per_sample = I2S_BITS_PER_SAMPLE_16BIT,
      .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
      .communication_format = I2S_COMM_FORMAT_STAND_I2S,
      .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
      .dma_buf_count = 8,
      .dma_buf_len = 320,
      .use_apll = false,
      .tx_desc_auto_clear = false,
      .fixed_mclk = 0,
  };
  i2s_pin_config_t pins = {
      .bck_io_num = I2S_PIN_NO_CHANGE,
      .ws_io_num = kPdmMicClkPin,
      .data_out_num = I2S_PIN_NO_CHANGE,
      .data_in_num = kPdmMicDataPin,
  };
#else
  i2s_config_t cfg = {
      .mode = static_cast<i2s_mode_t>(I2S_MODE_MASTER | I2S_MODE_RX),
      .sample_rate = kws_dsp::kSampleRate,
      .bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT,
      .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
      .communication_format = I2S_COMM_FORMAT_STAND_I2S,
      .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
      .dma_buf_count = 8,
      .dma_buf_len = 320,
      .use_apll = false,
      .tx_desc_auto_clear = false,
      .fixed_mclk = 0,
  };
  i2s_pin_config_t pins = {
      .bck_io_num = kInmp441SckPin,
      .ws_io_num = kInmp441WsPin,
      .data_out_num = I2S_PIN_NO_CHANGE,
      .data_in_num = kInmp441SdPin,
  };
#endif
  if (i2s_driver_install(I2S_NUM_0, &cfg, 0, nullptr) != ESP_OK) return false;
  if (i2s_set_pin(I2S_NUM_0, &pins) != ESP_OK) return false;
  return true;
}

void set_status_leds(kws_fsm::LEDColor color) {
  digitalWrite(kLedGreenPin, color == kws_fsm::LEDColor::GREEN ? HIGH : LOW);
  digitalWrite(kLedYellowPin, color == kws_fsm::LEDColor::YELLOW ? HIGH : LOW);
  digitalWrite(kLedRedPin, color == kws_fsm::LEDColor::RED ? HIGH : LOW);
}

// Core 0: I2S/PDM DMA read + 40-bin Log-Mel DSP
void audio_dsp_task(void* param) {
  (void)param;
#if USE_XIAO_PDM_MIC
  int16_t raw_hop[kws_dsp::kHopLen];
#else
  int32_t raw_hop[kws_dsp::kHopLen];
#endif
  float float_hop[kws_dsp::kHopLen];
  int hops_since_notify = 0;

  for (;;) {
    size_t bytes_read = 0;
    const esp_err_t err =
        i2s_read(I2S_NUM_0, raw_hop, sizeof(raw_hop), &bytes_read, portMAX_DELAY);
    if (err != ESP_OK || bytes_read != sizeof(raw_hop)) {
      vTaskDelay(pdMS_TO_TICKS(5));
      continue;
    }

    for (int i = 0; i < kws_dsp::kHopLen; ++i) {
#if USE_XIAO_PDM_MIC
      float_hop[i] = static_cast<float>(raw_hop[i]) / 32768.0f;
#else
      // INMP441 delivers 24-bit left-justified in 32-bit word
      const int32_t s24 = raw_hop[i] >> 8;
      float_hop[i] = static_cast<float>(s24) / 8388608.0f;
#endif
    }

    if (g_dsp.push_hop(float_hop)) {
      hops_since_notify++;
      if (hops_since_notify >= kInferenceIntervalHops &&
          g_dsp.frames_computed() >= kws_dsp::kNumFrames) {
        hops_since_notify = 0;
        portENTER_CRITICAL(&g_spec_mux);
        g_dsp.copy_spectrogram(g_shared_spec);
        portEXIT_CRITICAL(&g_spec_mux);
        if (g_inference_task_handle != nullptr) {
          xTaskNotifyGive(g_inference_task_handle);
        }
      }
    }
  }
}

// Core 1: tflite-micro INT8 inference + Two-Stage FSM + WebSocket alert push
void inference_fsm_task(void* param) {
  (void)param;
  float local_spec[kws_dsp::kNumFrames * kws_dsp::kNumMelBins];
  float probs[3] = {1.0f, 0.0f, 0.0f};
  bool last_btn_state = HIGH;
  uint32_t last_btn_change_ms = 0;

  for (;;) {
    // Wait up to 250 ms so countdown & D3 cancel remain responsive
    ulTaskNotifyTake(pdTRUE, pdMS_TO_TICKS(250));
    const uint32_t now_ms = millis();

    // Debounced Physical Cancel Button (D3)
    const bool btn = digitalRead(kCancelButtonPin);
    if (btn != last_btn_state && (now_ms - last_btn_change_ms) >= kButtonDebounceMs) {
      last_btn_change_ms = now_ms;
      last_btn_state = btn;
      if (btn == LOW) {
        auto tr = g_fsm.press_cancel_button(now_ms);
        if (tr.returned_idle) {
          Serial.printf("[FSM] %s -> IDLE_MONITORING\n", tr.reason);
          g_kws.load_stage(1);
        }
      }
    }

    if (g_ap.consume_ack_request()) {
      auto tr = g_fsm.acknowledge_alert(now_ms);
      if (tr.returned_idle) {
        Serial.println("[FSM] Caregiver WebSocket ACK -> IDLE_MONITORING");
        g_kws.load_stage(1);
      }
    }

    // Ensure correct stage INT8 model is loaded in shared PSRAM tensor_arena
    g_kws.load_stage(g_fsm.active_stage());

    portENTER_CRITICAL(&g_spec_mux);
    memcpy(local_spec, g_shared_spec, sizeof(local_spec));
    portEXIT_CRITICAL(&g_spec_mux);

    if (g_kws.predict(local_spec, probs)) {
      const auto tr = g_fsm.step(probs, now_ms);
      if (tr.entered_voice_verify) {
        Serial.println("[FSM] -> VOICE_VERIFY (playing MAX98357A prompt)");
        g_kws.load_stage(2);
        g_speaker.play_verify_prompt();
      } else if (tr.entered_alert) {
        Serial.printf("[FSM] -> ALERT_DISPATCHED (%s)\n", tr.reason);
      } else if (tr.returned_idle) {
        Serial.printf("[FSM] -> IDLE_MONITORING (%s)\n", tr.reason);
        g_kws.load_stage(1);
      }
    }

    set_status_leds(g_fsm.led_color());
    g_ap.broadcast_status(g_fsm, probs, now_ms, g_kws.arena_used_bytes());
  }
}

}  // namespace

void setup() {
  Serial.begin(115200);
  delay(400);
  Serial.println("\n=== ESP32-S3 Offline Two-Stage Emergency KWS ===");

  pinMode(kCancelButtonPin, INPUT_PULLUP);
  pinMode(kLedGreenPin, OUTPUT);
  pinMode(kLedYellowPin, OUTPUT);
  pinMode(kLedRedPin, OUTPUT);
  set_status_leds(kws_fsm::LEDColor::GREEN);

  g_dsp.init();
  if (!init_mic_i2s()) {
    Serial.println("[ERROR] I2S/PDM Microphone init failed!");
  }
  if (!g_speaker.init()) {
    Serial.println("[WARN] MAX98357A Speaker I2S init failed!");
  }
  if (!g_kws.init()) {
    Serial.println("[ERROR] TFLite-Micro PSRAM engine init failed!");
  }
  g_ap.init();

  xTaskCreatePinnedToCore(inference_fsm_task, "kws_infer_fsm", 8192, nullptr, 1,
                          &g_inference_task_handle, kInferenceCore);
  xTaskCreatePinnedToCore(audio_dsp_task, "kws_audio_dsp", 8192, nullptr, 2,
                          nullptr, kAudioCore);
}

void loop() {
  vTaskDelay(pdMS_TO_TICKS(1000));
}
