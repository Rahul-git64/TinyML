#pragma once

#include <Arduino.h>
#include <driver/i2s.h>
#include <math.h>
#include <stdint.h>

#include "config.h"

#if defined(__has_include)
#if __has_include("verify_prompt_pcm.h")
#define KWS_HAS_EMBEDDED_PROMPT 1
#include "verify_prompt_pcm.h"
#else
#define KWS_HAS_EMBEDDED_PROMPT 0
#endif
#else
#define KWS_HAS_EMBEDDED_PROMPT 0
#endif

namespace kws_audio {

class SpeakerPrompt {
 public:
  bool init() {
    i2s_config_t cfg = {
        .mode = static_cast<i2s_mode_t>(I2S_MODE_MASTER | I2S_MODE_TX),
        .sample_rate = 16000,
        .bits_per_sample = I2S_BITS_PER_SAMPLE_16BIT,
        .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
        .communication_format = I2S_COMM_FORMAT_STAND_I2S,
        .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
        .dma_buf_count = 6,
        .dma_buf_len = 256,
        .use_apll = false,
        .tx_desc_auto_clear = true,
        .fixed_mclk = 0,
    };
    i2s_pin_config_t pins = {
        .bck_io_num = kSpkBclkPin,
        .ws_io_num = kSpkLrcPin,
        .data_out_num = kSpkDinPin,
        .data_in_num = I2S_PIN_NO_CHANGE,
    };
    if (i2s_driver_install(I2S_NUM_1, &cfg, 0, nullptr) != ESP_OK) return false;
    if (i2s_set_pin(I2S_NUM_1, &pins) != ESP_OK) return false;
    i2s_zero_dma_buffer(I2S_NUM_1);
    initialized_ = true;
    return true;
  }

  void play_verify_prompt() {
    if (!initialized_) return;

#if KWS_HAS_EMBEDDED_PROMPT
    size_t written = 0;
    i2s_write(I2S_NUM_1, g_verify_prompt_pcm, g_verify_prompt_pcm_len, &written,
              portMAX_DELAY);
#else
    // Built-in 3-tone urgent voice-verification chime (16 kHz 16-bit mono)
    // Replace by placing `xxd -i` 16kHz PCM in include/verify_prompt_pcm.h
    play_tone(880.0f, 220);
    play_silence(80);
    play_tone(660.0f, 220);
    play_silence(80);
    play_tone(987.77f, 320);
#endif
    i2s_zero_dma_buffer(I2S_NUM_1);
  }

 private:
  void play_tone(float freq_hz, uint32_t duration_ms) {
    constexpr int kChunk = 256;
    int16_t buf[kChunk];
    const uint32_t total_samples = (16000 * duration_ms) / 1000;
    uint32_t sent = 0;
    while (sent < total_samples) {
      const uint32_t n = (total_samples - sent < kChunk) ? (total_samples - sent) : kChunk;
      for (uint32_t i = 0; i < n; ++i) {
        const float t = static_cast<float>(sent + i) / 16000.0f;
        const float env = sinf(3.14159265f * static_cast<float>(sent + i) /
                               static_cast<float>(total_samples));
        buf[i] = static_cast<int16_t>(11000.0f * env * sinf(2.0f * 3.14159265f * freq_hz * t));
      }
      size_t written = 0;
      i2s_write(I2S_NUM_1, buf, n * sizeof(int16_t), &written, portMAX_DELAY);
      sent += n;
    }
  }

  void play_silence(uint32_t duration_ms) {
    constexpr int kChunk = 256;
    int16_t buf[kChunk] = {0};
    const uint32_t total_samples = (16000 * duration_ms) / 1000;
    uint32_t sent = 0;
    while (sent < total_samples) {
      const uint32_t n = (total_samples - sent < kChunk) ? (total_samples - sent) : kChunk;
      size_t written = 0;
      i2s_write(I2S_NUM_1, buf, n * sizeof(int16_t), &written, portMAX_DELAY);
      sent += n;
    }
  }

  bool initialized_{false};
};

}  // namespace kws_audio
