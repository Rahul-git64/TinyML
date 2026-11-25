#pragma once

#include <Arduino.h>
#include <esp_heap_caps.h>
#include <math.h>
#include <stdint.h>

#include "config.h"
#include "dsp_tables.h"
#include "stage1_wake_model.h"
#include "stage2_verify_model.h"

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace kws_ml {

class DualStageTflmEngine {
 public:
  bool init() {
    tensor_arena_ = static_cast<uint8_t*>(
        heap_caps_malloc(kTensorArenaSize, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
    if (!tensor_arena_) {
      tensor_arena_ = static_cast<uint8_t*>(malloc(kTensorArenaSize));
    }
    if (!tensor_arena_) {
      Serial.println("[TFLM] Failed to allocate tensor_arena in PSRAM/SRAM");
      return false;
    }

    resolver_.AddConv2D();
    resolver_.AddDepthwiseConv2D();
    resolver_.AddAveragePool2D();
    resolver_.AddReshape();
    resolver_.AddFullyConnected();
    resolver_.AddSoftmax();

    return load_stage(1);
  }

  bool load_stage(uint8_t stage) {
    if (stage == active_stage_ && interpreter_ != nullptr) {
      return true;
    }
    const unsigned char* model_bytes =
        (stage == 2) ? g_stage2_verify_model_data : g_stage1_wake_model_data;
    const tflite::Model* model = tflite::GetModel(model_bytes);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
      Serial.printf("[TFLM] Schema mismatch: %lu != %d\n",
                    static_cast<unsigned long>(model->version()),
                    TFLITE_SCHEMA_VERSION);
      return false;
    }

    // Reconstruct interpreter in-place reusing the single shared PSRAM tensor_arena
    if (interpreter_ != nullptr) {
      interpreter_->~MicroInterpreter();
      interpreter_ = nullptr;
    }
    interpreter_ = new (interpreter_storage_) tflite::MicroInterpreter(
        model, resolver_, tensor_arena_, kTensorArenaSize);

    if (interpreter_->AllocateTensors() != kTfLiteOk) {
      Serial.printf("[TFLM] AllocateTensors failed for stage %u\n", stage);
      return false;
    }

    input_ = interpreter_->input(0);
    output_ = interpreter_->output(0);
    active_stage_ = stage;
    return true;
  }

  bool predict(const float* spec_49x40, float out_probs[3]) {
    if (!interpreter_ || !input_ || !output_) return false;

    const float in_scale = input_->params.scale;
    const int32_t in_zp = input_->params.zero_point;
    constexpr int kTotalInputs = kws_dsp::kNumFrames * kws_dsp::kNumMelBins;

    for (int i = 0; i < kTotalInputs; ++i) {
      int32_t q = static_cast<int32_t>(lroundf(spec_49x40[i] / in_scale)) + in_zp;
      if (q < -128) q = -128;
      if (q > 127) q = 127;
      input_->data.int8[i] = static_cast<int8_t>(q);
    }

    if (interpreter_->Invoke() != kTfLiteOk) {
      Serial.println("[TFLM] Invoke() failed");
      return false;
    }

    const float out_scale = output_->params.scale;
    const int32_t out_zp = output_->params.zero_point;
    float sum = 0.0f;
    for (int c = 0; c < 3; ++c) {
      float p = (static_cast<int32_t>(output_->data.int8[c]) - out_zp) * out_scale;
      if (p < 0.0f) p = 0.0f;
      if (p > 1.0f) p = 1.0f;
      out_probs[c] = p;
      sum += p;
    }
    if (sum > 1e-6f) {
      for (int c = 0; c < 3; ++c) {
        out_probs[c] /= sum;
      }
    }
    return true;
  }

  uint8_t active_stage() const { return active_stage_; }
  size_t arena_used_bytes() const {
    return interpreter_ ? interpreter_->arena_used_bytes() : 0;
  }

 private:
  uint8_t* tensor_arena_{nullptr};
  tflite::MicroMutableOpResolver<6> resolver_;
  alignas(16) uint8_t interpreter_storage_[sizeof(tflite::MicroInterpreter)]{};
  tflite::MicroInterpreter* interpreter_{nullptr};
  TfLiteTensor* input_{nullptr};
  TfLiteTensor* output_{nullptr};
  uint8_t active_stage_{0};
};

}  // namespace kws_ml
