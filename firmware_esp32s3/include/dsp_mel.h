#pragma once

#include <math.h>
#include <stdint.h>
#include <string.h>

#include "dsp_tables.h"

#if defined(__has_include)
#if __has_include("dsps_fft2r.h")
#define KWS_HAS_ESP_DSP 1
#include "dsps_fft2r.h"
#else
#define KWS_HAS_ESP_DSP 0
#endif
#else
#define KWS_HAS_ESP_DSP 0
#endif

namespace kws_dsp {

class StreamingLogMelDSP {
 public:
  bool init() {
    reset();
#if KWS_HAS_ESP_DSP
    if (!fft_initialized_) {
      esp_err_t ret = dsps_fft2r_init_fc32(nullptr, kNfft);
      fft_initialized_ = (ret == ESP_OK);
    }
#endif
    return true;
  }

  void reset() {
    buf_len_ = 0;
    frames_computed_ = 0;
    memset(sample_buf_, 0, sizeof(sample_buf_));
    for (int f = 0; f < kNumFrames; ++f) {
      for (int m = 0; m < kNumMelBins; ++m) {
        spec_ring_[f][m] = kFeatClipMin;
      }
    }
  }

  // Push one 320-sample (20 ms) hop normalized to [-1.0f, 1.0f].
  // Returns true if a new 40-bin Log-Mel frame was appended to spec_ring_.
  bool push_hop(const float* hop_320) {
    if (buf_len_ + kHopLen > kWinLen + kHopLen) {
      buf_len_ = 0;
    }
    memcpy(sample_buf_ + buf_len_, hop_320, kHopLen * sizeof(float));
    buf_len_ += kHopLen;

    bool computed = false;
    while (buf_len_ >= kWinLen) {
      float frame_feats[kNumMelBins];
      compute_frame_log_mel(sample_buf_, frame_feats);

      // Shift spectrogram ring left by 1 frame and append newest frame
      memmove(&spec_ring_[0][0], &spec_ring_[1][0],
              (kNumFrames - 1) * kNumMelBins * sizeof(float));
      memcpy(&spec_ring_[kNumFrames - 1][0], frame_feats,
             kNumMelBins * sizeof(float));

      // Shift sample buffer left by kHopLen (keep 160-sample overlap)
      buf_len_ -= kHopLen;
      if (buf_len_ > 0) {
        memmove(sample_buf_, sample_buf_ + kHopLen, buf_len_ * sizeof(float));
      }
      frames_computed_++;
      computed = true;
    }
    return computed;
  }

  void copy_spectrogram(float* dst_49x40) const {
    memcpy(dst_49x40, spec_ring_, kNumFrames * kNumMelBins * sizeof(float));
  }

  uint32_t frames_computed() const { return frames_computed_; }

 private:
  void compute_frame_log_mel(const float* frame_480, float* out_40) {
    // Pack real signal into complex buffer [re0, im0, re1, im1, ...]
    for (int i = 0; i < kWinLen; ++i) {
      fft_cmplx_[2 * i] = frame_480[i] * kHannWindow[i];
      fft_cmplx_[2 * i + 1] = 0.0f;
    }
    for (int i = kWinLen; i < kNfft; ++i) {
      fft_cmplx_[2 * i] = 0.0f;
      fft_cmplx_[2 * i + 1] = 0.0f;
    }

#if KWS_HAS_ESP_DSP
    if (fft_initialized_) {
      dsps_fft2r_fc32(fft_cmplx_, kNfft);
      dsps_bit_rev_fc32(fft_cmplx_, kNfft);
    } else {
      radix2_fft(fft_cmplx_, kNfft);
    }
#else
    radix2_fft(fft_cmplx_, kNfft);
#endif

    // Power spectrum over bins 0..256 (Nyquist)
    for (int k = 0; k < kNumFftBins; ++k) {
      const float re = fft_cmplx_[2 * k];
      const float im = fft_cmplx_[2 * k + 1];
      power_bins_[k] = re * re + im * im;
    }

    // Sparse HTK Mel filterbank dot product + log scaling
    for (int m = 0; m < kNumMelBins; ++m) {
      const uint16_t start = kMelBinStart[m];
      const uint16_t len = kMelBinLen[m];
      const uint16_t w_off = kMelWeightOffset[m];
      float energy = 0.0f;
      for (uint16_t j = 0; j < len; ++j) {
        energy += power_bins_[start + j] * kMelWeights[w_off + j];
      }
      float val = (logf(energy + kLogEps) - kLogOffset) / kLogScale;
      if (val < kFeatClipMin) val = kFeatClipMin;
      if (val > kFeatClipMax) val = kFeatClipMax;
      out_40[m] = val;
    }
  }

  static void radix2_fft(float* data, int n) {
    // Bit-reversal permutation
    int j = 0;
    for (int i = 0; i < n; ++i) {
      if (i < j) {
        float tr = data[2 * i];
        float ti = data[2 * i + 1];
        data[2 * i] = data[2 * j];
        data[2 * i + 1] = data[2 * j + 1];
        data[2 * j] = tr;
        data[2 * j + 1] = ti;
      }
      int m = n >> 1;
      while (m >= 1 && j >= m) {
        j -= m;
        m >>= 1;
      }
      j += m;
    }
    // Cooley-Tukey butterfly stages
    for (int step = 2; step <= n; step <<= 1) {
      const int half = step >> 1;
      const float theta = -2.0f * 3.14159265358979323846f / static_cast<float>(step);
      const float wpr = cosf(theta);
      const float wpi = sinf(theta);
      for (int group = 0; group < n; group += step) {
        float wr = 1.0f;
        float wi = 0.0f;
        for (int k = 0; k < half; ++k) {
          const int i1 = group + k;
          const int i2 = i1 + half;
          const float tr = wr * data[2 * i2] - wi * data[2 * i2 + 1];
          const float ti = wr * data[2 * i2 + 1] + wi * data[2 * i2];
          data[2 * i2] = data[2 * i1] - tr;
          data[2 * i2 + 1] = data[2 * i1 + 1] - ti;
          data[2 * i1] += tr;
          data[2 * i1 + 1] += ti;
          const float next_wr = wr * wpr - wi * wpi;
          wi = wr * wpi + wi * wpr;
          wr = next_wr;
        }
      }
    }
  }

  float sample_buf_[kWinLen + kHopLen]{};
  int buf_len_{0};
  uint32_t frames_computed_{0};
  float spec_ring_[kNumFrames][kNumMelBins]{};
  float fft_cmplx_[2 * kNfft]{};
  float power_bins_[kNumFftBins]{};
  bool fft_initialized_{false};
};

}  // namespace kws_dsp
