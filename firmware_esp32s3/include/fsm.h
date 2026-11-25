#pragma once

#include <stdint.h>

namespace kws_fsm {

enum class FSMState : uint8_t {
  IDLE_MONITORING = 0,
  VOICE_VERIFY = 1,
  ALERT_DISPATCHED = 2,
};

enum class LEDColor : uint8_t {
  GREEN = 0,
  YELLOW = 1,
  RED = 2,
};

struct FSMConfig {
  float stage1_threshold = 0.80f;
  uint8_t stage1_consecutive_hits = 2;
  float stage2_confirm_threshold = 0.75f;
  float stage2_cancel_threshold = 0.75f;
  uint32_t verify_timeout_ms = 10000;
  uint32_t prompt_grace_ms = 2200;
  uint32_t post_cancel_cooldown_ms = 1000;
};

struct FSMTransition {
  FSMState previous_state;
  FSMState current_state;
  LEDColor led_color;
  bool entered_voice_verify;
  bool entered_alert;
  bool returned_idle;
  const char* reason;
};

class TwoStageFSM {
 public:
  explicit TwoStageFSM(const FSMConfig& cfg = FSMConfig()) : cfg_(cfg) {}

  FSMState state() const { return state_; }
  uint8_t active_stage() const { return state_ == FSMState::VOICE_VERIFY ? 2 : 1; }
  const char* alert_reason() const { return alert_reason_; }
  uint8_t consecutive_help_hits() const { return consecutive_help_hits_; }

  LEDColor led_color() const {
    if (state_ == FSMState::IDLE_MONITORING) return LEDColor::GREEN;
    if (state_ == FSMState::VOICE_VERIFY) return LEDColor::YELLOW;
    return LEDColor::RED;
  }

  const char* state_name() const {
    switch (state_) {
      case FSMState::IDLE_MONITORING:
        return "IDLE_MONITORING";
      case FSMState::VOICE_VERIFY:
        return "VOICE_VERIFY";
      case FSMState::ALERT_DISPATCHED:
        return "ALERT_DISPATCHED";
    }
    return "UNKNOWN";
  }

  uint32_t remaining_verify_ms(uint32_t now_ms) const {
    if (state_ != FSMState::VOICE_VERIFY) return 0;
    const uint32_t elapsed = now_ms - verify_started_ms_;
    if (elapsed >= cfg_.verify_timeout_ms) return 0;
    return cfg_.verify_timeout_ms - elapsed;
  }

  bool in_prompt_grace(uint32_t now_ms) const {
    if (state_ != FSMState::VOICE_VERIFY) return false;
    return (now_ms - verify_started_ms_) < cfg_.prompt_grace_ms;
  }

  FSMTransition press_cancel_button(uint32_t now_ms) {
    consecutive_help_hits_ = 0;
    cooldown_until_ms_ = now_ms + cfg_.post_cancel_cooldown_ms;
    const FSMState prev = state_;
    if (prev == FSMState::IDLE_MONITORING) {
      return {prev, state_, led_color(), false, false, false, "Idle D3 press"};
    }
    state_ = FSMState::IDLE_MONITORING;
    alert_reason_ = "";
    return {prev, state_, led_color(), false, false, true,
            "Physical Cancel Button (D3) pressed"};
  }

  FSMTransition acknowledge_alert(uint32_t now_ms) {
    const FSMState prev = state_;
    if (prev != FSMState::ALERT_DISPATCHED) {
      return {prev, state_, led_color(), false, false, false, ""};
    }
    consecutive_help_hits_ = 0;
    cooldown_until_ms_ = now_ms + cfg_.post_cancel_cooldown_ms;
    state_ = FSMState::IDLE_MONITORING;
    alert_reason_ = "";
    return {prev, state_, led_color(), false, false, true,
            "Caregiver Acknowledged & Reset"};
  }

  // Stage 1 probs: [background, unknown, help]
  // Stage 2 probs: [silence_unknown, confirm, cancel]
  FSMTransition step(const float probs[3], uint32_t now_ms) {
    const FSMState prev = state_;

    if (state_ == FSMState::IDLE_MONITORING) {
      if (static_cast<int32_t>(now_ms - cooldown_until_ms_) < 0) {
        consecutive_help_hits_ = 0;
        return {prev, state_, led_color(), false, false, false, ""};
      }
      const float p_help = probs[2];
      if (p_help >= cfg_.stage1_threshold) {
        consecutive_help_hits_++;
      } else {
        consecutive_help_hits_ = 0;
      }
      if (consecutive_help_hits_ >= cfg_.stage1_consecutive_hits) {
        consecutive_help_hits_ = 0;
        verify_started_ms_ = now_ms;
        state_ = FSMState::VOICE_VERIFY;
        return {prev, state_, led_color(), true, false, false,
                "Wake-word 'help' detected"};
      }
      return {prev, state_, led_color(), false, false, false, ""};
    }

    if (state_ == FSMState::VOICE_VERIFY) {
      if (remaining_verify_ms(now_ms) == 0) {
        state_ = FSMState::ALERT_DISPATCHED;
        alert_reason_ = "10s Timeout (Silence Fail-Safe)";
        return {prev, state_, led_color(), false, true, false, alert_reason_};
      }
      if (in_prompt_grace(now_ms)) {
        return {prev, state_, led_color(), false, false, false, ""};
      }
      const float p_confirm = probs[1];
      const float p_cancel = probs[2];
      if (p_cancel >= cfg_.stage2_cancel_threshold && p_cancel >= p_confirm) {
        cooldown_until_ms_ = now_ms + cfg_.post_cancel_cooldown_ms;
        state_ = FSMState::IDLE_MONITORING;
        return {prev, state_, led_color(), false, false, true,
                "Voice 'cancel' detected"};
      }
      if (p_confirm >= cfg_.stage2_confirm_threshold) {
        state_ = FSMState::ALERT_DISPATCHED;
        alert_reason_ = "Voice Confirmed";
        return {prev, state_, led_color(), false, true, false, alert_reason_};
      }
      return {prev, state_, led_color(), false, false, false, ""};
    }

    return {prev, state_, led_color(), false, false, false, ""};
  }

 private:
  FSMConfig cfg_;
  FSMState state_{FSMState::IDLE_MONITORING};
  uint8_t consecutive_help_hits_{0};
  uint32_t verify_started_ms_{0};
  uint32_t cooldown_until_ms_{0};
  const char* alert_reason_{""};
};

}  // namespace kws_fsm
