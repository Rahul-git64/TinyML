"""Two-Stage Finite State Machine (FSM) for Emergency KWS.

Mirrors `firmware_esp32s3/include/fsm.h` identically:
  - IDLE_MONITORING (Green LED): Stage 1 detects 'help' over N consecutive hops.
  - VOICE_VERIFY (Yellow LED): Plays local prompt, runs 10s countdown, swaps to Stage 2:
      * 'cancel' OR Physical D3 button -> IDLE_MONITORING
      * 'confirm' -> ALERT_DISPATCHED ("Voice Confirmed")
      * 10s timeout (silence fail-safe) -> ALERT_DISPATCHED ("10s Timeout (Silence Fail-Safe)")
  - ALERT_DISPATCHED (Red LED): Caregiver alert active until D3 cancel or Caregiver Acknowledge.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class FSMState(str, Enum):
    IDLE_MONITORING = "IDLE_MONITORING"
    VOICE_VERIFY = "VOICE_VERIFY"
    ALERT_DISPATCHED = "ALERT_DISPATCHED"


class LEDColor(str, Enum):
    GREEN = "GREEN"
    YELLOW = "YELLOW"
    RED = "RED"


@dataclass
class FSMConfig:
    stage1_threshold: float = 0.80
    stage1_consecutive_hits: int = 2
    stage2_confirm_threshold: float = 0.75
    stage2_cancel_threshold: float = 0.75
    verify_timeout_sec: float = 10.0
    prompt_grace_sec: float = 2.2
    post_cancel_cooldown_sec: float = 1.0


@dataclass
class FSMEvent:
    timestamp: float
    from_state: FSMState
    to_state: FSMState
    reason: str
    confidence: float = 0.0


@dataclass
class FSMTransitionResult:
    previous_state: FSMState
    current_state: FSMState
    led_color: LEDColor
    entered_voice_verify: bool = False
    entered_alert: bool = False
    returned_idle: bool = False
    event: Optional[FSMEvent] = None


class TwoStageFSM:
    """Deterministic Two-Stage KWS Finite State Machine."""

    def __init__(self, config: Optional[FSMConfig] = None) -> None:
        self.config: FSMConfig = config or FSMConfig()
        self.state: FSMState = FSMState.IDLE_MONITORING
        self.consecutive_help_hits: int = 0
        self.verify_started_at: float = 0.0
        self.cooldown_until: float = 0.0
        self.alert_reason: str = ""
        self.alert_timestamp: Optional[float] = None
        self.alert_confidence: float = 0.0
        self.last_probs: Dict[str, float] = {}
        self.events: List[FSMEvent] = []

    @property
    def led_color(self) -> LEDColor:
        if self.state == FSMState.IDLE_MONITORING:
            return LEDColor.GREEN
        if self.state == FSMState.VOICE_VERIFY:
            return LEDColor.YELLOW
        return LEDColor.RED

    @property
    def active_stage(self) -> int:
        return 2 if self.state == FSMState.VOICE_VERIFY else 1

    def remaining_verify_seconds(self, now: float) -> float:
        if self.state != FSMState.VOICE_VERIFY:
            return 0.0
        elapsed = max(0.0, now - self.verify_started_at)
        return max(0.0, self.config.verify_timeout_sec - elapsed)

    def in_prompt_grace_period(self, now: float) -> bool:
        if self.state != FSMState.VOICE_VERIFY:
            return False
        return (now - self.verify_started_at) < self.config.prompt_grace_sec

    def _record_transition(
        self,
        to_state: FSMState,
        reason: str,
        now: float,
        confidence: float = 0.0,
    ) -> FSMTransitionResult:
        prev = self.state
        self.state = to_state
        evt = FSMEvent(
            timestamp=now,
            from_state=prev,
            to_state=to_state,
            reason=reason,
            confidence=confidence,
        )
        self.events.append(evt)
        if len(self.events) > 100:
            self.events = self.events[-100:]

        return FSMTransitionResult(
            previous_state=prev,
            current_state=to_state,
            led_color=self.led_color,
            entered_voice_verify=(to_state == FSMState.VOICE_VERIFY and prev != FSMState.VOICE_VERIFY),
            entered_alert=(to_state == FSMState.ALERT_DISPATCHED and prev != FSMState.ALERT_DISPATCHED),
            returned_idle=(to_state == FSMState.IDLE_MONITORING and prev != FSMState.IDLE_MONITORING),
            event=evt,
        )

    def press_cancel_button(self, now: float) -> FSMTransitionResult:
        """Handle Physical Cancel Button (D3) press in any state."""
        self.consecutive_help_hits = 0
        self.cooldown_until = now + self.config.post_cancel_cooldown_sec
        if self.state == FSMState.IDLE_MONITORING:
            return FSMTransitionResult(
                previous_state=self.state,
                current_state=self.state,
                led_color=self.led_color,
            )
        self.alert_reason = ""
        self.alert_timestamp = None
        self.alert_confidence = 0.0
        return self._record_transition(
            FSMState.IDLE_MONITORING,
            "Physical Cancel Button (D3) pressed",
            now=now,
            confidence=1.0,
        )

    def acknowledge_alert(self, now: float) -> FSMTransitionResult:
        """Caregiver acknowledges alert and resets system back to IDLE_MONITORING."""
        if self.state != FSMState.ALERT_DISPATCHED:
            return FSMTransitionResult(
                previous_state=self.state,
                current_state=self.state,
                led_color=self.led_color,
            )
        self.consecutive_help_hits = 0
        self.cooldown_until = now + self.config.post_cancel_cooldown_sec
        self.alert_reason = ""
        self.alert_timestamp = None
        self.alert_confidence = 0.0
        return self._record_transition(
            FSMState.IDLE_MONITORING,
            "Caregiver Acknowledged & Reset",
            now=now,
            confidence=1.0,
        )

    def step(self, probs: Dict[str, float], now: float) -> FSMTransitionResult:
        """Advance FSM one inference tick given class probabilities from active stage."""
        self.last_probs = dict(probs)
        prev = self.state

        if self.state == FSMState.IDLE_MONITORING:
            if now < self.cooldown_until:
                self.consecutive_help_hits = 0
                return FSMTransitionResult(prev, self.state, self.led_color)

            p_help = float(probs.get("help", 0.0))
            if p_help >= self.config.stage1_threshold:
                self.consecutive_help_hits += 1
            else:
                self.consecutive_help_hits = 0

            if self.consecutive_help_hits >= max(1, int(self.config.stage1_consecutive_hits)):
                self.consecutive_help_hits = 0
                self.verify_started_at = now
                return self._record_transition(
                    FSMState.VOICE_VERIFY,
                    f"Wake-word 'help' detected ({p_help:.2f})",
                    now=now,
                    confidence=p_help,
                )
            return FSMTransitionResult(prev, self.state, self.led_color)

        if self.state == FSMState.VOICE_VERIFY:
            # Check 10s silence fail-safe timeout first
            if self.remaining_verify_seconds(now) <= 0.0:
                self.alert_reason = "10s Timeout (Silence Fail-Safe)"
                self.alert_timestamp = now
                self.alert_confidence = 1.0
                return self._record_transition(
                    FSMState.ALERT_DISPATCHED,
                    self.alert_reason,
                    now=now,
                    confidence=1.0,
                )

            # Ignore acoustic voice triggers during speaker prompt playback grace window
            if self.in_prompt_grace_period(now):
                return FSMTransitionResult(prev, self.state, self.led_color)

            p_cancel = float(probs.get("cancel", 0.0))
            p_confirm = float(probs.get("confirm", 0.0))

            if p_cancel >= self.config.stage2_cancel_threshold and p_cancel >= p_confirm:
                self.cooldown_until = now + self.config.post_cancel_cooldown_sec
                return self._record_transition(
                    FSMState.IDLE_MONITORING,
                    f"Voice 'cancel' detected ({p_cancel:.2f})",
                    now=now,
                    confidence=p_cancel,
                )

            if p_confirm >= self.config.stage2_confirm_threshold:
                self.alert_reason = "Voice Confirmed"
                self.alert_timestamp = now
                self.alert_confidence = p_confirm
                return self._record_transition(
                    FSMState.ALERT_DISPATCHED,
                    f"Voice 'confirm' detected ({p_confirm:.2f})",
                    now=now,
                    confidence=p_confirm,
                )

            return FSMTransitionResult(prev, self.state, self.led_color)

        # ALERT_DISPATCHED remains active until D3 cancel or caregiver acknowledge
        return FSMTransitionResult(prev, self.state, self.led_color)
