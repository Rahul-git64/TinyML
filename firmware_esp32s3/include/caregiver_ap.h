#pragma once

#include <Arduino.h>
#include <ArduinoJson.h>
#include <AsyncTCP.h>
#include <ESPAsyncWebServer.h>
#include <WiFi.h>

#include "config.h"
#include "fsm.h"

namespace kws_net {

static const char kCaregiverHtml[] PROGMEM = R"rawliteral(
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>ESP32-S3 Offline Caregiver Alert Dashboard</title>
<style>
  body { font-family: system-ui, sans-serif; background:#0b0f17; color:#e6edf3; margin:0; padding:20px; }
  .card { background:#131a26; border:1px solid #263248; border-radius:12px; padding:16px; margin-bottom:14px; }
  .alert { background:#450a0a; border:2px solid #ef4444; }
  .pill { display:inline-block; padding:4px 12px; border-radius:999px; font-weight:700; }
  button { background:#ef4444; color:#fff; border:none; border-radius:8px; padding:10px 16px; font-weight:700; cursor:pointer; }
</style>
</head>
<body>
  <h2>ESP32-S3 Offline Emergency KWS — Caregiver Dashboard (192.168.4.1)</h2>
  <div id="statusCard" class="card">
    <div>FSM State: <span id="state" class="pill">--</span> &nbsp; LED: <b id="led">--</b> &nbsp; Countdown: <b id="rem">--</b></div>
    <div style="margin-top:8px;">Reason: <span id="reason">--</span></div>
    <div style="margin-top:8px;">Probs: <code id="probs">--</code></div>
  </div>
  <div class="card">
    <h3>Patient Medical ID Card</h3>
    <div id="patient">Waiting for WebSocket telemetry...</div>
    <div style="margin-top:12px;"><button onclick="sendAck()">Acknowledge / Reset Alert</button></div>
  </div>
<script>
  const ws = new WebSocket("ws://" + location.host + "/ws");
  ws.onmessage = (e) => {
    const d = JSON.parse(e.data);
    document.getElementById("state").textContent = d.state;
    document.getElementById("led").textContent = d.led;
    document.getElementById("rem").textContent = (d.verify_rem_ms / 1000).toFixed(1) + "s";
    document.getElementById("reason").textContent = d.reason || "Monitoring";
    document.getElementById("probs").textContent = JSON.stringify(d.probs);
    const sc = document.getElementById("statusCard");
    sc.className = (d.state === "ALERT_DISPATCHED") ? "card alert" : "card";
    if (d.patient) {
      document.getElementById("patient").innerHTML =
        `<b>${d.patient.name}</b> (Age ${d.patient.age}, Blood ${d.patient.blood})<br>` +
        `Conditions: ${d.patient.conditions}<br>` +
        `Allergies: ${d.patient.allergies}<br>` +
        `Medications: ${d.patient.medications}<br>` +
        `Emergency Contact: ${d.patient.contact}<br>` +
        `Address: ${d.patient.address}`;
    }
  };
  function sendAck() { ws.send("ACK"); }
</script>
</body>
</html>
)rawliteral";

class CaregiverSoftAP {
 public:
  void init() {
    WiFi.mode(WIFI_AP);
    IPAddress local_ip(kSoftApIp[0], kSoftApIp[1], kSoftApIp[2], kSoftApIp[3]);
    IPAddress gateway(kSoftApIp[0], kSoftApIp[1], kSoftApIp[2], kSoftApIp[3]);
    IPAddress subnet(255, 255, 255, 0);
    WiFi.softAPConfig(local_ip, gateway, subnet);
    WiFi.softAP(kSoftApSsid, kSoftApPass);

    ws_.onEvent([this](AsyncWebSocket* server, AsyncWebSocketClient* client,
                       AwsEventType type, void* arg, uint8_t* data, size_t len) {
      if (type == WS_EVT_DATA && len >= 3 && memcmp(data, "ACK", 3) == 0) {
        ack_requested_ = true;
      }
    });
    server_.addHandler(&ws_);
    server_.on("/", HTTP_GET, [](AsyncWebServerRequest* req) {
      req->send_P(200, "text/html", kCaregiverHtml);
    });
    server_.begin();
    Serial.printf("[SoftAP] SSID=%s IP=%s\n", kSoftApSsid, WiFi.softAPIP().toString().c_str());
  }

  bool consume_ack_request() {
    if (!ack_requested_) return false;
    ack_requested_ = false;
    return true;
  }

  void broadcast_status(const kws_fsm::TwoStageFSM& fsm,
                        const float probs[3],
                        uint32_t now_ms,
                        size_t arena_used_bytes) {
    ws_.cleanupClients();
    if (ws_.count() == 0) return;

    JsonDocument doc;
    doc["state"] = fsm.state_name();
    doc["stage"] = fsm.active_stage();
    doc["led"] = (fsm.led_color() == kws_fsm::LEDColor::GREEN)
                     ? "GREEN"
                     : (fsm.led_color() == kws_fsm::LEDColor::YELLOW ? "YELLOW" : "RED");
    doc["verify_rem_ms"] = fsm.remaining_verify_ms(now_ms);
    doc["reason"] = fsm.alert_reason();
    doc["arena_bytes"] = arena_used_bytes;

    JsonArray arr = doc["probs"].to<JsonArray>();
    arr.add(probs[0]);
    arr.add(probs[1]);
    arr.add(probs[2]);

    if (fsm.state() == kws_fsm::FSMState::ALERT_DISPATCHED) {
      JsonObject pat = doc["patient"].to<JsonObject>();
      pat["name"] = kPatientName;
      pat["age"] = kPatientAge;
      pat["blood"] = kPatientBlood;
      pat["conditions"] = kPatientConditions;
      pat["allergies"] = kPatientAllergies;
      pat["medications"] = kPatientMedications;
      pat["contact"] = kPatientEmergencyContact;
      pat["address"] = kPatientAddress;
    }

    String out;
    serializeJson(doc, out);
    ws_.textAll(out);
  }

 private:
  AsyncWebServer server_{80};
  AsyncWebSocket ws_{"/ws"};
  volatile bool ack_requested_{false};
};

}  // namespace kws_net
