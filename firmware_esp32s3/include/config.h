#pragma once

#include <stdint.h>

// -------------------- Audio Input (Mic) --------------------
#ifndef USE_XIAO_PDM_MIC
#define USE_XIAO_PDM_MIC 1
#endif

// XIAO ESP32S3 Sense onboard PDM Mic
constexpr int kPdmMicClkPin = 42;
constexpr int kPdmMicDataPin = 41;

// Generic ESP32-S3 + INMP441 I2S Mic
constexpr int kInmp441SckPin = 7;
constexpr int kInmp441WsPin = 8;
constexpr int kInmp441SdPin = 9;

// -------------------- Audio Output (MAX98357A I2S Amp) --------------------
constexpr int kSpkBclkPin = 1;
constexpr int kSpkLrcPin = 2;
constexpr int kSpkDinPin = 3;

// -------------------- Physical Cancel Button & Status LEDs --------------------
// D3 on XIAO ESP32S3 maps to GPIO4 (active LOW with internal pullup)
constexpr int kCancelButtonPin = 4;
constexpr uint32_t kButtonDebounceMs = 50;

constexpr int kLedGreenPin = 21;   // XIAO onboard user LED / Green indicator
constexpr int kLedYellowPin = 43;  // Optional external Yellow LED
constexpr int kLedRedPin = 44;     // Optional external Red LED

// -------------------- FreeRTOS Dual-Core & TFLM Memory --------------------
constexpr int kAudioCore = 0;
constexpr int kInferenceCore = 1;
constexpr int kInferenceIntervalHops = 10;  // 10 * 20 ms = 200 ms between inferences
constexpr size_t kTensorArenaSize = 64 * 1024;  // Shared PSRAM tensor arena

// -------------------- Offline Caregiver Wi-Fi SoftAP --------------------
constexpr const char* kSoftApSsid = "EmergencyKWS-AP";
constexpr const char* kSoftApPass = "caregiver123";
constexpr uint8_t kSoftApIp[4] = {192, 168, 4, 1};

// -------------------- Pre-saved Patient Medical ID Card --------------------
constexpr const char* kPatientName = "Eleanor Vance";
constexpr const char* kPatientAge = "74";
constexpr const char* kPatientBlood = "O+";
constexpr const char* kPatientConditions = "Atrial Fibrillation, Mild Asthma, Fall Risk";
constexpr const char* kPatientAllergies = "Penicillin, Sulfa drugs";
constexpr const char* kPatientMedications = "Apixaban 5mg BID, Albuterol PRN";
constexpr const char* kPatientEmergencyContact = "Marcus Vance (Son) - +1-555-0194";
constexpr const char* kPatientAddress = "42 Maplewood Terrace, Apt 3B";
