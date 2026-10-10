#include <Arduino.h>

// =====================================================
// Servo on GPIO14 — 50 Hz PWM driven straight from LEDC
// =====================================================
#define SERVO_PIN 14

#define PWM_FREQ_HZ 50
#define PWM_RES_BITS 16
#define PERIOD_US 20000UL

#define PULSE_MIN_US 500   // 0 degrees
#define PULSE_MAX_US 2500  // 180 degrees

#define ANGLE_MIN 0
#define ANGLE_MAX 180
#define START_ANGLE 90

#define SWEEP_STEP_MS 15  // ~12 degrees/second

static int currentAngle = START_ANGLE;
static bool attached = true;

static bool sweeping = false;
static int sweepDir = 1;
static unsigned long lastStepMs = 0;

static char cmdBuf[32];
static uint8_t cmdLen = 0;

static uint32_t angleToDuty(int angle) {
  angle = constrain(angle, ANGLE_MIN, ANGLE_MAX);
  long us = map(angle, ANGLE_MIN, ANGLE_MAX, PULSE_MIN_US, PULSE_MAX_US);
  return (uint32_t)((us << PWM_RES_BITS) / PERIOD_US);
}

static void attachServo() {
#if ESP_ARDUINO_VERSION_MAJOR >= 3
  ledcAttach(SERVO_PIN, PWM_FREQ_HZ, PWM_RES_BITS);
#else
  ledcSetup(0, PWM_FREQ_HZ, PWM_RES_BITS);
  ledcAttachPin(SERVO_PIN, 0);
#endif
  attached = true;
}

static void detachServo() {
#if ESP_ARDUINO_VERSION_MAJOR >= 3
  ledcDetach(SERVO_PIN);
#else
  ledcDetachPin(SERVO_PIN);
#endif
  attached = false;
}

static void writeAngle(int angle) {
  angle = constrain(angle, ANGLE_MIN, ANGLE_MAX);
  currentAngle = angle;
  if (!attached) attachServo();
#if ESP_ARDUINO_VERSION_MAJOR >= 3
  ledcWrite(SERVO_PIN, angleToDuty(angle));
#else
  ledcWrite(0, angleToDuty(angle));
#endif
}

static void writeWidth(unsigned int us) {
  us = constrain(us, PULSE_MIN_US, PULSE_MAX_US);
  currentAngle = map(us, PULSE_MIN_US, PULSE_MAX_US, ANGLE_MIN, ANGLE_MAX);
  if (!attached) attachServo();
  uint32_t duty = (uint32_t)((us << PWM_RES_BITS) / PERIOD_US);
#if ESP_ARDUINO_VERSION_MAJOR >= 3
  ledcWrite(SERVO_PIN, duty);
#else
  ledcWrite(0, duty);
#endif
}

// Parses one newline-terminated command:
//   A<deg> / <deg>  set angle      e.g. "A90" or "90"
//   W<us>           raw pulse      e.g. "W1500"
//   S               toggle sweep
//   C               center (90)
//   D               detach (stop pulses)
//   ?               help
static void execute(const char* cmd) {
  if (!*cmd) return;

  if (cmd[0] == '?') {
    Serial.println("A<deg>|<deg>=angle  W<us>=width  S=sweep  C=center  D=detach  ?=help");
    return;
  }
  if (cmd[0] == 'S' || cmd[0] == 's') {
    sweeping = !sweeping;
    if (sweeping) {
      sweepDir = (currentAngle >= ANGLE_MAX) ? -1 : (currentAngle <= ANGLE_MIN) ? 1 : sweepDir;
      lastStepMs = millis();
    }
    Serial.printf("[servo] sweep %s\n", sweeping ? "on" : "off");
    return;
  }
  if (cmd[0] == 'C' || cmd[0] == 'c') {
    sweeping = false;
    writeAngle(90);
    Serial.println("[servo] angle -> 90");
    return;
  }
  if (cmd[0] == 'D' || cmd[0] == 'd') {
    sweeping = false;
    detachServo();
    Serial.println("[servo] detached");
    return;
  }

  const char* num = cmd;
  unsigned int value = 0;

  if (cmd[0] == 'A' || cmd[0] == 'a') {
    num++;
  } else if (cmd[0] == 'W' || cmd[0] == 'w') {
    num++;
    if (!isdigit(*num)) {
      Serial.println("[servo] bad width");
      return;
    }
    value = strtoul(num, NULL, 10);
    sweeping = false;
    writeWidth(value);
    Serial.printf("[servo] width -> %u us (%d deg)\n", value, currentAngle);
    return;
  }

  if (!isdigit(*num)) {
    Serial.println("[servo] unknown command — try ?");
    return;
  }
  value = strtoul(num, NULL, 10);
  if (value > ANGLE_MAX) {
    Serial.println("[servo] angle out of range (0-180)");
    return;
  }
  sweeping = false;
  writeAngle(value);
  Serial.printf("[servo] angle -> %u\n", value);
}

void setup() {
  Serial.begin(115200);
  attachServo();
  writeAngle(START_ANGLE);

  Serial.println("\n============================");
  Serial.println("Servo ready (signal=GPIO14)");
  Serial.println("A<deg>|<deg>  W<us>  S=sweep  C=center  D=detach  ?=help");
  Serial.println("============================");
}

void loop() {
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n' || c == '\r') {
      if (cmdLen) {
        cmdBuf[cmdLen] = '\0';
        execute(cmdBuf);
        cmdLen = 0;
      }
    } else if (cmdLen < sizeof(cmdBuf) - 1) {
      cmdBuf[cmdLen++] = c;
    }
  }

  if (sweeping && (millis() - lastStepMs) >= SWEEP_STEP_MS) {
    lastStepMs = millis();
    int next = currentAngle + sweepDir;
    if (next >= ANGLE_MAX) {
      next = ANGLE_MAX;
      sweepDir = -1;
    } else if (next <= ANGLE_MIN) {
      next = ANGLE_MIN;
      sweepDir = 1;
    }
    writeAngle(next);
  }
}
