#include <Arduino.h>

#define TRIG_PIN 4  // D4
#define ECHO_PIN 2  // D2

#define SOUND_SPEED_CM 0.0343f
#define MAX_DISTANCE_CM 400.0f

static float readDistanceCm() {
  digitalWrite(TRIG_PIN, LOW);
  delayMicroseconds(2);
  digitalWrite(TRIG_PIN, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIG_PIN, LOW);

  long duration = pulseIn(ECHO_PIN, HIGH, 30000UL);
  if (duration == 0) {
    return -1.0f;
  }
  float cm = (duration * SOUND_SPEED_CM) / 2.0f;
  if (cm > MAX_DISTANCE_CM) {
    return -1.0f;
  }
  return cm;
}

void setup() {
  Serial.begin(115200);
  pinMode(TRIG_PIN, OUTPUT);
  pinMode(ECHO_PIN, INPUT);
  digitalWrite(TRIG_PIN, LOW);
  Serial.println("HC-SR04 ready (TRIG=D4/GPIO4, ECHO=D2/GPIO2)");
}

void loop() {
  float cm = readDistanceCm();
  if (cm < 0.0f) {
    Serial.println("Distance: out of range");
  } else {
    Serial.printf("Distance: %.2f cm\n", cm);
  }
  delay(200);
}
