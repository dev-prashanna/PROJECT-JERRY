#include <WiFi.h>
#include <WiFiUdp.h>

// =====================================================
// SoftAP credentials — PC must join this AP to reach 192.168.4.1
// =====================================================
const char* ssid = "Prakash 2g";
const char* password = "P1rakash@*#1";

WiFiUDP udp;
const unsigned int udpPort = 8888;
char packetBuffer[256];

// =====================================================
// L298N Motor Driver Pins
// =====================================================
// Motor A
#define ENA 25
#define A_IN1 18
#define A_IN2 19

// Motor B
#define ENB 14
#define B_IN1 21
#define B_IN2 22

// PWM via analogWrite() — LEDC under the hood, 8-bit duty = 0..255
#define PWM_FREQ 5000
#define PWM_RES_BITS 8

// Core 3.x takes per-pin arguments, core 2.x applies them globally to pins
// attached later. Must run before the first analogWrite().
static void setupPwm() {
#if ESP_ARDUINO_VERSION_MAJOR >= 3
  analogWriteResolution(ENA, PWM_RES_BITS);
  analogWriteResolution(ENB, PWM_RES_BITS);
  analogWriteFrequency(ENA, PWM_FREQ);
  analogWriteFrequency(ENB, PWM_FREQ);
#else
  analogWriteResolution(PWM_RES_BITS);
  analogWriteFrequency(PWM_FREQ);
#endif
}

int motorSpeed = 200;  // 0 - 255

// Safety: stop the car if no command arrives for this long.
// 0 = disable failsafe.
const unsigned long COMMAND_TIMEOUT_MS = 2000;

char lastAction = 'S';
unsigned long lastCommandMs = 0;

// =====================================================
// Motor Functions
// =====================================================
void writeSpeed() {
  analogWrite(ENA, motorSpeed);
  analogWrite(ENB, motorSpeed);
}

void forward() {
  writeSpeed();
  digitalWrite(A_IN1, HIGH);
  digitalWrite(A_IN2, LOW);
  digitalWrite(B_IN1, HIGH);
  digitalWrite(B_IN2, LOW);
}

void backward() {
  writeSpeed();
  digitalWrite(A_IN1, LOW);
  digitalWrite(A_IN2, HIGH);
  digitalWrite(B_IN1, LOW);
  digitalWrite(B_IN2, HIGH);
}

void left() {
  writeSpeed();
  digitalWrite(A_IN1, HIGH);
  digitalWrite(A_IN2, LOW);
  digitalWrite(B_IN1, LOW);
  digitalWrite(B_IN2, HIGH);
}

void right() {
  writeSpeed();
  digitalWrite(A_IN1, LOW);
  digitalWrite(A_IN2, HIGH);
  digitalWrite(B_IN1, HIGH);
  digitalWrite(B_IN2, LOW);
}

void stopCar() {
  analogWrite(ENA, 0);
  analogWrite(ENB, 0);
  digitalWrite(A_IN1, LOW);
  digitalWrite(A_IN2, LOW);
  digitalWrite(B_IN1, LOW);
  digitalWrite(B_IN2, LOW);
  lastAction = 'S';
}

// Executes a single-letter drive command from driver.py
void execute(char command) {
  switch (command) {
    case 'F': forward(); lastAction = 'F'; break;
    case 'B': backward(); lastAction = 'B'; break;
    case 'L': left(); lastAction = 'L'; break;
    case 'R': right(); lastAction = 'R'; break;
    case 'S': stopCar(); break;
    default: break;  // unknown command ignored
  }
}

void setup() {
  Serial.begin(115200);

  pinMode(ENA, OUTPUT);
  pinMode(A_IN1, OUTPUT);
  pinMode(A_IN2, OUTPUT);
  pinMode(ENB, OUTPUT);
  pinMode(B_IN1, OUTPUT);
  pinMode(B_IN2, OUTPUT);

  setupPwm();

  stopCar();  // safe, stopped state at boot

  WiFi.softAP(ssid, password);
  udp.begin(udpPort);
  lastCommandMs = millis();

  Serial.println("\n============================");
  Serial.print("UDP Receiver Started on Port: ");
  Serial.println(udpPort);
  Serial.print("ESP32 IP: ");
  Serial.println(WiFi.softAPIP());
  Serial.println("============================");
}

void loop() {
  int packetSize = udp.parsePacket();

  if (packetSize) {
    int len = udp.read(packetBuffer, sizeof(packetBuffer) - 1);
    if (len > 0) {
      packetBuffer[len] = '\0';
      lastCommandMs = millis();

      char command = packetBuffer[0];
      if (command == 'V') {
        motorSpeed = constrain(atoi(&packetBuffer[1]), 0, 255);
        Serial.printf("[cmd] speed -> %d\n", motorSpeed);
        if (lastAction != 'S') execute(lastAction);  // apply immediately while driving
      } else {
        Serial.printf("[cmd] action -> %c\n", command);
        execute(command);
      }
    }
  }

  // Failsafe: silence from the controller means stop.
  if (COMMAND_TIMEOUT_MS > 0 && lastAction != 'S' &&
      (millis() - lastCommandMs) > COMMAND_TIMEOUT_MS) {
    Serial.println("[safety] command timeout -> stop");
    stopCar();
  }
}
