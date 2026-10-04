#include <WiFi.h>
#include <ESPmDNS.h>
#include <ArduinoOTA.h>
#include <esp_http_server.h>
#include "esp_camera.h"

// =====================================================
// WiFi — Prakash 2G (station mode)
// =====================================================
const char* ssid = "Prakash 2g";
const char* password = "P1rakash@*#1";
const char* mdns_hostname = "esp32cam";
const char* OTA_PASSWORD = "prakashcam";

// =====================================================
// AI-Thinker ESP32-CAM pin map (OV2640)
// =====================================================
#define PWDN_GPIO_NUM     32
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM      0
#define SIOD_GPIO_NUM     26
#define SIOC_GPIO_NUM     27

#define Y9_GPIO_NUM       35
#define Y8_GPIO_NUM       34
#define Y7_GPIO_NUM       39
#define Y6_GPIO_NUM       36
#define Y5_GPIO_NUM       21
#define Y4_GPIO_NUM       19
#define Y3_GPIO_NUM       18
#define Y2_GPIO_NUM        5
#define VSYNC_GPIO_NUM    25
#define HREF_GPIO_NUM     23
#define PCLK_GPIO_NUM     22

#define LED_GPIO_NUM       4

static const char* TAG = "stream";
static bool cameraOk = false;

// Rolling FPS of frames actually pushed out over /stream.
static uint32_t streamFpsCount = 0;
static uint32_t streamFpsStart = 0;
static float streamFps = 0.0f;

static void noteStreamFrame() {
  streamFpsCount++;
  uint32_t now = millis();
  if (streamFpsStart == 0) {
    streamFpsStart = now;
    return;
  }
  uint32_t dt = now - streamFpsStart;
  if (dt >= 1000) {
    streamFps = streamFpsCount * 1000.0f / dt;
    streamFpsCount = 0;
    streamFpsStart = now;
  }
}

// =====================================================
// Camera
// =====================================================
bool initCamera() {
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sccb_sda = SIOD_GPIO_NUM;
  config.pin_sccb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;
  config.grab_mode = CAMERA_GRAB_LATEST;
  config.fb_location = CAMERA_FB_IN_PSRAM;
  config.jpeg_quality = 12;
  config.fb_count = 2;

  if (psramFound()) {
    config.frame_size = FRAMESIZE_VGA;
  } else {
    config.frame_size = FRAMESIZE_QVGA;
    config.fb_count = 1;
    config.fb_location = CAMERA_FB_IN_DRAM;
  }

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    ESP_LOGE(TAG, "Camera init failed: 0x%x", err);
    return false;
  }

  sensor_t* s = esp_camera_sensor_get();
  if (s != NULL) {
    s->set_framesize(s, FRAMESIZE_VGA);
    s->set_brightness(s, 1);
    s->set_saturation(s, 0);
  }

  pinMode(LED_GPIO_NUM, OUTPUT);
  digitalWrite(LED_GPIO_NUM, LOW);
  return true;
}

// =====================================================
// HTTP handlers
// =====================================================
#define PART_BOUNDARY "123456789000000000000987654321"
static const char* STREAM_CONTENT_TYPE =
    "multipart/x-mixed-replace;boundary=" PART_BOUNDARY;
static const char* STREAM_BOUNDARY = "\r\n--" PART_BOUNDARY "\r\n";
#define STREAM_BOUNDARY_LEN (sizeof("\r\n--" PART_BOUNDARY "\r\n") - 1)
static const char* STREAM_PART = "Content-Type: image/jpeg\r\nContent-Length: %u\r\n\r\n";

// Optional live tuning: /stream?size=qvga&quality=14
// size: qvga | hvga | vga | svga | hd | <0..16>   quality: 4..63 (lower = bigger)
static void applyCaptureParams(httpd_req_t* req) {
  sensor_t* s = esp_camera_sensor_get();
  if (!s) return;

  // The query can arrive either via the helper or still embedded in req->uri
  // depending on IDF version — handle both.
  char query[64] = {0};
  const char* q = strchr(req->uri, '?');
  if (q) {
    strncpy(query, q + 1, sizeof(query) - 1);
  } else if (httpd_req_get_url_query_str(req, query, sizeof(query)) != ESP_OK) {
    return;
  }

  char val[16];
  if (httpd_query_key_value(query, "size", val, sizeof(val)) == ESP_OK) {
    framesize_t fs = FRAMESIZE_INVALID;
    if (!strcasecmp(val, "qqvga")) fs = FRAMESIZE_QQVGA;
    else if (!strcasecmp(val, "qvga")) fs = FRAMESIZE_QVGA;
    else if (!strcasecmp(val, "cif")) fs = FRAMESIZE_CIF;
    else if (!strcasecmp(val, "hvga")) fs = FRAMESIZE_HVGA;
    else if (!strcasecmp(val, "vga")) fs = FRAMESIZE_VGA;
    else if (!strcasecmp(val, "svga")) fs = FRAMESIZE_SVGA;
    else if (!strcasecmp(val, "hd")) fs = FRAMESIZE_HD;
    else {
      int n = atoi(val);
      if (n >= 0 && n < FRAMESIZE_INVALID) fs = (framesize_t)n;
    }
    if (fs != FRAMESIZE_INVALID) {
      s->set_framesize(s, fs);
      Serial.printf("[cam] framesize -> %d\n", (int)fs);
    }
  }

  if (httpd_query_key_value(query, "quality", val, sizeof(val)) == ESP_OK) {
    int qq = atoi(val);
    if (qq >= 4 && qq <= 63) {
      s->set_quality(s, qq);
      Serial.printf("[cam] jpeg quality -> %d\n", qq);
    }
  }
}

static esp_err_t stream_handler(httpd_req_t* req) {
  if (!cameraOk) return httpd_resp_send_500(req);

  applyCaptureParams(req);

  esp_err_t res = httpd_resp_set_type(req, STREAM_CONTENT_TYPE);
  if (res != ESP_OK) return res;
  httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
  httpd_resp_set_hdr(req, "Cache-Control", "no-store, no-cache, must-revalidate");

  // One contiguous buffer per frame -> a single TCP write instead of three
  // small ones (the small writes were what Nagle + delayed-ACK stalled on).
  char part_hdr[64];
  uint8_t* out = NULL;
  size_t cap = 0;

  while (true) {
    camera_fb_t* fb = esp_camera_fb_get();
    if (!fb) {
      ESP_LOGE(TAG, "Frame capture failed");
      res = ESP_FAIL;
      break;
    }

    size_t hlen = snprintf(part_hdr, sizeof(part_hdr), STREAM_PART, fb->len);
    size_t need = STREAM_BOUNDARY_LEN + hlen + fb->len;

    if (need > cap) {
      uint8_t* grown = (uint8_t*)realloc(out, need);
      if (!grown) {
        esp_camera_fb_return(fb);
        res = ESP_ERR_NO_MEM;
        break;
      }
      out = grown;
      cap = need;
    }

    size_t off = 0;
    memcpy(out + off, STREAM_BOUNDARY, STREAM_BOUNDARY_LEN);
    off += STREAM_BOUNDARY_LEN;
    memcpy(out + off, part_hdr, hlen);
    off += hlen;
    memcpy(out + off, fb->buf, fb->len);
    off += fb->len;
    esp_camera_fb_return(fb);

    res = httpd_resp_send_chunk(req, (const char*)out, off);
    if (res != ESP_OK) break;
    noteStreamFrame();
  }

  free(out);
  return res;
}

static esp_err_t capture_handler(httpd_req_t* req) {
  if (!cameraOk) return httpd_resp_send_500(req);

  applyCaptureParams(req);

  camera_fb_t* fb = esp_camera_fb_get();
  if (!fb) {
    httpd_resp_send_500(req);
    return ESP_FAIL;
  }

  httpd_resp_set_type(req, "image/jpeg");
  httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
  httpd_resp_set_hdr(req, "Content-Disposition", "inline; filename=capture.jpg");
  esp_err_t res = httpd_resp_send(req, (const char*)fb->buf, fb->len);
  esp_camera_fb_return(fb);
  return res;
}

static esp_err_t status_handler(httpd_req_t* req) {
  sensor_t* s = esp_camera_sensor_get();
  char json[256];
  snprintf(json, sizeof(json),
           "{\"camera\":%s,\"psram\":%s,\"framesize\":%d,\"quality\":%d,\"fb_count\":%d,\"stream_fps\":%.1f,\"heap\":%u}",
           cameraOk ? "true" : "false",
           psramFound() ? "true" : "false",
           s ? s->status.framesize : -1,
           s ? s->status.quality : -1,
           psramFound() ? 2 : 1,
           streamFps,
           (unsigned)ESP.getFreeHeap());
  httpd_resp_set_type(req, "application/json");
  return httpd_resp_send(req, json, strlen(json));
}

static httpd_handle_t startWebServer() {
  httpd_config_t config = HTTPD_DEFAULT_CONFIG();
  config.server_port = 80;
  config.ctrl_port = 32768;
  config.max_open_sockets = 4;
  config.max_uri_handlers = 4;
  config.stack_size = 8192;
  config.lru_purge_enable = true;

  httpd_handle_t camera_httpd = NULL;
  if (httpd_start(&camera_httpd, &config) != ESP_OK) {
    ESP_LOGE(TAG, "httpd_start failed");
    return NULL;
  }

  const httpd_uri_t stream_uri = {
    .uri = "/stream", .method = HTTP_GET, .handler = stream_handler, .user_ctx = NULL
  };
  const httpd_uri_t capture_uri = {
    .uri = "/capture", .method = HTTP_GET, .handler = capture_handler, .user_ctx = NULL
  };
  const httpd_uri_t status_uri = {
    .uri = "/status", .method = HTTP_GET, .handler = status_handler, .user_ctx = NULL
  };

  httpd_register_uri_handler(camera_httpd, &stream_uri);
  httpd_register_uri_handler(camera_httpd, &capture_uri);
  httpd_register_uri_handler(camera_httpd, &status_uri);

  ESP_LOGI(TAG, "HTTP server on port 80");
  return camera_httpd;
}

// =====================================================
// Setup
// =====================================================
void setup() {
  Serial.begin(115200);
  delay(200);
  Serial.println();
  Serial.println("[boot] ESP32-CAM streamer");

  cameraOk = initCamera();
  if (cameraOk) {
    Serial.println("[ok] camera ready");
  } else {
    Serial.println("[warn] camera init failed — network will still come up; /status reports camera:false");
  }

  WiFi.mode(WIFI_STA);
  WiFi.setSleep(false);
  WiFi.setHostname(mdns_hostname);
  WiFi.begin(ssid, password);
  Serial.printf("[wifi] connecting to \"%s\"", ssid);

  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 60) {
    delay(500);
    Serial.print(".");
    attempts++;
  }

  if (WiFi.status() != WL_CONNECTED) {
    Serial.println();
    Serial.println("[fatal] WiFi connect failed — restarting");
    delay(2000);
    ESP.restart();
  }

  Serial.println();
  Serial.print("[wifi] connected, IP: ");
  Serial.println(WiFi.localIP());

  if (MDNS.begin(mdns_hostname)) {
    MDNS.addService("http", "tcp", 80);
    Serial.printf("[mdns] http://%s.local/\n", mdns_hostname);
  } else {
    Serial.println("[mdns] failed to start");
  }

  // Wireless updates — no more IO0/RST dance.
  ArduinoOTA.setHostname(mdns_hostname);
  ArduinoOTA.setPassword(OTA_PASSWORD);
  ArduinoOTA.setMdnsEnabled(false);  // we already own the mDNS instance
  ArduinoOTA.onStart([]() {
    Serial.printf("\n[ota] start (%s)\n",
                  ArduinoOTA.getCommand() == U_FLASH ? "flash" : "fs");
  });
  ArduinoOTA.onProgress([](unsigned p, unsigned t) {
    static int last = -1;
    int pct = t ? (int)(p * 100 / t) : 0;
    if (pct != last && pct % 10 == 0) {
      Serial.printf("[ota] %d%%\n", pct);
      last = pct;
    }
  });
  ArduinoOTA.onEnd([]() { Serial.println("\n[ota] done, rebooting"); });
  ArduinoOTA.onError([](ota_error_t e) {
    Serial.printf("[ota] error %u\n", (unsigned)e);
  });
  ArduinoOTA.begin();
  Serial.printf("[ota] ready -> pio run -e esp32cam_ota -t upload\n");

  startWebServer();
  digitalWrite(LED_GPIO_NUM, LOW);  // flash LED stays off
}

// =====================================================
// Loop
// =====================================================
void loop() {
  ArduinoOTA.handle();

  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("[wifi] lost connection, reconnecting...");
    WiFi.reconnect();
    delay(1000);
  }
  delay(5);
}
