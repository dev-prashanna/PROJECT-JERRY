#include <WiFi.h>
#include <ESPmDNS.h>
#include <esp_http_server.h>
#include <esp_heap_caps.h>
#include <lwip/sockets.h>
#include <lwip/tcp.h>
#include <errno.h>
#include <atomic>
#include <esp_system.h>
#include "esp_camera.h"
#include "secrets.h"

// =====================================================
// WiFi (station mode) — credentials in include/secrets.h
// =====================================================
const char* mdns_hostname = "esp32cam";

// AI-Thinker ESP32-CAM pin map (OV2640)
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
static int fbCount = 2;
static httpd_handle_t captureServer = NULL;
static httpd_handle_t streamServer = NULL;
// A disconnect invalidates an active stream, even if WiFi reconnects before
// the HTTP task gets scheduled again.
static std::atomic<uint32_t> networkGeneration{0};

// Sensor mutations come from two httpd tasks (/stream on 81, /capture on 80)
static SemaphoreHandle_t sensorMutex = NULL;
static void sensorLock()   { if (sensorMutex) xSemaphoreTake(sensorMutex, portMAX_DELAY); }
static void sensorUnlock() { if (sensorMutex) xSemaphoreGive(sensorMutex); }

// Rolling FPS of frames actually pushed out over /stream.
static uint32_t streamFpsCount = 0;
static uint32_t streamFpsStart = 0;
static uint32_t lastStreamFrameMs = 0;
static float streamFps = 0.0f;

// Auto-tuner: JPEG quality (0-63, lower = better image / bigger frames).
// Smoothness first — if the network can't sustain the target fps the tuner
// trades quality for frame rate; when there is headroom it climbs back.
static int autoQuality = 10;
static bool qualityManual = false;  // true once ?quality= overrides the tuner
static const int AUTO_Q_MIN = 10;   // best quality we ever settle at (HD)
static const int AUTO_Q_MAX = 25;   // give up quality only up to here
static const float TARGET_FPS = 20.0f;
static const float DEGRADE_FPS = 15.0f;              // below this: trade quality for fps
static const int RECOVER_WINDOWS = 2;                // consecutive good windows before improving
static int recoverStreak = 0;

// Caller holds sensorMutex, including while changing tuner state.
static void applyQualityLocked(int q) {
  sensor_t* s = esp_camera_sensor_get();
  if (!s) return;
  if (s->set_quality(s, q) == 0) {
    autoQuality = q;
    Serial.printf("[cam] quality -> %d\n", q);
  }
}

static void noteStreamFrame() {
  sensorLock();
  streamFpsCount++;
  lastStreamFrameMs = millis();
  uint32_t now = millis();
  if (streamFpsStart == 0) {
    streamFpsStart = now;
    streamFpsCount = 0;
    sensorUnlock();
    return;
  }
  uint32_t dt = now - streamFpsStart;
  if (dt >= 1000) {
    streamFps = streamFpsCount * 1000.0f / dt;

    // Degrade immediately; recover only after RECOVER_WINDOWS consecutive
    // windows near target so quality doesn't oscillate.
    if (qualityManual) {
      // operator pinned the quality via ?quality= — leave it alone
    } else if (streamFps < DEGRADE_FPS && autoQuality < AUTO_Q_MAX) {
      recoverStreak = 0;
      applyQualityLocked(autoQuality + 1);
    } else if (streamFps >= TARGET_FPS * 0.9f && autoQuality > AUTO_Q_MIN) {
      if (++recoverStreak >= RECOVER_WINDOWS) {
        recoverStreak = 0;
        applyQualityLocked(autoQuality - 1);
      }
    } else {
      recoverStreak = 0;
    }

    streamFpsCount = 0;
    streamFpsStart = now;
  }
  sensorUnlock();
}

// =====================================================
// Camera
// =====================================================
bool initCamera() {
  camera_config_t config = {};
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
  config.grab_mode = CAMERA_GRAB_LATEST;  // always newest frame, no lag build-up
  config.fb_location = CAMERA_FB_IN_PSRAM;
  config.jpeg_quality = 10;
  config.fb_count = 2;

  bool hasPsram = psramFound();
  if (hasPsram) {
    // Allocate for the largest supported OV2640 size before switching to HD.
    // Changing sensor resolution later does not grow driver frame buffers.
    config.frame_size = FRAMESIZE_UXGA;
  } else {
    config.frame_size = FRAMESIZE_QVGA;
    config.fb_count = 1;
    config.fb_location = CAMERA_FB_IN_DRAM;
  }
  fbCount = config.fb_count;

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    ESP_LOGE(TAG, "Camera init failed: 0x%x", err);
    return false;
  }

  sensor_t* s = esp_camera_sensor_get();
  if (s != NULL) {
    sensorLock();
    int result = hasPsram ? s->set_framesize(s, FRAMESIZE_HD) : 0;
    if (result == 0) result = s->set_quality(s, autoQuality);
    s->set_brightness(s, 0);
    s->set_saturation(s, 0);
    sensorUnlock();
    if (result != 0) {
      Serial.println("[cam] sensor rejected startup settings");
      esp_camera_deinit();
      return false;
    }
  } else {
    esp_camera_deinit();
    return false;
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
// size: qvga | hvga | vga | svga | hd | numeric up to UXGA
// quality: 4..63 (lower = bigger), or auto to resume tuning
static bool parseInteger(const char* value, int* result) {
  char* end = NULL;
  errno = 0;
  long parsed = strtol(value, &end, 10);
  if (errno || end == value || *end || parsed < 0 || parsed > 63) return false;
  *result = (int)parsed;
  return true;
}

static esp_err_t badParams(httpd_req_t* req, const char* message) {
  httpd_resp_send_err(req, HTTPD_400_BAD_REQUEST, message);
  return ESP_FAIL;
}

static esp_err_t applyCaptureParams(httpd_req_t* req) {
  sensor_t* s = esp_camera_sensor_get();
  if (!s) return ESP_FAIL;

  char query[64] = {0};
  if (httpd_req_get_url_query_len(req) == 0) return ESP_OK;
  if (httpd_req_get_url_query_str(req, query, sizeof(query)) != ESP_OK)
    return badParams(req, "Query too long");

  char val[16];
  framesize_t fs = FRAMESIZE_INVALID;
  int q = -1;
  bool automatic = false;
  esp_err_t param = httpd_query_key_value(query, "size", val, sizeof(val));
  if (param != ESP_OK && param != ESP_ERR_NOT_FOUND)
    return badParams(req, "Invalid size");
  if (param == ESP_OK) {
    if (!strcasecmp(val, "qqvga")) fs = FRAMESIZE_QQVGA;
    else if (!strcasecmp(val, "qvga")) fs = FRAMESIZE_QVGA;
    else if (!strcasecmp(val, "cif")) fs = FRAMESIZE_CIF;
    else if (!strcasecmp(val, "hvga")) fs = FRAMESIZE_HVGA;
    else if (!strcasecmp(val, "vga")) fs = FRAMESIZE_VGA;
    else if (!strcasecmp(val, "svga")) fs = FRAMESIZE_SVGA;
    else if (!strcasecmp(val, "hd")) fs = FRAMESIZE_HD;
    else {
      int n;
      if (parseInteger(val, &n) && n <= FRAMESIZE_UXGA) fs = (framesize_t)n;
    }
    if (fs == FRAMESIZE_INVALID || fs > FRAMESIZE_UXGA ||
        (!psramFound() && fs > FRAMESIZE_QVGA))
      return badParams(req, "Unsupported resolution or insufficient PSRAM");
  }

  param = httpd_query_key_value(query, "quality", val, sizeof(val));
  if (param != ESP_OK && param != ESP_ERR_NOT_FOUND)
    return badParams(req, "Invalid quality");
  if (param == ESP_OK) {
    automatic = !strcasecmp(val, "auto");
    if (!automatic && (!parseInteger(val, &q) || q < 4))
      return badParams(req, "Quality must be 4..63 or auto");
  }
  sensorLock();
  int result = 0;
  if (fs != FRAMESIZE_INVALID) result = s->set_framesize(s, fs);
  if (result == 0 && q >= 0) {
    result = s->set_quality(s, q);
    if (result == 0) { autoQuality = q; qualityManual = true; }
  }
  if (result == 0 && automatic) {
    qualityManual = false;
    applyQualityLocked(constrain(autoQuality, AUTO_Q_MIN, AUTO_Q_MAX));
  }
  if (fs != FRAMESIZE_INVALID || q >= 0 || automatic) {
    // A window spanning two resolutions/qualities cannot tune either one.
    streamFpsCount = 0;
    streamFpsStart = millis();
    streamFps = 0;
  }
  recoverStreak = 0;
  sensorUnlock();
  return result == 0 ? ESP_OK : badParams(req, "Sensor rejected settings");
}

static esp_err_t stream_handler(httpd_req_t* req) {
  if (!cameraOk) return httpd_resp_send_500(req);

  if (applyCaptureParams(req) != ESP_OK) return ESP_FAIL;
  sensorLock();
  streamFpsCount = 0;
  streamFpsStart = millis();
  streamFps = 0;
  recoverStreak = 0;
  sensorUnlock();

  esp_err_t res = httpd_resp_set_type(req, STREAM_CONTENT_TYPE);
  if (res != ESP_OK) return res;
  httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
  httpd_resp_set_hdr(req, "Cache-Control", "no-store, no-cache, must-revalidate");

  // Stage each multipart frame in PSRAM and return the camera buffer before
  // network I/O. HTTP chunk framing still involves multiple socket writes;
  // openSocket disables Nagle on the actual server-side connection.
  char part_hdr[64];
  uint8_t* out = NULL;
  size_t cap = 0;
  const uint32_t generation = networkGeneration.load();

  while (WiFi.status() == WL_CONNECTED &&
         networkGeneration.load() == generation) {
    sensorLock();
    camera_fb_t* fb = esp_camera_fb_get();
    if (!fb) {
      sensorUnlock();
      ESP_LOGE(TAG, "Frame capture failed");
      res = ESP_FAIL;
      break;
    }
    if (fb->format != PIXFORMAT_JPEG || fb->len < 4 ||
        fb->buf[0] != 0xff || fb->buf[1] != 0xd8) {
      esp_camera_fb_return(fb);
      sensorUnlock();
      Serial.println("[stream] invalid JPEG frame; closing connection");
      res = ESP_FAIL;
      break;
    }

    size_t hlen = snprintf(part_hdr, sizeof(part_hdr), STREAM_PART, fb->len);
    size_t need = STREAM_BOUNDARY_LEN + hlen + fb->len;

    if (need > cap) {
      // Content is rewritten every iteration, so no copy of old data needed.
      size_t nextCap = (need + 16383) & ~((size_t)16383);
      uint8_t* grown = (uint8_t*)heap_caps_realloc(out, nextCap,
          (psramFound() ? MALLOC_CAP_SPIRAM : MALLOC_CAP_INTERNAL) | MALLOC_CAP_8BIT);
      if (!grown) {
        esp_camera_fb_return(fb);
        sensorUnlock();
        res = ESP_ERR_NO_MEM;
        Serial.printf("[stream] cannot allocate %u-byte frame staging buffer\n",
                      (unsigned)nextCap);
        break;
      }
      out = grown;
      cap = nextCap;
    }

    size_t off = 0;
    memcpy(out + off, STREAM_BOUNDARY, STREAM_BOUNDARY_LEN);
    off += STREAM_BOUNDARY_LEN;
    memcpy(out + off, part_hdr, hlen);
    off += hlen;
    memcpy(out + off, fb->buf, fb->len);
    off += fb->len;
    esp_camera_fb_return(fb);
    sensorUnlock();

    res = httpd_resp_send_chunk(req, (const char*)out, off);
    if (res != ESP_OK) break;
    noteStreamFrame();
    // Give other tasks a turn even with small frames on a fast LAN.
    delay(1);
  }

  // This is an endless multipart response. Returning ESP_OK after a WiFi
  // loss would leave an incomplete response attached to a reusable socket.
  if (res == ESP_OK) res = ESP_FAIL;
  free(out);
  sensorLock();
  streamFps = 0;
  lastStreamFrameMs = 0;
  sensorUnlock();
  return res;
}

static esp_err_t capture_handler(httpd_req_t* req) {
  if (!cameraOk) return httpd_resp_send_500(req);

  if (applyCaptureParams(req) != ESP_OK) return ESP_FAIL;

  sensorLock();
  camera_fb_t* fb = esp_camera_fb_get();
  if (!fb) {
    sensorUnlock();
    httpd_resp_send_500(req);
    return ESP_FAIL;
  }

  // Do not hold the camera mutex or a driver buffer while a still-image
  // client is receiving data; that would pause the live stream.
  size_t len = fb->len;
  uint8_t* image = (uint8_t*)heap_caps_malloc(len,
      (psramFound() ? MALLOC_CAP_SPIRAM : MALLOC_CAP_INTERNAL) | MALLOC_CAP_8BIT);
  if (!image) {
    esp_camera_fb_return(fb);
    sensorUnlock();
    httpd_resp_send_500(req);
    return ESP_ERR_NO_MEM;
  }
  memcpy(image, fb->buf, len);
  esp_camera_fb_return(fb);
  sensorUnlock();

  httpd_resp_set_type(req, "image/jpeg");
  httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
  httpd_resp_set_hdr(req, "Content-Disposition", "inline; filename=capture.jpg");
  esp_err_t res = httpd_resp_send(req, (const char*)image, len);
  free(image);
  return res;
}

static esp_err_t status_handler(httpd_req_t* req) {
  sensor_t* s = esp_camera_sensor_get();
  int framesize = -1, quality = -1;
  sensorLock();
  if (s) {
    framesize = s->status.framesize;
    quality = s->status.quality;
  }

  // Report fps only while frames are actually flowing (2 s staleness window)
  float fpsOut = (lastStreamFrameMs && millis() - lastStreamFrameMs < 2000)
                     ? streamFps : 0.0f;
  bool manual = qualityManual;
  sensorUnlock();

  char json[320];
  snprintf(json, sizeof(json),
           "{\"camera\":%s,\"psram\":%s,\"framesize\":%d,\"quality\":%d,"
           "\"fb_count\":%d,\"quality_auto\":%s,\"stream_fps\":%.1f,\"heap\":%u,\"heap_psram\":%u,"
           "\"rssi\":%d,\"ip\":\"%s\",\"uptime_s\":%u}",
           cameraOk ? "true" : "false",
           psramFound() ? "true" : "false",
           framesize,
           quality,
           fbCount,
           manual ? "false" : "true",
           fpsOut,
           (unsigned)ESP.getFreeHeap(),
           (unsigned)heap_caps_get_free_size(MALLOC_CAP_SPIRAM),
           WiFi.status() == WL_CONNECTED ? (int)WiFi.RSSI() : 0,
           WiFi.status() == WL_CONNECTED ? WiFi.localIP().toString().c_str() : "",
           (unsigned)(millis() / 1000));
  httpd_resp_set_type(req, "application/json");
  return httpd_resp_send(req, json, strlen(json));
}

// =====================================================
// Two httpd instances: /stream owns its own task so it
// can block forever without starving /capture and /status.
// =====================================================
static esp_err_t openSocket(httpd_handle_t handle, int sockfd) {
  int enabled = 1;
  return setsockopt(sockfd, IPPROTO_TCP, TCP_NODELAY, &enabled,
                    sizeof(enabled)) == 0 ? ESP_OK : ESP_FAIL;
}

static httpd_handle_t startCaptureServer() {
  httpd_config_t config = HTTPD_DEFAULT_CONFIG();
  config.server_port = 80;
  config.ctrl_port = 32768;
  config.max_open_sockets = 4;
  config.max_uri_handlers = 4;
  config.stack_size = 8192;
  config.lru_purge_enable = true;
  config.open_fn = openSocket;
  config.send_wait_timeout = 3;

  httpd_handle_t handle = NULL;
  esp_err_t err = httpd_start(&handle, &config);
  if (err != ESP_OK) {
    Serial.printf("[http] capture/status server start failed: 0x%x\n", err);
    return NULL;
  }

  const httpd_uri_t capture_uri = {
    .uri = "/capture", .method = HTTP_GET, .handler = capture_handler, .user_ctx = NULL
  };
  const httpd_uri_t status_uri = {
    .uri = "/status", .method = HTTP_GET, .handler = status_handler, .user_ctx = NULL
  };
  if (httpd_register_uri_handler(handle, &capture_uri) != ESP_OK ||
      httpd_register_uri_handler(handle, &status_uri) != ESP_OK) {
    Serial.println("[http] capture/status route registration failed");
    httpd_stop(handle);
    return NULL;
  }

  Serial.println("[http] capture/status server on port 80");
  return handle;
}

static httpd_handle_t startStreamServer() {
  httpd_config_t config = HTTPD_DEFAULT_CONFIG();
  config.server_port = 81;
  config.ctrl_port = 32769;
  config.max_open_sockets = 2;
  config.max_uri_handlers = 2;
  config.stack_size = 8192;
  config.lru_purge_enable = true;
  config.open_fn = openSocket;
  config.send_wait_timeout = 3;

  httpd_handle_t handle = NULL;
  esp_err_t err = httpd_start(&handle, &config);
  if (err != ESP_OK) {
    Serial.printf("[http] stream server start failed: 0x%x\n", err);
    return NULL;
  }

  const httpd_uri_t stream_uri = {
    .uri = "/stream", .method = HTTP_GET, .handler = stream_handler, .user_ctx = NULL
  };
  if (httpd_register_uri_handler(handle, &stream_uri) != ESP_OK) {
    Serial.println("[http] stream route registration failed");
    httpd_stop(handle);
    return NULL;
  }

  Serial.println("[http] stream server on port 81");
  return handle;
}

// =====================================================
// WiFi / mDNS
// =====================================================
static bool mdnsUp = false;
static uint32_t lastWifiRetryMs = 0;
static uint32_t lastMdnsRetryMs = 0;
static uint32_t lastServerRetryMs = 0;
static uint32_t lastIpReportMs = 0;
static uint32_t reportedIp = 0;
static bool serversNeedRestart = false;
static bool setupReady = false;
static const uint32_t WIFI_RETRY_MS = 30000;
static const uint32_t MDNS_RETRY_MS = 10000;
static const uint32_t SERVER_RETRY_MS = 5000;
static const uint32_t IP_REPORT_MS = 30000;

struct NetworkEvent {
  arduino_event_id_t type;
  uint8_t reason;
};
static QueueHandle_t networkEvents = NULL;

// Arduino invokes this on another task. Keep HTTP/mDNS work and serial
// reporting in loop(); never stop an HTTP server from the WiFi callback.
static void onWifiEvent(arduino_event_id_t event, arduino_event_info_t info) {
  if (event != ARDUINO_EVENT_WIFI_STA_CONNECTED &&
      event != ARDUINO_EVENT_WIFI_STA_GOT_IP &&
      event != ARDUINO_EVENT_WIFI_STA_DISCONNECTED &&
      event != ARDUINO_EVENT_WIFI_STA_LOST_IP) return;
  if (event == ARDUINO_EVENT_WIFI_STA_DISCONNECTED ||
      event == ARDUINO_EVENT_WIFI_STA_LOST_IP) {
    networkGeneration.fetch_add(1);
  }
  NetworkEvent message = {event, 0};
  if (event == ARDUINO_EVENT_WIFI_STA_DISCONNECTED)
    message.reason = info.wifi_sta_disconnected.reason;
  xQueueSend(networkEvents, &message, 0);
}

static void stopNetworkServices() {
  if (mdnsUp) {
    MDNS.end();
    mdnsUp = false;
  }
  if (streamServer) {
    httpd_stop(streamServer);
    streamServer = NULL;
  }
  if (captureServer) {
    httpd_stop(captureServer);
    captureServer = NULL;
  }
}

static void reportAddress() {
  String ip = WiFi.localIP().toString();
  Serial.printf("[wifi] IP: %s  RSSI: %d dBm\n", ip.c_str(), (int)WiFi.RSSI());
  if (streamServer)
    Serial.printf("[stream] http://%s:81/stream\n", ip.c_str());
  if (captureServer)
    Serial.printf("[status] http://%s/status\n", ip.c_str());
  lastIpReportMs = millis();
}

static void startMdns() {
  if (mdnsUp || !captureServer || WiFi.status() != WL_CONNECTED) return;
  lastMdnsRetryMs = millis();
  if (MDNS.begin(mdns_hostname)) {
    MDNS.addService("http", "tcp", 80);
    if (streamServer) MDNS.addService("mjpeg", "tcp", 81);
    Serial.printf("[mdns] %s.local (status on :80, stream on :81)\n", mdns_hostname);
    mdnsUp = true;
  } else {
    lastMdnsRetryMs = millis();
    Serial.println("[mdns] failed to start");
  }
}

// Begin association without a 30-second startup stall. GOT_IP and later
// reconnects are handled identically by serviceNetwork().
static void startWifi() {
  WiFi.persistent(false);
  WiFi.setHostname(mdns_hostname);
  WiFi.onEvent(onWifiEvent);
  WiFi.mode(WIFI_STA);
  WiFi.setSleep(false);
  WiFi.setAutoReconnect(true);
  Serial.printf("[wifi] connecting to \"%s\" (2.4 GHz station mode)\n", WIFI_SSID);
  lastWifiRetryMs = millis();
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
}

static void serviceNetwork() {
  static uint32_t servicedGeneration = 0;
  NetworkEvent event;
  while (xQueueReceive(networkEvents, &event, 0) == pdTRUE) {
    switch (event.type) {
      case ARDUINO_EVENT_WIFI_STA_CONNECTED:
        Serial.println("[wifi] associated with access point; waiting for DHCP IP");
        break;
      case ARDUINO_EVENT_WIFI_STA_GOT_IP:
        // Report even if the router reassigns the same address.
        reportedIp = 0;
        serversNeedRestart = true;
        break;
      case ARDUINO_EVENT_WIFI_STA_DISCONNECTED:
        Serial.printf("[wifi] disconnected: reason %u (%s)\n", event.reason,
                      WiFi.disconnectReasonName((wifi_err_reason_t)event.reason));
        break;
      case ARDUINO_EVENT_WIFI_STA_LOST_IP:
        Serial.println("[wifi] DHCP IP lost");
        break;
      default:
        break;
    }
  }

  // This also catches a brief disconnect if the diagnostic queue was full.
  uint32_t generation = networkGeneration.load();
  if (generation != servicedGeneration) {
    servicedGeneration = generation;
    serversNeedRestart = true;
  }
  uint32_t now = millis();
  uint32_t ip = (uint32_t)WiFi.localIP();
  bool connected = WiFi.status() == WL_CONNECTED && ip != 0;
  if (!connected || serversNeedRestart || (reportedIp && reportedIp != ip)) {
    stopNetworkServices();
    serversNeedRestart = false;
    reportedIp = 0;
    lastServerRetryMs = now - SERVER_RETRY_MS;
    lastMdnsRetryMs = now - MDNS_RETRY_MS;
  }
  if (!connected) {
    if (now - lastWifiRetryMs >= WIFI_RETRY_MS) {
      lastWifiRetryMs = now;
      Serial.printf("[wifi] no IP after 30s (status=%d); retrying\n", (int)WiFi.status());
      if (!WiFi.reconnect()) Serial.println("[wifi] reconnect request failed");
    }
    return;
  }

  lastWifiRetryMs = now;
  bool servicesChanged = false;
  if ((!captureServer || !streamServer) && now - lastServerRetryMs >= SERVER_RETRY_MS) {
    lastServerRetryMs = now;
    if (!captureServer) { captureServer = startCaptureServer(); servicesChanged = true; }
    if (!streamServer) { streamServer = startStreamServer(); servicesChanged = true; }
    // Refresh advertised services if a previously failed server recovers.
    if (mdnsUp) { MDNS.end(); mdnsUp = false; }
    lastMdnsRetryMs = now - MDNS_RETRY_MS;
  }
  if (reportedIp != ip || servicesChanged || now - lastIpReportMs >= IP_REPORT_MS) {
    reportedIp = ip;
    reportAddress();
  }
  if (!mdnsUp && now - lastMdnsRetryMs >= MDNS_RETRY_MS) startMdns();
}

// =====================================================
// Setup
// =====================================================
void setup() {
  Serial.begin(115200);
  delay(200);
  Serial.println();
  Serial.printf("[boot] ESP32-CAM streamer built %s %s; serial=115200; reset_reason=%d\n",
                __DATE__, __TIME__, (int)esp_reset_reason());

  sensorMutex = xSemaphoreCreateMutex();
  if (!sensorMutex) {
    Serial.println("[fatal] cannot allocate camera mutex");
    return;
  }
  networkEvents = xQueueCreate(8, sizeof(NetworkEvent));
  if (!networkEvents) {
    Serial.println("[fatal] cannot allocate WiFi event queue");
    return;
  }

  // Start WiFi before camera initialization, so a camera failure does not
  // prevent association. Network services are started only after GOT_IP.
  startWifi();

  cameraOk = initCamera();
  Serial.printf("[cam] PSRAM=%s; frame buffers=%d\n", psramFound() ? "yes" : "no", fbCount);
  if (cameraOk) {
    Serial.println("[ok] camera ready");
  } else {
    Serial.println("[warn] camera init failed — network will still come up; /status reports camera:false");
  }

  setupReady = true;
  serviceNetwork();
}

// =====================================================
// Loop — background WiFi/mDNS recovery only
// =====================================================
void loop() {
  if (setupReady) serviceNetwork();
  delay(10);
}
