#include <WiFi.h>
#include <HTTPClient.h>
#include <Adafruit_ADXL345_U.h>
#include <Adafruit_Sensor.h>
#include <Wire.h>
#include <ArduinoJson.h>

// Definindo configurações de pinos, taxa de amostragem e LEDS
const int LED_PIN = 2;
const int SAMPLE_RATE = 200;
const int NUM_SAMPLES = 200;
const int I2C_SDA = 21, I2C_SCL = 22;

// Configurando WIFI
const char* SSID = "Belucci";
const char* PASSWORD = "974524782";

// CONFIGURANDO SERVIDOR HTTP
const char* SERVER_URL = "http://192.168.99.160:4242";

Adafruit_ADXL345_Unified accel = Adafruit_ADXL345_Unified(12345);
HTTPClient http;


// LED INDICATIVO DE PROCEDIMENTOS
void blinkLED(int times, int delayMs){
  for (int i = 0; i < times; i++){
    digitalWrite(LED_PIN, HIGH);
    delay(delayMs);
    digitalWrite(LED_PIN, LOW);
    delay(delayMs);
  }
}
void setup() {
  Serial.begin(115200);
  pinMode(LED_PIN, OUTPUT);


  Wire.begin(I2C_SDA, I2C_SCL);
  Serial.println("INICIALIZANDO I2C");

  if (!accel.begin()){
    Serial.println("ADXL345 não encontrado!");
    while (1) blinkLED(3, 200);
  }
  blinkLED(2, 100);


  accel.setRange(ADXL345_RANGE_4_G);
  accel.setDataRate(ADXL345_DATARATE_200_HZ);

  // Conectando WiFi
  Serial.print("Connecting to WiFI");
  WiFi.begin(SSID, PASSWORD);
  while(WiFi.status() != WL_CONNECTED){
    blinkLED(1, 500);
    Serial.print(".");
  }
  blinkLED(5, 50);
  Serial.printf("\nConnected! IP: %s\n", WiFi.localIP().toString().c_str());
}

bool checkServerReady(){
  http.begin(SERVER_URL);
  int httpCode = http.GET();
  bool ready = (httpCode == HTTP_CODE_OK && http.getString() == "1");
  http.end();
  return ready;
}

void sendData(JsonDocument & json){
  http.begin(SERVER_URL);
  http.addHeader("Content-Type", "application/json");
  String jsonString;
  serializeJson(json, jsonString);
  int httpCode = http.POST(jsonString);
  if (httpCode <= 0) Serial.println("Error sending data");
  http.end();
}

void loop() {
  //if (!checkServerReady()){
  //  Serial.println("SERVIDOR PRONTO");
  //  delay(100);
  //  return;
  //}

  // Preparar JSON
  DynamicJsonDocument json(3* JSON_ARRAY_SIZE(NUM_SAMPLES) + JSON_OBJECT_SIZE(3));
  JsonArray x_data = json.createNestedArray("x");
  JsonArray y_data = json.createNestedArray("y");
  JsonArray z_data = json.createNestedArray("z");

  // Coletar data
  unsigned long startTime = millis();
  int samples = 0;

  while (samples < NUM_SAMPLES){
    if (millis() - startTime >= (samples * (1000/ SAMPLE_RATE))){

      sensors_event_t event;
      accel.getEvent(&event);

      x_data.add(event.acceleration.x);
      y_data.add(event.acceleration.y);
      z_data.add(event.acceleration.z);

      Serial.printf("Samples %d: X:%.2f Y:%.2f Z:%.2f\n",
                  samples, event.acceleration.x, event.acceleration.y,
                  event.acceleration.z);

      if (samples % 50 == 0) {
        Serial.printf("Samples %d: X:%.2f Y:%.2f Z:%.2f\n",
          samples, event.acceleration.x, event.acceleration.y,
          event.acceleration.z);
      }
    
      samples++;
      delay(100);
    }
  }

  digitalWrite(LED_PIN, HIGH);
  sendData(json);
  digitalWrite(LED_PIN, LOW);
  delay(10);
}
