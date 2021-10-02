#include <ESP8266WiFi.h>
#include <ESP8266WebServer.h>
#include "DHT.h"
#define DHTPIN 2     // what digital pin the DHT22 is conected to
#define DHTTYPE DHT22   // there are multiple kinds of DHT sensors

const char* ssid = "lamb";
const char* pswd = "asdfghjkl88";

ESP8266WebServer server(80);

DHT dht(DHTPIN, DHTTYPE);

void setup() {
  Serial.begin(9600);
  Serial.println(F("Device Started"));
  
  Serial.println("Connecting to ");
  Serial.println(ssid);

  //connect to local wi-fi network
  WiFi.begin(ssid, pswd);

  //check wi-fi is connected to wi-fi network
  while (WiFi.status() != WL_CONNECTED) {
    delay(1000);
    Serial.print(".");
  }
  Serial.println("");
  Serial.println("WiFi connected..!");
  Serial.print("Got IP: ");  Serial.println(WiFi.localIP());

  server.on("/", handle_OnConnect);
  server.onNotFound(handle_NotFound);

  server.begin();
  Serial.println("HTTP server started");

  dht.begin();
}


void loop() {

  server.handleClient();


  /*delay(2000);

  // Reading temperature or humidity takes about 250 milliseconds!
  // Sensor readings may also be up to 2 seconds 'old' (its a very slow sensor)
  float h = dht.readHumidity();
  // Read temperature as Celsius (the default)
  float t = dht.readTemperature();
  // Read temperature as Fahrenheit (isFahrenheit = true)
  float f = dht.readTemperature(true);

  // Check if any reads failed and exit early (to try again).
  if (isnan(h) || isnan(t) || isnan(f)) {
    Serial.println(F("Failed to read from DHT sensor!"));
    return;
  }

  // Compute heat index in Fahrenheit (the default)
  float hif = dht.computeHeatIndex(f, h);
  // Compute heat index in Celsius (isFahreheit = false)
  float hic = dht.computeHeatIndex(t, h, false);

  Serial.print(F("Humidity: "));
  Serial.print(h);
  Serial.print(F("%  Temperature: "));
  Serial.print(t);
  Serial.print(F("°C "));
  Serial.print(f);*/
}

void handle_OnConnect(){
  float Temperature;
  float Humidity;

  Temperature = dht.readTemperature();
  Humidity = dht.readHumidity();

  server.send(200, "text/plain", String(Temperature)+" *C\n"+String(Humidity)+"%");
}



void handle_NotFound(){
  server.send(404, "text/plain", "Not Found");  
}
