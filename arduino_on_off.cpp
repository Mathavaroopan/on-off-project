const int LED_ON = 2;   // Pin for ON LED
const int LED_OFF = 3;  // Pin for OFF LED

void setup() {
  Serial.begin(9600);
  pinMode(LED_ON, OUTPUT);
  pinMode(LED_OFF, OUTPUT);
  digitalWrite(LED_ON, LOW);
  digitalWrite(LED_OFF, LOW);
}

void loop() {
  if (Serial.available() > 0) {
    String state = Serial.readStringUntil('\n');
    state.trim();  // Remove any whitespace
    
    if (state == "on") {
      digitalWrite(LED_ON, HIGH);
      digitalWrite(LED_OFF, LOW);
    } else if (state == "off") {
      digitalWrite(LED_ON, LOW);
      digitalWrite(LED_OFF, HIGH);
    }
  }
}