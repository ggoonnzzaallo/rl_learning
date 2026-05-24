
int receivedValue = 0;

void setup() {
  Serial.begin(9600); // Initialize serial communication
  while (!Serial) {
    ; // Wait for serial connection
  }
  Serial.println("Arduino ready!");
}

void loop() {
  // Check if data is available
  if (Serial.available() > 0) {
    // Read the incoming value
    receivedValue = Serial.parseInt();
    
    // Perform some action (e.g., blink an LED)
    if (receivedValue > 0) {
      digitalWrite(LED_BUILTIN, HIGH); // Turn on LED
      delay(receivedValue);           // Delay for the received value in ms
      digitalWrite(LED_BUILTIN, LOW); // Turn off LED
    }

    // Send a response back to Python
    Serial.print("Received: ");
    Serial.println(receivedValue);
  }
}
