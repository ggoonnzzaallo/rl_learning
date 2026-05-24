#include <Servo.h>

Servo myservo;  // Create servo object
int receivedValue = 0;

void setup() {
  Serial.begin(9600);
  while (!Serial) {
    ; // Wait for serial connection
  }
  myservo.attach(11);  // Attaches the servo on pin 11
  Serial.println("Arduino ready!");
}

void loop() {
  if (Serial.available() > 0) {
    receivedValue = Serial.parseInt();
    
    // Ensure the angle is within valid range (0-180 degrees)
    receivedValue = constrain(receivedValue, 0, 180);
    
    // Move servo to the received angle
    myservo.write(receivedValue);

    // Send a response back to Python
    Serial.print("Moved servo to: ");
    Serial.println(receivedValue);
  }
}
