//See this for context: https://www.makerguides.com/how-to-control-a-360-degree-servo-motor-with-arduino/ 
//The servos I bought from Aliexpress are 360 degree servos
//PWM signal controls speed, not position
//90deg is stop, 0-90deg is one directon, 90-180deg is the other direction

#include <Servo.h>
Servo myservo;  // Create servo object

void setup() {
  Serial.println("Arduino ready!");
  myservo.attach(11);
  myservo.write(90);
}

void loop() {
  
  myservo.write(45);
  delay(2000);
  
  myservo.write(90); 
  delay(1000);

  myservo.write(135);
  delay(2000);

  myservo.write(90); 
  delay(1000);

}
