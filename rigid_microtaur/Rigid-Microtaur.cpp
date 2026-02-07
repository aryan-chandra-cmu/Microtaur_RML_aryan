#include "leg_functions.h"
#include "leg_constants.h"
#include "wifi.h"
#include "gaits.h"

#include <WiFiNINA.h>
#include <math.h>
#include <ArxTypeTraits.h>
#include <Dynamixel.h>


void setup() {
  
  // put your setup code here, to run odnce:
  Serial.begin(9600);               // set the Serial Monitor to match this baud rate
  delay(1000); // wait for the Serial Monitor to open

  pinMode(LED_BUILTIN, OUTPUT); // setup LED
  
  configureDynamixels(); // configure the servos
  
  setUpWifi(); // set up the wifi access point

}


void loop() {
  
  //graphMotorPosition(motorOfInterest); // graph the position of a motor on the Serial Plotter
  lookAtWifi(); // tracks actions from the client

  // Get position of all motors
  syncReadPosition();

  //////////////////////// GAITS ////////////////////////

  int stepUpTime = 35; // time for each step up (legs moving up)
  int stepFowardTime = 35; // time for each step foward (legs moving foward)
  int stepDownTime = 35; // time for each step down (legs moving down)
  int sweepTime = 75; // time for each sweep (legs moving back)
  int restTime = 100; // time for each rest (legs moving back to resting position)

  if (changeInAction) { // if there is a change in action
    switch (currMotion) { // switch between the different motions
      case (motion::SIT): // if the current motion is sit
        time = 1000;
        frontsAngle = 225; // goal position for the front legs (odd #) on right side of S-Microatur
        backsAngle = 135;  // goal position for the back legs (even #) on right side of S-Microatur
        time = 1000; // time for motors to move to goal position, in miliseconds
        setPID(); // set PID gains for position
        moveAllMotors(frontsAngle, backsAngle, time);  // move all motors to goal position over time
        changeInAction = false; // there is no longer a change in action
        break;

      case (motion::STAND):
        time = 1000;
        frontsAngle = 315;
        backsAngle = 45;
        setPID();
        moveAllMotors(frontsAngle, backsAngle, time);  // move all motors to goal position over time
        changeInAction = false;
        break;
        
      case (motion::TROT):
        // A PAIR [motors 1,2,5,6]
        // B PAIR [motors 7,8,3,4]
        setPID(); // set PID gains for position

        // initially set the legs to standing position
        moveAllMotors(315, 45, 1000);
        delay(1000);

        // move the legs in a trotting motion, repeat 4 times
        for (int z = 1; z <= 5; z++){
          //Serial.println("B Step Up");
          BStepUp(stepUpTime);
          //printMotorPositions();

          //Serial.println("B Step Foward");
          BStepFoward(stepFowardTime);
          //printMotorPositions();

          //Serial.println("B Step Down");
          BStepDown(stepDownTime);
          //printMotorPositions();

          //Serial.println("B Sweep");
          BSweep(sweepTime);
          //printMotorPositions();

          //Serial.println("B Rest");
          BRest(restTime);
          //printMotorPositions();
          
          delay(300);


          //Serial.println("A Step Up");
          AStepUp(stepUpTime);
          //printMotorPositions();
          
          //Serial.println("A Step Foward");
          AStepFoward(stepFowardTime);
          //printMotorPositions();

          //Serial.println("A Step Down");
          AStepDown(stepDownTime);
          //printMotorPositions();

          //Serial.println("A Sweep");
          ASweep(sweepTime);
          //printMotorPositions();

          //Serial.println("A Rest");
          ARest(restTime);
          //printMotorPositions();

          delay(300);
        }

        currMotion = motion::SIT;
        break;

      case (motion::RESTART_SERVOS):
        restartServos();
        currMotion = motion::SIT; // go back to sitting position
        break;

      case (motion::PLAYFUL_DOG):
        time = 500;
        setPID();
        frontsAngle = 315;
        backsAngle = 90;

        moveMotor(1, frontsAngle, time);
        moveMotor(2, backsAngle, time);
        moveMotor(7, 360 - backsAngle, time);
        moveMotor(8, 360 - frontsAngle, time);


        syncWriteTime();
        syncWritePosition();

        delay(2000);

        moveMotor(1, 225, time);
        moveMotor(2, 135, time);
        moveMotor(7, 225, time);
        moveMotor(8, 135, time);

        syncWriteTime();
        syncWritePosition();

        delay(150);

        moveMotor(3, frontsAngle, time);
        moveMotor(4, backsAngle, time);
        moveMotor(5, 360 - backsAngle, time);
        moveMotor(6, 360 - frontsAngle, time);

        syncWriteTime();
        syncWritePosition();

        delay(2000);

        currMotion = motion::SIT;
        break;
    }
  }
}
