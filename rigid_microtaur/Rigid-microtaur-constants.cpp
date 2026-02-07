#include "leg_constants.h"
#include "iostream"

DynamixelShield dxl;
using namespace ControlTableItem;

// Sync read/write structs
ParamForSyncWriteInst_t sync_write_param;
ParamForSyncReadInst_t sync_read_param;
RecvInfoFromStatusInst_t read_result;

const int MOTOR_COUNT = 8;

// These arrays are of length 9 to have index = motor number
float present_position[MOTOR_COUNT + 1];
float goal_position[MOTOR_COUNT + 1];
float goal_time[MOTOR_COUNT + 1];

float start_position_offset[MOTOR_COUNT + 1];

short present_current[MOTOR_COUNT + 1];

int total_current_usage = 0;

bool changeInAction = true;

bool hasSpine = false;

bool spineRigid = false;

motion currMotion = motion::SIT;  // set initial motion to sit

// Choose motor to graph on Serial Plotter (for debugging)
int motorOfInterest = 5;

// positions for the front and back legs
int frontsAngle = 0;
int backsAngle = 0;

// positions for each type of leg (look at microtaur from its eyes to see which leg is which)
int right_front = 0;
int right_back = 0;
int left_front = 0;
int left_back = 0;

// initial time for legs to move to goal position
int time = 1000;

// Variable to store the offset of the front motors (side with the battery)
int offset = -15;

// Spine motor goal position for Rigid-Microtaur (no spine motor)
float spineGoalPosition = 0.0;
