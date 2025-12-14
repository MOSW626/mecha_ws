#!/usr/bin/env python3
# Controls driving based on "forward", "green", "left", "non", "red", "right" values.

import RPi.GPIO as GPIO
import time

# ==================== GPIO Settings ====================
DIR_PIN = 16
PWM_PIN = 12
SERVO_PIN = 13

MOTOR_FREQ = 1000
SERVO_FREQ = 50
SERVO_MAX_DUTY = 12
SERVO_MIN_DUTY = 3

# Speed settings
SPEED_SLOW = 18  # Very slow driving
SERVO_ANGLE_CENTER = 90
SERVO_ANGLE_MAX = 140
SERVO_ANGLE_MIN = 30

# GPIO control variables
motor_pwm = None
servo_pwm = None
initialized = False

# Driving state variables
last_direction = "forward"

def init_drive():
    """Initialize driving module"""
    global motor_pwm, servo_pwm, initialized

    if initialized:
        return

    GPIO.setmode(GPIO.BCM)
    GPIO.setup([DIR_PIN, PWM_PIN, SERVO_PIN], GPIO.OUT)

    motor_pwm = GPIO.PWM(PWM_PIN, MOTOR_FREQ)
    servo_pwm = GPIO.PWM(SERVO_PIN, SERVO_FREQ)
    motor_pwm.start(0)
    servo_pwm.start(0)

    initialized = True
    print("✓ Driving module initialized")

def set_servo_angle(degree):
    """Set servo motor angle"""
    global servo_pwm
    if not initialized or servo_pwm is None:
        return

    degree = max(SERVO_ANGLE_MIN, min(SERVO_ANGLE_MAX, degree))
    duty = SERVO_MIN_DUTY + (degree * (SERVO_MAX_DUTY - SERVO_MIN_DUTY) / 180.0)
    servo_pwm.ChangeDutyCycle(duty)

def move_forward(speed):
    """Move forward"""
    global motor_pwm
    if not initialized or motor_pwm is None:
        return

    GPIO.output(DIR_PIN, GPIO.HIGH)
    motor_pwm.ChangeDutyCycle(speed)

def move_backward(speed):
    """Move backward"""
    global motor_pwm
    if not initialized or motor_pwm is None:
        return

    GPIO.output(DIR_PIN, GPIO.LOW)
    motor_pwm.ChangeDutyCycle(speed)

def stop_motor():
    """Stop"""
    global motor_pwm
    if not initialized or motor_pwm is None:
        return

    motor_pwm.ChangeDutyCycle(0)

def drive(direction, angle=None):
    """
    Controls driving based on judgment result.

    Args:
        direction: One of "forward", "green", "left", "non", "red", "right"
        angle: Optional calculated servo angle (degrees). If provided, uses this instead of fixed angles.
    """
    global last_direction

    if not initialized:
        print("⚠ Driving module not initialized.")
        return

    if direction == "red":
        # Red light: stop
        stop_motor()
        set_servo_angle(SERVO_ANGLE_CENTER)
        last_direction = "forward"

    elif direction == "green":
        # Green light: continue driving
        # Use calculated angle if provided, otherwise maintain previous direction
        if angle is not None:
            set_servo_angle(angle)
        elif last_direction == "left":
            set_servo_angle(50)
        elif last_direction == "right":
            set_servo_angle(130)
        else:
            set_servo_angle(SERVO_ANGLE_CENTER)
        move_forward(SPEED_SLOW)

    elif direction == "left":
        # Turn left - use calculated angle if provided
        if angle is not None:
            set_servo_angle(angle)
        else:
            set_servo_angle(50)
        move_forward(SPEED_SLOW)
        last_direction = "left"

    elif direction == "right":
        # Turn right - use calculated angle if provided
        if angle is not None:
            set_servo_angle(angle)
        else:
            set_servo_angle(130)
        move_forward(SPEED_SLOW)
        last_direction = "right"

    elif direction == "forward":
        # Go straight - use calculated angle if provided
        if angle is not None:
            set_servo_angle(angle)
        else:
            set_servo_angle(SERVO_ANGLE_CENTER)
        move_forward(SPEED_SLOW)
        last_direction = "forward"

    elif direction == "non":
        # No line: proceed slowly maintaining previous direction
        if angle is not None:
            set_servo_angle(angle)
        elif last_direction == "left":
            set_servo_angle(50)
        elif last_direction == "right":
            set_servo_angle(130)
        else:
            set_servo_angle(SERVO_ANGLE_CENTER)
        move_forward(SPEED_SLOW)

    else:
        # Unknown direction: stop
        print(f"⚠ Unknown direction: {direction}")
        stop_motor()

def cleanup_drive():
    """Cleanup driving module"""
    global motor_pwm, servo_pwm, initialized

    if not initialized:
        return

    stop_motor()
    set_servo_angle(SERVO_ANGLE_CENTER)

    if motor_pwm is not None:
        motor_pwm.stop()
    if servo_pwm is not None:
        servo_pwm.stop()

    GPIO.cleanup()
    initialized = False
    print("✓ Driving module cleaned up")

