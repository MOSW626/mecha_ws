import RPi.GPIO as GPIO
import time
import sys
import tty
import termios
#--------------#
# Motor Setups #
#--------------#
DIR_PIN = 16 # DON'T CHANGE! GPIO Pin number of main motor rotation direction (HIGH: FORWARD, LOW: BACKWARD)
PWM_PIN = 12 # DON'T CHANGE! GPIO Pin number of main motor speed control using PWM.
SERVO_PIN = 13 # DON'T CHANGE! GPIO Pin number of servo motor angle control using PWM.
MOTOR_FREQ = 1000 # DON'T CHANGE! Main motor PWM frequency: 1000 Hz
MOTOR_SPEED = 30 # Main motor Speed: 0 to 100 allowed.
SERVO_FREQ = 50 # DON'T CHANGE! Servo motor PWM frequency: 50 Hz
SERVO_MAX_DUTY = 12 # DON'T CHANGE! Max Servo PWM DutyCycle Input
SERVO_MIN_DUTY = 3 # DON'T CHANGE! Min Servo PWM DutyCycle Input

# DON'T CHANGE! GPIO Pin Setups
GPIO.setmode(GPIO.BCM)
GPIO.setup(DIR_PIN, GPIO.OUT)
GPIO.setup(PWM_PIN, GPIO.OUT)
GPIO.setup(SERVO_PIN, GPIO.OUT)

# Set PWM controller using PWM library in GPIO
motor_pwm = GPIO.PWM(PWM_PIN, MOTOR_FREQ)
servo_pwm = GPIO.PWM(SERVO_PIN, SERVO_FREQ)

# Initialize PWMs
motor_pwm.start(0)
servo_pwm.start(0)

# Function: Servo Motor Position(angle) control
# Input: angle(unit: degree) -> Output: servo PWM dutycycle between 3 to 12
def set_servo_angle(degree):
    if degree > 180:
        degree = 180
    elif degree < 0:
        degree = 0
    duty = SERVO_MIN_DUTY + (degree * (SERVO_MAX_DUTY - SERVO_MIN_DUTY) / 180.0)
    servo_pwm.ChangeDutyCycle(duty)
    print(f" servo_angle: {degree}, Duty: {round(duty, 2)}")
    time.sleep(0.5)
    servo_pwm.ChangeDutyCycle(0)
    # Function: Main Motor Forward Speed control
    # Input: x -> Output: DIR_Pin HIGH & PWM Dutycycle which we defined as "MOTOR_SPEED"
    
def move_forward():
    GPIO.output(DIR_PIN, GPIO.HIGH)
    motor_pwm.ChangeDutyCycle(MOTOR_SPEED)
    print(" motor forward")
    # Function: Main Motor Backward Speed control
    # Input: x -> Output: DIR_Pin LOW & PWM Dutycycle which we defined as"MOTOR_SPEED"

def move_backward():
    GPIO.output(DIR_PIN, GPIO.LOW)
    motor_pwm.ChangeDutyCycle(MOTOR_SPEED)
    print("motor backward")
    # Function: Main Motor Stop
    # Input: x -> Output: PWM Dutycycle as 0
def stop_motor():
    motor_pwm.ChangeDutyCycle(0)
    print(" motor stop")
    # Function: Get Keyboard Input
    # Input: x -> Output: a character we selected in keyboard
def get_key():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch

# Main
def main():
    try:
        while True:
            key = get_key()
            if key == 'r':
                set_servo_angle(135)
                move_forward()
            elif key == 'l':
                set_servo_angle(45)
                move_forward()
            elif key == 'f':
                set_servo_angle(90)
                move_forward()
            elif key == 'b':
                set_servo_angle(90)
                move_backward()
            elif key == 's':
                stop_motor()
            elif key == 'q':
                print(" quit.")
                break
            else:
                print(f"unknown: {key}")
    finally:
        motor_pwm.stop()
        servo_pwm.stop()
        GPIO.cleanup()
    
if __name__ == "__main__":
    main()