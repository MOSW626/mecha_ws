#!/usr/bin/env python3
# Run as follows:
# python3 main.py -l : Line tracing only
# python3 main.py -d : Driving only
# python3 main.py : Both modes
# When using both, switch from line tracing to driving mode at green light.
# When green, drive straight at max speed (100) for about 1.3 seconds while switching.
# Uses threads naturally. (Raspberry Pi 4B)

import argparse
import threading
import time
import sys
import os
from picamera2 import Picamera2
import RPi.GPIO as GPIO

# Import other files as modules
try:
    import linetracing_cv
    import linetracing_ml
    import driving
except ImportError as e:
    print(f"Module import error: {e}")
    sys.exit(1)

try:
    import tflite_runtime.interpreter as tflite
    USE_TFLITE = True
except ImportError:
    USE_TFLITE = False

# ==================== GPIO Settings ====================
DIR_PIN = 16
PWM_PIN = 12
SERVO_PIN = 13
TRIG_LEFT = 17
ECHO_LEFT = 27
TRIG_RIGHT = 5
ECHO_RIGHT = 6

MOTOR_FREQ = 1000
SERVO_FREQ = 50
SERVO_MAX_DUTY = 12
SERVO_MIN_DUTY = 3

GPIO.setmode(GPIO.BCM)
GPIO.setup([DIR_PIN, PWM_PIN, SERVO_PIN], GPIO.OUT)
GPIO.setup([TRIG_LEFT, TRIG_RIGHT], GPIO.OUT)
GPIO.setup([ECHO_LEFT, ECHO_RIGHT], GPIO.IN)

motor_pwm = GPIO.PWM(PWM_PIN, MOTOR_FREQ)
servo_pwm = GPIO.PWM(SERVO_PIN, SERVO_FREQ)
motor_pwm.start(0)
servo_pwm.start(0)

# Global variables
mode_lock = threading.Lock()
current_mode = "linetracing"  # "linetracing" or "driving"
should_stop = False

# ==================== Motor Control Functions ====================
def set_servo_angle(degree):
    """Set servo motor angle"""
    degree = max(45, min(135, degree))
    duty = SERVO_MIN_DUTY + (degree * (SERVO_MAX_DUTY - SERVO_MIN_DUTY) / 180.0)
    servo_pwm.ChangeDutyCycle(duty)

def move_forward(speed):
    """Move forward"""
    GPIO.output(DIR_PIN, GPIO.HIGH)
    motor_pwm.ChangeDutyCycle(speed)

def stop_motor():
    """Stop"""
    motor_pwm.ChangeDutyCycle(0)

# ==================== Mode Switching ====================
def switch_to_driving():
    """Switch from line tracing to driving mode"""
    global current_mode

    print("🟢 Green light detected - switching to driving mode")

    # Drive straight at max speed (100) for 1.3 seconds
    set_servo_angle(90)  # Straight
    move_forward(100)  # Max speed
    time.sleep(1.3)

    with mode_lock:
        current_mode = "driving"

    print("Switched to driving mode")

# ==================== Line Tracing Thread ====================
def line_tracing_thread(picam2):
    """Line tracing thread - uses linetracing_cv logic"""
    global current_mode, should_stop

    # Use linetracing_cv functions
    img_center = linetracing_cv.IMG_WIDTH / 2
    lost_line_count = 0
    max_lost_count = 10

    # Load ML model (optional)
    interpreter = None
    inp = None
    out = None
    if USE_TFLITE and os.path.exists("./cnn.tflite"):
        try:
            interpreter = tflite.Interpreter(model_path="./cnn.tflite")
            interpreter.allocate_tensors()
            inp = interpreter.get_input_details()[0]
            out = interpreter.get_output_details()[0]
        except:
            interpreter = None

    # Initial settings
    set_servo_angle(linetracing_cv.SERVO_ANGLE_CENTER)
    time.sleep(0.1)

    try:
        while not should_stop:
            with mode_lock:
                if current_mode != "linetracing":
                    time.sleep(0.1)
                    continue

            # Capture frame
            frame_rgb = picam2.capture_array()

            # Detect traffic light
            traffic_light = linetracing_cv.detect_traffic_light(frame_rgb)

            # Also check with ML
            if interpreter:
                try:
                    # Use linetracing_ml preprocessing function
                    img = linetracing_ml.preprocess_frame(frame_rgb)
                    interpreter.set_tensor(inp["index"], img)
                    interpreter.invoke()
                    probs = interpreter.get_tensor(out["index"])[0]
                    pred_id = int(np.argmax(probs))
                    pred_label = linetracing_ml.LABELS[pred_id]
                    confidence = probs[pred_id]
                    if pred_label in ['red', 'green'] and confidence > 0.7:
                        traffic_light = pred_label
                except:
                    pass

            if traffic_light == 'red':
                print("🔴 Red light detected - stopping")
                stop_motor()
                set_servo_angle(linetracing_cv.SERVO_ANGLE_CENTER)
                while True:
                    frame_rgb = picam2.capture_array()
                    traffic_light = linetracing_cv.detect_traffic_light(frame_rgb)
                    if interpreter:
                        try:
                            img = linetracing_ml.preprocess_frame(frame_rgb)
                            interpreter.set_tensor(inp["index"], img)
                            interpreter.invoke()
                            probs = interpreter.get_tensor(out["index"])[0]
                            pred_id = int(np.argmax(probs))
                            pred_label = linetracing_ml.LABELS[pred_id]
                            confidence = probs[pred_id]
                            if pred_label == 'green' and confidence > 0.7:
                                traffic_light = 'green'
                        except:
                            pass
                    if traffic_light == 'green':
                        switch_to_driving()
                        break
                    time.sleep(0.1)
                continue

            # Preprocess image and detect line
            roi, roi_top = linetracing_cv.preprocess_image(frame_rgb)
            binary, top_center, bottom_center, line_angle = linetracing_cv.detect_line_with_angle(roi)

            # Calculate control output
            angle, center_error = linetracing_cv.calculate_control_output(
                bottom_center, line_angle, img_center
            )

            if bottom_center is None:
                lost_line_count += 1
                if lost_line_count > max_lost_count:
                    print("⚠ Cannot find line - stopping")
                    stop_motor()
                else:
                    move_forward(linetracing_cv.SPEED_SLOW)
            else:
                lost_line_count = 0
                if angle is not None:
                    set_servo_angle(angle)
                    move_forward(linetracing_cv.SPEED_SLOW)

            time.sleep(0.01)

    except Exception as e:
        print(f"Line tracing thread error: {e}")
        import traceback
        traceback.print_exc()

# ==================== Driving Thread ====================
def driving_thread():
    """Driving thread - uses driving module logic"""
    global current_mode, should_stop

    # Use driving module ultrasonic sensor functions
    last_left = None
    last_right = None

    # Set initial servo angle
    set_servo_angle(90)
    time.sleep(0.05)

    try:
        while not should_stop:
            with mode_lock:
                if current_mode != "driving":
                    time.sleep(0.1)
                    continue

            # Read ultrasonic sensors
            raw_left = driving.read_stable(driving.TRIG_LEFT, driving.ECHO_LEFT)
            raw_right = driving.read_stable(driving.TRIG_RIGHT, driving.ECHO_RIGHT)

            # Smoothing
            left = driving.smooth(last_left, raw_left)
            right = driving.smooth(last_right, raw_right)
            last_left, last_right = left, right

            if left is None or right is None:
                continue

            # Use driving module control logic
            error = driving.ref_distance_difference - (right - left)
            output = driving.Kp * error

            angle_cmd = driving.base_angle - output
            angle_cmd = max(45.0, min(135.0, angle_cmd))

            speed_cmd = driving.SPEED_ULTRASONIC - driving.speed_angle_diff * abs(output)

            if speed_cmd < 0.0:
                speed_cmd = 0.0

            set_servo_angle(angle_cmd)
            move_forward(speed_cmd)

            time.sleep(0.001)

    except Exception as e:
        print(f"Driving thread error: {e}")
        import traceback
        traceback.print_exc()

# ==================== Main Functions ====================
def line_tracing_only():
    """Run line tracing only"""
    linetracing_cv.main()

def driving_only():
    """Run driving only"""
    driving.driving_mode()

def both_modes():
    """Use both line tracing and driving"""
    global should_stop

    print("=" * 60)
    print("Integrated Mode: Line Tracing + Driving")
    print("=" * 60)

    # Initialize camera
    print("Initializing camera...")
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"format": "RGB888", "size": (640, 480)}
    )
    picam2.configure(config)
    picam2.start()
    time.sleep(1)
    print("✓ Camera initialized\n")

    # Start threads
    lt_thread = threading.Thread(target=line_tracing_thread, args=(picam2,), daemon=True)
    dr_thread = threading.Thread(target=driving_thread, daemon=True)

    lt_thread.start()
    dr_thread.start()

    print("Line tracing thread started")
    print("Driving thread started")
    print("Integrated mode running... (Press Ctrl+C to exit)\n")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")
        should_stop = True
        lt_thread.join(timeout=2)
        dr_thread.join(timeout=2)
    finally:
        stop_motor()
        set_servo_angle(90)
        picam2.stop()
        motor_pwm.stop()
        servo_pwm.stop()
        GPIO.cleanup()
        print("System shutdown complete")

def main():
    parser = argparse.ArgumentParser(description='Autonomous driving system')
    parser.add_argument('-l', '--linetracing', action='store_true', help='Run line tracing only')
    parser.add_argument('-d', '--driving', action='store_true', help='Run driving only')

    args = parser.parse_args()

    if args.linetracing:
        print("Running in line tracing mode.")
        line_tracing_only()
    elif args.driving:
        print("Running in driving mode.")
        driving_only()
    else:
        print("Running in integrated mode.")
        both_modes()

if __name__ == "__main__":
    main()
