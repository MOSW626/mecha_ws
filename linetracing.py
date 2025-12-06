#!/usr/bin/env python3
# Main file that combines linetracing_drive.py and linetracing_Judgment.py
# Combines CV and ML judgments and performs driving.

import time
from picamera2 import Picamera2

# Module imports
import linetracing_cv
import linetracing_ml
import linetracing_Judgment
import linetracing_drive

def main():
    """Hybrid line tracing main loop"""
    print("=" * 60)
    print("Hybrid Line Tracing (CV + ML)")
    print(f"CV weight: {linetracing_Judgment.CV_WEIGHT}, ML weight: {linetracing_Judgment.ML_WEIGHT}")
    print("=" * 60)

    # Initialize modules
    print("\nInitializing modules...")
    linetracing_cv.init_cv()

    if not linetracing_ml.init_ml():
        print("✗ ML model loading failed. Using CV only without ML.")
        use_ml = False
    else:
        use_ml = True

    linetracing_drive.init_drive()
    print("✓ All modules initialized\n")

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

    # Initial settings
    linetracing_drive.set_servo_angle(90)
    time.sleep(0.1)

    print("Line tracing started!\n")

    # Waiting for green light state
    waiting_for_green = False

    # Track consecutive "non" states for backward movement
    non_count = 0
    MAX_NON_COUNT = 10  # Number of consecutive "non" before backing up
    BACKUP_SPEED = 5  # Speed for backing up
    BACKUP_DURATION = 0.6  # Duration to backup (seconds)

    try:
        while True:
            # Capture frame
            frame_rgb = picam2.capture_array()

            # CV judgment
            cv_result = linetracing_cv.judge_cv(frame_rgb)

            # ML judgment
            ml_result = None
            if use_ml:
                ml_result = linetracing_ml.judge_ml(frame_rgb)

            # Combine judgments
            final_judgment = linetracing_Judgment.combine_judgments(cv_result, ml_result)

            # Handle red light
            if final_judgment == "red" and not waiting_for_green:
                print("🔴 Red light detected - stopping")
                linetracing_drive.drive("red")
                waiting_for_green = True
                non_count = 0  # Reset non count

            # Waiting for green light
            if waiting_for_green:
                if final_judgment == "green":
                    print("🟢 Green light detected - resuming")
                    time.sleep(0.5)
                    waiting_for_green = False
                    non_count = 0  # Reset non count
                else:
                    # Continue waiting
                    time.sleep(0.1)
                    continue

            # Driving control
            if not waiting_for_green:
                # Handle "non" state - backup if line lost for too long
                if final_judgment == "non":
                    non_count += 1
                    if non_count >= MAX_NON_COUNT:
                        print(f"⚠ Line lost for {non_count} frames - backing up to find line")
                        # Stop first
                        linetracing_drive.stop_motor()
                        time.sleep(0.1)
                        # Backup
                        linetracing_drive.move_backward(BACKUP_SPEED)
                        time.sleep(BACKUP_DURATION)
                        # Stop after backup
                        linetracing_drive.stop_motor()
                        time.sleep(0.1)
                        # Try rotating slightly left and right to find line
                        print("  Searching for line by rotating...")
                        for angle in [50, 130, 90]:  # Left, right, center
                            linetracing_drive.set_servo_angle(angle)
                            time.sleep(0.2)
                            # Check if line found
                            test_frame = picam2.capture_array()
                            test_cv = linetracing_cv.judge_cv(test_frame)
                            if test_cv != "non":
                                print(f"  ✓ Line found at angle {angle}")
                                break
                        non_count = 0  # Reset counter after backup
                        print("✓ Backup complete - resuming line search")
                    else:
                        # Continue with normal non handling - slow down more
                        linetracing_drive.drive(final_judgment)
                else:
                    # Line found - reset non count
                    if non_count > 0:
                        print(f"✓ Line found after {non_count} non frames")
                    non_count = 0
                    linetracing_drive.drive(final_judgment)

                # Debug output
                cv_info = f"CV: {cv_result}" if cv_result else "CV: None"
                ml_info = f"ML: {ml_result}" if ml_result else "ML: None"
                non_info = f" | Non: {non_count}/{MAX_NON_COUNT}" if final_judgment == "non" else ""
                print(f"Direction: {final_judgment} | {cv_info} | {ml_info}{non_info}")

            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\nExiting due to keyboard interrupt...")
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        linetracing_drive.cleanup_drive()
        picam2.stop()
        print("System shutdown complete")

if __name__ == "__main__":
    main()
