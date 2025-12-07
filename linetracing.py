#!/usr/bin/env python3
# Main file that combines linetracing_drive.py and linetracing_Judgment.py
# Combines CV and ML judgments and performs driving.

import time
import argparse
import os
import cv2
import numpy as np
from picamera2 import Picamera2
from PIL import Image

# Module imports
import linetracing_cv
import linetracing_ml
import linetracing_Judgment
import linetracing_drive

def main():
    """Hybrid line tracing main loop"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Hybrid Line Tracing (CV + ML)')
    parser.add_argument('-testcase', type=str, default=None,
                        help='Test case name for saving captured images (e.g., -testcase test1)')
    args = parser.parse_args()

    # Setup image capture if testcase is provided
    capture_enabled = args.testcase is not None
    frame_counter = 0
    image_counter = 0
    CAPTURE_INTERVAL = 20  # Capture every 20 frames

    if capture_enabled:
        # Ensure line_log directory exists
        log_dir = "line_log"
        os.makedirs(log_dir, exist_ok=True)
        print(f"📸 Image capture enabled: {log_dir}/{args.testcase}_<NUM>.jpg")

    print("=" * 60)
    print("Hybrid Line Tracing (CV + ML)")
    # [수정 1] 가중치 관련 출력 삭제 (더 이상 사용하지 않음)
    print("Priority Mode: ML(Traffic Light) -> CV(Driving)")
    print("=" * 60)

    # Initialize modules
    print("\nInitializing modules...")
    linetracing_cv.init_cv()

    if not linetracing_ml.init_ml():
        print("✗ ML model loading failed. Using CV only without ML.")
        use_ml = False
    else:
        use_ml = True
        print("✓ ML model loaded successfully")
        # Check if model is actually available
        # [수정 2] linetracing_ml 구조 변경에 맞춰 체크 로직 단순화
        if linetracing_ml.interpreter is None:
            print("⚠ Warning: ML interpreter is None!")
            use_ml = False

    linetracing_drive.init_drive()
    print("✓ All modules initialized\n")

    # Initialize camera
    print("Initializing camera...")
    picam2 = Picamera2()

    # [수정 3] 카메라 설정 문법 에러 수정 (main=... 사용)
    config = picam2.create_preview_configuration(
        main={"size": (640, 480), "format": "RGB888"}
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

            # CV judgment (with debug info if capturing)
            if capture_enabled and frame_counter == CAPTURE_INTERVAL - 1:
                cv_result, cv_debug = linetracing_cv.judge_cv(frame_rgb, return_debug=True)
            else:
                cv_result = linetracing_cv.judge_cv(frame_rgb)
                cv_debug = None

            # ML judgment
            ml_result = None
            if use_ml:
                try:
                    ml_result = linetracing_ml.judge_ml(frame_rgb)
                except Exception as e:
                    if frame_counter == 0:  # Only print error once per capture interval
                        print(f"⚠ ML judgment error: {e}")
                    ml_result = None

            # Combine judgments
            final_judgment = linetracing_Judgment.combine_judgments(cv_result, ml_result)

            # Capture image every 20 frames if testcase is provided
            if capture_enabled:
                frame_counter += 1
                if frame_counter >= CAPTURE_INTERVAL:
                    frame_counter = 0
                    image_counter += 1

                    image = Image.fromarray(frame_rgb)
                    filename = f"line_log/{args.testcase}_{image_counter:04d}.jpg"
                    image.save(filename)

                    # Save debug image with CV processing results
                    if cv_debug:
                        debug_filename = f"line_log/{args.testcase}_{image_counter:04d}_debug.jpg"
                        debug_img = create_debug_image(frame_rgb, cv_debug, cv_result, ml_result, final_judgment)
                        debug_image_pil = Image.fromarray(debug_img)
                        debug_image_pil.save(debug_filename)
                        print(f"📸 Captured: {filename} + {debug_filename} (CV: {cv_result}, ML: {ml_result}, Final: {final_judgment})")
                    else:
                        print(f"📸 Captured: {filename} (CV: {cv_result}, ML: {ml_result}, Final: {final_judgment})")

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

def create_debug_image(frame_rgb, cv_debug, cv_result, ml_result, final_judgment):
    """Create debug image showing CV processing results"""
    # frame_rgb는 RGB 형식이어야 함 (Picamera2는 RGB 반환)
    # OpenCV 표시를 위해 BGR로 변환
    frame_bgr = cv2.cvtColor(frame_rgb.copy(), cv2.COLOR_RGB2BGR)
    h_orig, w_orig = frame_bgr.shape[:2]

    # Resize to match CV processing size
    img_resized = cv2.resize(frame_bgr, (linetracing_cv.IMG_WIDTH, linetracing_cv.IMG_HEIGHT))
    h, w = img_resized.shape[:2]

    # Draw ROI rectangle (on resized image)
    roi_top = int(h * linetracing_cv.ROI_TOP)
    roi_bottom = h
    cv2.rectangle(img_resized, (0, roi_top), (w, roi_bottom), (0, 255, 0), 2)

    # Draw line centers if detected
    if cv_debug.get('bottom_center') is not None:
        bottom_y = int(roi_top + (roi_bottom - roi_top) * 0.8)
        cv2.circle(img_resized, (int(cv_debug['bottom_center']), bottom_y), 5, (255, 0, 0), -1)
        cv2.line(img_resized, (w // 2, bottom_y),
                (int(cv_debug['bottom_center']), bottom_y), (0, 0, 255), 2)

    if cv_debug.get('top_center') is not None:
        top_y = int(roi_top + (roi_bottom - roi_top) * 0.2)
        cv2.circle(img_resized, (int(cv_debug['top_center']), top_y), 5, (0, 255, 255), -1)

    # Draw binary image on the side
    if cv_debug.get('binary') is not None:
        binary = cv_debug['binary']
        # Convert binary to 3-channel for display
        binary_colored = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        # Resize binary to fit on the side
        binary_h = int(h * 0.3)
        binary_w = int(binary_colored.shape[1] * binary_h / binary_colored.shape[0])
        binary_resized = cv2.resize(binary_colored, (binary_w, binary_h))
        # Place binary image on the right side
        img_resized[0:binary_h, w-binary_w:w] = binary_resized[:, :min(binary_w, w)]

    # Add text information
    info_y = 20
    cv2.putText(img_resized, f"CV: {cv_result}", (10, info_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(img_resized, f"ML: {ml_result}", (10, info_y + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(img_resized, f"Final: {final_judgment}", (10, info_y + 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    if cv_debug.get('bottom_center') is not None:
        cv2.putText(img_resized, f"Bottom: {cv_debug['bottom_center']:.1f}", (10, info_y + 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    else:
        cv2.putText(img_resized, "Bottom: None", (10, info_y + 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    if cv_debug.get('line_angle') is not None:
        cv2.putText(img_resized, f"Angle: {cv_debug['line_angle']:.1f}deg", (10, info_y + 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # Convert back to RGB
    debug_img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    return debug_img_rgb

if __name__ == "__main__":
    main()
