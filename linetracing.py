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
    parser = argparse.ArgumentParser(description='Hybrid Line Tracing (CV + ML)')
    parser.add_argument('-testcase', type=str, default=None,
                        help='Test case name for saving captured images (e.g., -testcase test1)')
    args = parser.parse_args()

    # [수정 1] 로그 저장 설정
    capture_enabled = args.testcase is not None
    frame_counter = 0
    image_counter = 0

    # ★ 여기를 수정했습니다: 20 -> 3
    # 3프레임마다 저장 (너무 자주 저장하면 렉 걸림. 1로 하면 모든 프레임 저장)
    CAPTURE_INTERVAL = 3

    if capture_enabled:
        log_dir = "line_log"
        os.makedirs(log_dir, exist_ok=True)
        print(f"📸 Image capture enabled: {log_dir}/{args.testcase}_<NUM>.jpg (Interval: {CAPTURE_INTERVAL})")

    print("=" * 60)
    print("Hybrid Line Tracing (CV + ML)")
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
        if linetracing_ml.interpreter is None:
            print("⚠ Warning: ML interpreter is None!")
            use_ml = False

    linetracing_drive.init_drive()
    print("✓ All modules initialized\n")

    # Initialize camera
    print("Initializing camera...")
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"size": (640, 480), "format": "RGB888"}
    )
    picam2.configure(config)
    picam2.start()
    time.sleep(1)
    print("✓ Camera initialized\n")

    linetracing_drive.set_servo_angle(90)
    time.sleep(0.1)

    print("Line tracing started!\n")

    waiting_for_green = False
    non_count = 0
    MAX_NON_COUNT = 10
    BACKUP_SPEED = 5
    BACKUP_DURATION = 0.6

    consecutive_red_count = 0
    consecutive_green_count = 0
    DETECTION_REQUIREMENT = 3

    try:
        while True:
            frame_rgb = picam2.capture_array()

            # [수정 2] 캡처 타이밍이면 무조건 디버그 정보를 요청
            is_capture_frame = capture_enabled and (frame_counter >= CAPTURE_INTERVAL - 1)

            if is_capture_frame:
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
                    if frame_counter == 0:
                        print(f"⚠ ML judgment error: {e}")
                    ml_result = None

            # Combine judgments
            raw_judgment = linetracing_Judgment.combine_judgments(cv_result, ml_result)
            final_judgment = raw_judgment

            # [수정 3] 주행 문제 해결 로직 적용 (CV가 라인을 잘 보고 있으면 ML의 Green 무시)
            if raw_judgment == "green" and cv_result in ["forward", "left", "right"]:
                # ML은 Green이라지만 CV는 주행 중 -> ML 오탐 무시
                final_judgment = cv_result
                consecutive_green_count = 0

            # 필터링 로직
            elif raw_judgment == "red":
                consecutive_red_count += 1
                consecutive_green_count = 0
            elif raw_judgment == "green":
                consecutive_green_count += 1
                consecutive_red_count = 0
            else:
                consecutive_red_count = 0
                consecutive_green_count = 0

            # Red/Green 확정 로직
            if consecutive_red_count >= DETECTION_REQUIREMENT:
                final_judgment = "red"
            elif consecutive_green_count >= DETECTION_REQUIREMENT:
                final_judgment = "green"
            else:
                # 카운트 부족 시 CV 결과 따름 (Red/Green 오탐 방지)
                if raw_judgment in ["red", "green"]:
                    final_judgment = cv_result if cv_result else "non"
                else:
                    final_judgment = raw_judgment

            # [수정 4] 저장 로직: 조건 없이 Interval 되면 무조건 저장
            if is_capture_frame:
                frame_counter = 0 # 카운터 리셋
                image_counter += 1

                # 원본 저장
                image = Image.fromarray(frame_rgb)
                filename = f"line_log/{args.testcase}_{image_counter:04d}.jpg"
                image.save(filename)

                # 디버그 이미지 저장 (cv_debug가 있으면 무조건)
                if cv_debug:
                    debug_filename = f"line_log/{args.testcase}_{image_counter:04d}_debug.jpg"
                    debug_img = create_debug_image(frame_rgb, cv_debug, cv_result, ml_result, final_judgment)
                    debug_image_pil = Image.fromarray(debug_img)
                    debug_image_pil.save(debug_filename)
                    print(f"📸 Captured: {filename} (Final: {final_judgment})")
                else:
                    # cv_debug가 없더라도 원본은 저장됨
                    print(f"📸 Captured: {filename} (No Debug Info)")

            else:
                # 캡처 안 하는 프레임은 카운트만 증가
                if capture_enabled:
                    frame_counter += 1

            # --- 주행 로직 ---
            # Handle red light
            if final_judgment == "red" and not waiting_for_green:
                print("🔴 Red light detected - stopping")
                linetracing_drive.drive("red")
                waiting_for_green = True
                non_count = 0

            # Waiting for green light
            if waiting_for_green:
                if final_judgment == "green":
                    print("🟢 Green light detected - resuming")
                    time.sleep(0.5)
                    waiting_for_green = False
                    non_count = 0
                else:
                    time.sleep(0.1)
                    continue

            # Driving control
            if not waiting_for_green:
                if final_judgment == "non":
                    non_count += 1
                    if non_count >= MAX_NON_COUNT:
                        print(f"⚠ Line lost for {non_count} frames - backing up")
                        linetracing_drive.stop_motor()
                        time.sleep(0.1)
                        linetracing_drive.move_backward(BACKUP_SPEED)
                        time.sleep(BACKUP_DURATION)
                        linetracing_drive.stop_motor()
                        time.sleep(0.1)
                        print("  Searching for line by rotating...")
                        for angle in [50, 130, 90]:
                            linetracing_drive.set_servo_angle(angle)
                            time.sleep(0.2)
                            test_frame = picam2.capture_array()
                            test_cv = linetracing_cv.judge_cv(test_frame)
                            if test_cv != "non":
                                break
                        non_count = 0
                    else:
                        linetracing_drive.drive(final_judgment)
                else:
                    if non_count > 0:
                        print(f"✓ Line found after {non_count} non frames")
                    non_count = 0
                    linetracing_drive.drive(final_judgment)

                # 터미널 출력 최소화 (로그 확인용)
                # print(f"Direction: {final_judgment}")

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
    # 기존 create_debug_image 함수와 동일
    frame_bgr = cv2.cvtColor(frame_rgb.copy(), cv2.COLOR_RGB2BGR)
    h_orig, w_orig = frame_bgr.shape[:2]

    img_resized = cv2.resize(frame_bgr, (linetracing_cv.IMG_WIDTH, linetracing_cv.IMG_HEIGHT))
    h, w = img_resized.shape[:2]

    roi_top = int(h * linetracing_cv.ROI_TOP)
    roi_bottom = h
    cv2.rectangle(img_resized, (0, roi_top), (w, roi_bottom), (0, 255, 0), 2)

    if cv_debug.get('bottom_center') is not None:
        bottom_y = int(roi_top + (roi_bottom - roi_top) * 0.8)
        cv2.circle(img_resized, (int(cv_debug['bottom_center']), bottom_y), 5, (255, 0, 0), -1)
        cv2.line(img_resized, (w // 2, bottom_y),
                (int(cv_debug['bottom_center']), bottom_y), (0, 0, 255), 2)

    if cv_debug.get('top_center') is not None:
        top_y = int(roi_top + (roi_bottom - roi_top) * 0.2)
        cv2.circle(img_resized, (int(cv_debug['top_center']), top_y), 5, (0, 255, 255), -1)

    if cv_debug.get('binary') is not None:
        binary = cv_debug['binary']
        binary_colored = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        binary_h = int(h * 0.3)
        binary_w = int(binary_colored.shape[1] * binary_h / binary_colored.shape[0])
        binary_resized = cv2.resize(binary_colored, (binary_w, binary_h))
        img_resized[0:binary_h, w-binary_w:w] = binary_resized[:, :min(binary_w, w)]

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
        cv2.putText(img_resized, "Bottom: None", (10, info_y +
