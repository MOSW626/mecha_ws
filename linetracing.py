#!/usr/bin/env python3
# linetracing.py
# Modulized version: Run until Green light departure

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

def run_linetracing_sequence():
    """
    Executes the traffic light sequence.
    Returns True when Green light is detected and the car starts moving.
    """
    # ---------------- Log Settings ----------------
    # 메인에서 호출 시 로그가 필요하다면 여기서 설정 (기본값 하드코딩 예시)
    capture_enabled = True
    args_testcase = "switch_test" # 로그 파일 접두사

    frame_counter = 0
    image_counter = 0
    CAPTURE_INTERVAL = 3

    if capture_enabled:
        log_dir = "line_log"
        os.makedirs(log_dir, exist_ok=True)
        print(f"📸 Capture Enabled: {log_dir}")

    # ---------------- Module Init ----------------
    print("\n[Init] Modules...")
    linetracing_cv.init_cv()

    if not linetracing_ml.init_ml():
        print("✗ ML Failed. Using CV only.")
        use_ml = False
    else:
        use_ml = True
        print("✓ ML Loaded.")
        if linetracing_ml.interpreter is None:
            use_ml = False

    linetracing_drive.init_drive()

    # ---------------- Camera Init ----------------
    print("[Init] Camera...")
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"size": (640, 480), "format": "RGB888"}
    )
    picam2.configure(config)
    picam2.start()
    time.sleep(1)

    linetracing_drive.set_servo_angle(90)
    print("\n🏎️  Line Tracing Started! (10s Blind Mode) 🏎️\n")

    traffic_stage = 0
    start_time = time.time()

    non_count = 0
    MAX_NON_COUNT = 10
    BACKUP_SPEED = 5
    BACKUP_DURATION = 0.6

    consecutive_red_count = 0
    consecutive_green_count = 0
    DETECTION_REQUIREMENT = 3

    try:
        while True:
            # 1. Capture
            frame_rgb = picam2.capture_array()

            # 2. CV Process
            is_capture_frame = capture_enabled and (frame_counter >= CAPTURE_INTERVAL - 1)
            if is_capture_frame:
                cv_result, cv_debug = linetracing_cv.judge_cv(frame_rgb, return_debug=True)
            else:
                cv_result = linetracing_cv.judge_cv(frame_rgb)
                cv_debug = None

            # 3. Time Check (Stage 0 -> 1)
            elapsed_time = time.time() - start_time
            if traffic_stage == 0:
                if elapsed_time > 10.0:
                    print(f"\n⏰ 10 Seconds Passed! ML Activated. Searching for RED...\n")
                    traffic_stage = 1

            # 4. ML Process
            ml_result = None
            if use_ml and (traffic_stage == 1 or traffic_stage == 2):
                try:
                    raw_ml = linetracing_ml.judge_ml(frame_rgb)
                    if traffic_stage == 1:
                        ml_result = "red" if raw_ml == "red" else "noline"
                    elif traffic_stage == 2:
                        ml_result = "green" if raw_ml == "green" else "noline"
                except Exception:
                    ml_result = None
            else:
                ml_result = "noline"

            # 5. Final Judgment
            final_judgment = "non"

            # [Stage 0] 10초 무적
            if traffic_stage == 0:
                final_judgment = cv_result if cv_result else "non"

            # [Stage 1] RED 감지
            elif traffic_stage == 1:
                if ml_result == "red":
                    consecutive_red_count += 1
                    if consecutive_red_count >= DETECTION_REQUIREMENT:
                        print("\n🔴 RED Detected! Stopping... -> [Waiting for GREEN]\n")
                        traffic_stage = 2
                        consecutive_red_count = 0
                        final_judgment = "red"
                    else:
                        final_judgment = cv_result if cv_result else "non"
                else:
                    consecutive_red_count = 0
                    final_judgment = cv_result if cv_result else "non"

            # [Stage 2] GREEN 대기
            elif traffic_stage == 2:
                final_judgment = "red" # 정지 유지
                if ml_result == "green":
                    consecutive_green_count += 1
                    if consecutive_green_count >= DETECTION_REQUIREMENT:
                        print("\n🟢 GREEN Detected! GO! -> [Handing over to Low Defense]\n")
                        final_judgment = "green"
                        traffic_stage = 3
                        consecutive_green_count = 0
                else:
                    consecutive_green_count = 0

            # 6. Motor Control & Handover Logic
            if traffic_stage == 2 and final_judgment != "green":
                 linetracing_drive.stop_motor()
                 linetracing_drive.set_servo_angle(90)

            elif final_judgment == "green":
                 # ★ 중요: Green 신호를 받으면 앞으로 살짝 전진 후 루프 탈출
                 print("🚀 Green Start! Moving forward briefly...")
                 linetracing_drive.set_servo_angle(90)
                 linetracing_drive.move_forward(20) # 약간 속도 줌
                 time.sleep(1) # 1초간 직진하여 교차로/라인 통과

                 # 리소스 정리 및 제어권 반환을 위해 break
                 break

            else:
                # 일반 주행 (Stage 0, 1)
                if final_judgment == "non":
                    non_count += 1
                    if non_count >= MAX_NON_COUNT:
                        linetracing_drive.stop_motor()
                        time.sleep(0.1)
                        linetracing_drive.move_backward(BACKUP_SPEED)
                        time.sleep(BACKUP_DURATION)
                        linetracing_drive.stop_motor()
                        non_count = 0
                    else:
                        linetracing_drive.drive(final_judgment)
                else:
                    non_count = 0
                    linetracing_drive.drive(final_judgment)

            # 로그 저장 (옵션)
            if is_capture_frame:
                frame_counter = 0
                image_counter += 1
                image = Image.fromarray(frame_rgb)
                filename = f"line_log/{args_testcase}_S{traffic_stage}_{image_counter:04d}.jpg"
                image.save(filename)

            else:
                if capture_enabled: frame_counter += 1

            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\nStop.")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False
    finally:
        # ★ 매우 중요: 주행 모듈 정리 (GPIO 해제)
        # 이걸 해줘야 다음 low_defense 파일이 GPIO를 다시 잡을 때 충돌 안 남
        linetracing_drive.cleanup_drive()
        picam2.stop()
        print("✓ Linetracing Module Cleanup Complete.")

    return True

if __name__ == "__main__":
    run_linetracing_sequence()
