#!/usr/bin/env python3
# linetracing.py
# Logic: ML First -> If ML says "cv", run CV logic. -> If Green, Escape immediately inside loop.

import time
import os
import cv2
import numpy as np
from picamera2 import Picamera2
from PIL import Image

import linetracing_cv
import linetracing_ml
import linetracing_drive

def combine_ml2_cv_weighted(frame_rgb):
    """
    ML2와 CV를 가중치로 결합하여 최종 방향을 결정합니다.
    가중치: ML2(0.2) + CV(0.8)

    Returns: "forward", "left", "right", "non" 중 하나
    """
    # 방향을 숫자로 매핑
    direction_map = {
        "forward": 0,
        "left": 1,
        "right": 2,
        "non": 3,
        "noline": 3
    }

    # 역매핑 (숫자 -> 방향)
    reverse_map = {0: "forward", 1: "left", 2: "right", 3: "non"}

    # ML2 결과 가져오기
    ml2_result = linetracing_ml.judge_ml2(frame_rgb)

    # CV 결과 가져오기
    cv_result = linetracing_cv.judge_cv(frame_rgb)

    # 가중치 벡터 초기화 (forward, left, right, non)
    weighted_scores = np.zeros(4)

    # ML2 가중치 적용 (0.2)
    if ml2_result:
        ml2_lower = ml2_result.lower()
        # red/green은 제외하고 방향만 고려
        if ml2_lower in direction_map:
            ml2_idx = direction_map[ml2_lower]
            weighted_scores[ml2_idx] += 0.2

    # CV 가중치 적용 (0.8)
    if cv_result:
        cv_lower = cv_result.lower()
        if cv_lower in direction_map:
            cv_idx = direction_map[cv_lower]
            weighted_scores[cv_idx] += 0.8

    # 가장 높은 점수의 방향 선택
    if np.sum(weighted_scores) > 0:
        final_idx = int(np.argmax(weighted_scores))
        return reverse_map[final_idx]
    else:
        # 둘 다 결과가 없으면 "non" 반환
        return "non"

def run_linetracing_sequence():
    capture_enabled = False
    args_testcase = "log_ml_fix"

    frame_counter = 0
    image_counter = 0
    CAPTURE_INTERVAL = 3

    if capture_enabled:
        log_dir = "../logs/line_log"
        os.makedirs(log_dir, exist_ok=True)

    print("\n[Init] Modules...")
    linetracing_cv.init_cv()
    if not linetracing_ml.init_ml():
        print("✗ ML Failed.")
        return False
    linetracing_drive.init_drive()

    print("[Init] Camera...")
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"size": (640, 480), "format": "RGB888"})
    picam2.configure(config)
    picam2.start()
    time.sleep(1)

    linetracing_drive.set_servo_angle(90)
    print("\n🏎️  Line Tracing (Red/Green/CV Mode) Started! 🏎️\n")

    traffic_stage = 0
    start_time = time.time()

    consecutive_red_count = 0
    consecutive_green_count = 0
    DETECTION_REQUIREMENT = 3

    # 성공적으로 Green을 통과했는지 확인하는 플래그
    success_finish = False

    try:
        while True:
            frame_rgb = picam2.capture_array()
            is_capture_frame = capture_enabled and (frame_counter >= CAPTURE_INTERVAL - 1)

            # [Time Check]
            elapsed_time = time.time() - start_time
            if traffic_stage == 0:
                if elapsed_time > 10.0:
                    print(f"\n⏰ 10 Seconds Passed! Searching for RED...\n")
                    traffic_stage = 1

            # [ML Inference]
            ml_label = "noline"
            try:
                raw_ml = linetracing_ml.judge_ml(frame_rgb)
                if raw_ml is not None:
                    ml_label = raw_ml
            except:
                ml_label = "error"

            # [Logic & Control]
            cv_result = "OFF"
            final_action = "non"

            # === Stage 0: 10초 무적 ===
            if traffic_stage == 0:
                cv_result = linetracing_cv.judge_cv(frame_rgb)
                final_action = cv_result if cv_result else "forward"

            # === Stage 1: RED 감시 ===
            elif traffic_stage == 1:
                if ml_label == "red":
                    consecutive_red_count += 1
                    final_action = "red"
                    if consecutive_red_count >= DETECTION_REQUIREMENT:
                        print("\n🔴 RED Confirmed! Stopping... -> [Waiting for GREEN]\n")
                        traffic_stage = 2
                        consecutive_red_count = 0
                        linetracing_drive.stop_motor()

                elif ml_label == "cv":
                    consecutive_red_count = 0
                    # ML2와 CV를 가중치로 결합 (ML2: 0.2, CV: 0.8)
                    final_action = combine_ml2_cv_weighted(frame_rgb)

                else:
                    consecutive_red_count = 0
                    final_action = "non"

            # === Stage 2: GREEN 대기 ===
            elif traffic_stage == 2:
                final_action = "red" # 기본: 정지

                if ml_label == "green":
                    consecutive_green_count += 1
                    print(f"\rWaiting Green... {consecutive_green_count}/{DETECTION_REQUIREMENT}", end="")

                    if consecutive_green_count >= DETECTION_REQUIREMENT:
                        print("\n\n🟢 GREEN Confirmed! GO! -> Executing Escape Move...\n")

                        # ★ [수정] 탈출 주행을 루프 안에서 즉시 실행 (Cleanup 되기 전에!)
                        print("🚀 Escape Move: Driving Forward Blindly for 1.5 sec...")
                        linetracing_drive.set_servo_angle(90)
                        linetracing_drive.move_forward(100) # 속도 약간 증가 (20 -> 25)
                        time.sleep(0.7) # 1.5초간 직진

                        linetracing_drive.stop_motor()
                        success_finish = True # 성공 플래그 세팅
                        break # 루프 종료 -> finally로 이동
                else:
                    consecutive_green_count = 0

            # [Actuate]
            # 탈출 주행 중에는 아래 로직을 타지 않도록 함
            if not success_finish:
                if final_action == "red":
                    linetracing_drive.stop_motor()
                    linetracing_drive.set_servo_angle(90)
                elif final_action == "non" or final_action == "OFF":
                    linetracing_drive.stop_motor()
                else:
                    linetracing_drive.drive(final_action)

            # [Log]
            if not success_finish:
                stage_str = ["Blind", "FindRED", "WaitGRN"][traffic_stage]
                print(f"Stage: {stage_str} | ML: {ml_label:7s} | CV: {cv_result:7s} | Act: {final_action}")

            if is_capture_frame:
                frame_counter = 0
                image_counter += 1
                img = Image.fromarray(frame_rgb)
                img.save(f"../logs/line_log/{args_testcase}_S{traffic_stage}_{image_counter:04d}.jpg")
            else:
                if capture_enabled: frame_counter += 1

            time.sleep(0.01)

    except KeyboardInterrupt:
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False
    finally:
        # ★ 여기서 모든 자원을 해제합니다.
        # 탈출 주행은 이미 위(while 루프 안)에서 끝났으므로 안전하게 닫기만 하면 됩니다.
        linetracing_drive.cleanup_drive()
        picam2.stop()
        print("✓ Linetracing Cleanup Done.")

    # 성공적으로 탈출했으면 True 반환하여 main.py가 다음 단계로 넘어가게 함
    if success_finish:
        print("✓ Handing over to Low Defense.")
        return True

    return False

if __name__ == "__main__":
    run_linetracing_sequence()
