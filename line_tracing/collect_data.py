#!/usr/bin/env python3
"""
라인트레이싱 데이터 수집 스크립트
카메라에서 이미지를 캡처하고 키보드로 라벨링합니다.

사용법:
    python3 collect_data.py

키보드 입력:
    'f' - forward (직진)
    'l' - left (좌회전)
    'r' - right (우회전)
    'g' - green (초록불)
    's' - red (빨간불/정지)
    'n' - noline (라인 없음)
    'q' - 종료
    스페이스바 - 현재 프레임 저장
"""

import os
import time
import cv2
import numpy as np
from picamera2 import Picamera2
from datetime import datetime

# 클래스 정의 (camera_main_gpt.py와 동일한 순서)
CLASSES = {
    'g': 'green',      # 초록불
    'l': 'left',       # 좌회전
    'm': 'middle',     # 중앙/직진
    'n': 'noline',     # 라인 없음
    's': 'red',        # 빨간불/정지
    'r': 'right'       # 우회전
}

# 데이터 저장 경로
DATA_DIR = "data"
IMG_SIZE = 240  # 모델 입력 크기

def setup_directories():
    """클래스별 디렉토리 생성"""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    for class_name in CLASSES.values():
        class_dir = os.path.join(DATA_DIR, class_name)
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)
            print(f"✓ 디렉토리 생성: {class_dir}")

def save_image(frame, class_name):
    """이미지를 저장"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"{class_name}_{timestamp}.jpg"
    filepath = os.path.join(DATA_DIR, class_name, filename)

    # 리사이즈하여 저장 (모델 입력 크기에 맞춤)
    resized = cv2.resize(frame, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    cv2.imwrite(filepath, resized)
    return filepath

def main():
    print("=" * 60)
    print("라인트레이싱 데이터 수집 도구")
    print("=" * 60)
    print("\n키보드 입력:")
    print("  'g' - green (초록불)")
    print("  'l' - left (좌회전)")
    print("  'm' - middle (중앙/직진)")
    print("  'n' - noline (라인 없음)")
    print("  's' - red (빨간불/정지)")
    print("  'r' - right (우회전)")
    print("  스페이스바 - 현재 프레임 저장")
    print("  'q' - 종료")
    print("=" * 60)

    # 디렉토리 설정
    setup_directories()

    # 카메라 초기화
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"format": "RGB888", "size": (640, 480)}
    )
    picam2.configure(config)
    picam2.start()
    time.sleep(1)  # 카메라 워밍업

    print("\n카메라 시작 완료. 데이터 수집을 시작합니다...\n")

    # 통계
    counts = {class_name: 0 for class_name in CLASSES.values()}
    current_class = None
    last_key = None

    try:
        while True:
            # 프레임 캡처
            frame_rgb = picam2.capture_array()
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            # 화면에 정보 표시
            display_frame = frame_bgr.copy()

            # 현재 선택된 클래스 표시
            if current_class:
                text = f"Class: {current_class.upper()}"
                color = (0, 255, 0) if current_class in ['forward', 'green'] else (0, 0, 255) if current_class == 'red' else (255, 255, 0)
                cv2.putText(display_frame, text, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)

            # 통계 표시
            stats_text = " | ".join([f"{k}: {counts[k]}" for k in CLASSES.values()])
            cv2.putText(display_frame, stats_text, (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            # 안내 텍스트
            cv2.putText(display_frame, "Press key to select class, SPACE to save",
                       (10, display_frame.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

            cv2.imshow("Data Collection - Press key to select class, SPACE to save", display_frame)

            # 키 입력 처리
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                print("\n종료합니다...")
                break
            elif key == ord(' '):  # 스페이스바
                if current_class:
                    filepath = save_image(frame_bgr, current_class)
                    counts[current_class] += 1
                    print(f"✓ 저장: {filepath} ({current_class}) - 총 {counts[current_class]}개")
                else:
                    print("⚠ 클래스를 먼저 선택하세요 (f/l/r/g/s/n)")
            elif chr(key).lower() in CLASSES:
                current_class = CLASSES[chr(key).lower()]
                print(f"선택된 클래스: {current_class}")

            time.sleep(0.03)  # 약간의 지연

    except KeyboardInterrupt:
        print("\n키보드 인터럽트로 종료합니다...")
    finally:
        picam2.stop()
        cv2.destroyAllWindows()

        # 최종 통계 출력
        print("\n" + "=" * 60)
        print("수집 완료 통계:")
        print("=" * 60)
        total = 0
        for class_name, count in counts.items():
            print(f"  {class_name:10s}: {count:4d}개")
            total += count
        print(f"  {'Total':10s}: {total:4d}개")
        print("=" * 60)

if __name__ == "__main__":
    main()

