#!/usr/bin/env python3
"""
컴퓨터 비전 기반 라인트레이싱 (ML 없이)
카메라 이미지를 처리하여 흰색 라인을 추적합니다.

사용법:
    python3 line_tracing_cv.py
"""

import cv2
import numpy as np
import time
from picamera2 import Picamera2
import RPi.GPIO as GPIO

# ==================== GPIO 설정 ====================
DIR_PIN = 16
PWM_PIN = 12
SERVO_PIN = 13

MOTOR_FREQ = 1000
SERVO_FREQ = 50
SERVO_MAX_DUTY = 12
SERVO_MIN_DUTY = 3

# 속도 설정
SPEED_NORMAL = 50
SPEED_SLOW = 40
SERVO_ANGLE_CENTER = 90
SERVO_ANGLE_MAX = 135
SERVO_ANGLE_MIN = 45

# ==================== 이미지 처리 설정 ====================
IMG_WIDTH = 320  # 처리할 이미지 너비 (작게 하면 더 빠름)
IMG_HEIGHT = 240  # 처리할 이미지 높이
ROI_TOP = 0.4  # ROI 시작 위치 (상단 40% 제외)
ROI_BOTTOM = 1.0  # ROI 끝 위치

# 라인 검출 설정
WHITE_THRESHOLD = 200  # 흰색 임계값 (0-255)
MIN_LINE_WIDTH = 2  # 최소 라인 폭 (픽셀)
MAX_LINE_WIDTH = 20  # 최대 라인 폭 (픽셀)

# PID 제어 설정
Kp = 0.8  # 비례 게인
Ki = 0.0  # 적분 게인
Kd = 0.1  # 미분 게인

# ==================== GPIO 초기화 ====================
GPIO.setmode(GPIO.BCM)
GPIO.setup([DIR_PIN, PWM_PIN, SERVO_PIN], GPIO.OUT)

motor_pwm = GPIO.PWM(PWM_PIN, MOTOR_FREQ)
servo_pwm = GPIO.PWM(SERVO_PIN, SERVO_FREQ)
motor_pwm.start(0)
servo_pwm.start(0)

# PID 변수
prev_error = 0
integral = 0

# ==================== 모터 제어 함수 ====================
def set_servo_angle(degree):
    """서보 모터 각도 설정"""
    degree = max(SERVO_ANGLE_MIN, min(SERVO_ANGLE_MAX, degree))
    duty = SERVO_MIN_DUTY + (degree * (SERVO_MAX_DUTY - SERVO_MIN_DUTY) / 180.0)
    servo_pwm.ChangeDutyCycle(duty)

def move_forward(speed):
    """전진"""
    GPIO.output(DIR_PIN, GPIO.HIGH)
    motor_pwm.ChangeDutyCycle(speed)

def stop_motor():
    """정지"""
    motor_pwm.ChangeDutyCycle(0)

# ==================== 이미지 처리 함수 ====================
def preprocess_image(frame):
    """이미지 전처리"""
    # 리사이즈 (처리 속도 향상)
    img = cv2.resize(frame, (IMG_WIDTH, IMG_HEIGHT))

    # ROI 설정 (하단 부분만 사용)
    h, w = img.shape[:2]
    roi_top = int(h * ROI_TOP)
    roi_bottom = int(h * ROI_BOTTOM)
    roi = img[roi_top:roi_bottom, :]

    return roi, roi_top

def detect_line(roi):
    """라인 검출 및 중심 위치 계산"""
    # 그레이스케일 변환
    gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)

    # 가우시안 블러 (노이즈 제거)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 이진화 (흰색 라인 추출)
    _, binary = cv2.threshold(blurred, WHITE_THRESHOLD, 255, cv2.THRESH_BINARY)

    # 모폴로지 연산 (라인 연결)
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # 상단과 하단의 라인 중심 찾기
    h, w = binary.shape

    # 하단 중심 (더 신뢰할 수 있음)
    bottom_center = find_line_center(binary, int(h * 0.8))

    # 상단 중심
    top_center = find_line_center(binary, int(h * 0.2))

    return binary, top_center, bottom_center

def find_line_center(binary, y_pos):
    """특정 y 위치에서 라인의 중심 x 좌표 찾기"""
    row = binary[y_pos, :]

    # 흰색 픽셀 위치 찾기
    white_pixels = np.where(row > 128)[0]

    if len(white_pixels) == 0:
        return None  # 라인을 찾지 못함

    # 라인 중심 계산
    center = int(np.mean(white_pixels))

    # 라인 폭 확인
    line_width = white_pixels[-1] - white_pixels[0]

    # 너무 넓거나 좁으면 무시
    if line_width < MIN_LINE_WIDTH or line_width > MAX_LINE_WIDTH:
        return None

    return center

def calculate_error(bottom_center, top_center, img_center):
    """에러 계산 (라인 중심과 이미지 중심의 차이)"""
    if bottom_center is None:
        return None  # 라인을 찾지 못함

    # 하단 중심을 기준으로 에러 계산
    error = bottom_center - img_center

    # 상단 중심도 있으면 방향성 고려
    if top_center is not None:
        # 상단과 하단의 차이로 곡선 예측
        direction = top_center - bottom_center
        # 방향성 반영 (가중치 조정 가능)
        error = error + direction * 0.3

    return error

def pid_control(error):
    """PID 제어로 서보 각도 계산"""
    global prev_error, integral

    if error is None:
        # 라인을 찾지 못하면 이전 각도 유지
        return None

    # PID 계산
    integral += error
    integral = max(-100, min(100, integral))  # 적분 제한
    derivative = error - prev_error

    output = Kp * error + Ki * integral + Kd * derivative

    # 각도 변환 (에러를 각도로)
    # 에러 범위: -img_center ~ +img_center
    # 각도 범위: -45도 ~ +45도
    max_error = IMG_WIDTH / 2
    angle_offset = (error / max_error) * 45  # 최대 45도 회전

    angle = SERVO_ANGLE_CENTER - angle_offset
    angle = max(SERVO_ANGLE_MIN, min(SERVO_ANGLE_MAX, angle))

    prev_error = error

    return angle

def detect_traffic_light(frame):
    """트래픽 라이트 감지 (간단한 색상 기반)"""
    # ROI 설정 (상단 부분)
    h, w = frame.shape[:2]
    roi = frame[0:int(h*0.3), :]

    # HSV 변환
    hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)

    # 빨간색 범위
    red_lower1 = np.array([0, 50, 50])
    red_upper1 = np.array([10, 255, 255])
    red_lower2 = np.array([170, 50, 50])
    red_upper2 = np.array([180, 255, 255])

    red_mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
    red_mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)

    # 초록색 범위
    green_lower = np.array([40, 50, 50])
    green_upper = np.array([80, 255, 255])
    green_mask = cv2.inRange(hsv, green_lower, green_upper)

    # 픽셀 수 계산
    red_pixels = cv2.countNonZero(red_mask)
    green_pixels = cv2.countNonZero(green_mask)

    # 임계값 (조정 필요)
    threshold = 100

    if red_pixels > threshold:
        return 'red'
    elif green_pixels > threshold:
        return 'green'
    else:
        return None

# ==================== 메인 함수 ====================
def main():
    print("=" * 60)
    print("컴퓨터 비전 기반 라인트레이싱")
    print("=" * 60)
    print(f"이미지 크기: {IMG_WIDTH}x{IMG_HEIGHT}")
    print(f"ROI: 상단 {int(ROI_TOP*100)}% ~ 하단 100%")
    print("=" * 60)

    # 카메라 초기화
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"format": "RGB888", "size": (640, 480)}
    )
    picam2.configure(config)
    picam2.start()
    time.sleep(1)

    print("카메라 시작 완료. 라인트레이싱 시작!\n")

    # 초기 설정
    set_servo_angle(SERVO_ANGLE_CENTER)
    time.sleep(0.1)

    img_center = IMG_WIDTH / 2
    lost_line_count = 0
    max_lost_count = 10  # 라인을 잃은 최대 프레임 수

    try:
        while True:
            start_time = time.time()

            # 프레임 캡처
            frame_rgb = picam2.capture_array()

            # 트래픽 라이트 감지
            traffic_light = detect_traffic_light(frame_rgb)
            if traffic_light == 'red':
                print("🔴 빨간불 감지 - 정지")
                stop_motor()
                set_servo_angle(SERVO_ANGLE_CENTER)
                # 빨간불이 꺼질 때까지 대기
                while True:
                    frame_rgb = picam2.capture_array()
                    traffic_light = detect_traffic_light(frame_rgb)
                    if traffic_light == 'green':
                        print("🟢 초록불 감지 - 재시작")
                        time.sleep(0.5)  # 짧은 대기
                        break
                    time.sleep(0.1)

            # 이미지 전처리
            roi, roi_top = preprocess_image(frame_rgb)

            # 라인 검출
            binary, top_center, bottom_center = detect_line(roi)

            # 에러 계산
            error = calculate_error(bottom_center, top_center, img_center)

            if error is None:
                lost_line_count += 1
                if lost_line_count > max_lost_count:
                    # 라인을 오래 잃었으면 정지
                    print("⚠ 라인을 찾을 수 없습니다 - 정지")
                    stop_motor()
                else:
                    # 잠시 느리게 진행
                    move_forward(SPEED_SLOW)
            else:
                lost_line_count = 0

                # PID 제어
                angle = pid_control(error)

                if angle is not None:
                    set_servo_angle(angle)
                    move_forward(SPEED_NORMAL)

                    # 디버그 출력 (선택적)
                    # print(f"Error: {error:.1f}, Angle: {angle:.1f}°")

            # 화면 표시 (디버그용)
            display_frame = frame_rgb.copy()

            # ROI 표시
            h, w = display_frame.shape[:2]
            roi_top_px = int(h * ROI_TOP)
            cv2.rectangle(display_frame, (0, roi_top_px), (w, h), (0, 255, 0), 2)

            # 라인 중심 표시
            if bottom_center is not None:
                # 원본 프레임 좌표로 변환
                scale_x = w / IMG_WIDTH
                center_x = int(bottom_center * scale_x)
                center_y = int((h * ROI_TOP + h * 0.8) * (h / IMG_HEIGHT))
                cv2.circle(display_frame, (center_x, center_y), 5, (255, 0, 0), -1)
                cv2.line(display_frame, (w//2, center_y), (center_x, center_y), (0, 0, 255), 2)

            # 정보 표시
            info_text = f"Error: {error:.1f}" if error is not None else "No line"
            cv2.putText(display_frame, info_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            if traffic_light:
                cv2.putText(display_frame, f"Traffic: {traffic_light}", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                           (0, 0, 255) if traffic_light == 'red' else (0, 255, 0), 2)

            cv2.imshow("Line Tracing CV", display_frame)

            # 처리 시간 계산
            elapsed = time.time() - start_time
            fps = 1.0 / elapsed if elapsed > 0 else 0
            # print(f"FPS: {fps:.1f}")  # 디버그용

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("종료합니다...")
                break

            time.sleep(0.01)  # 최소 대기

    except KeyboardInterrupt:
        print("\n키보드 인터럽트로 종료합니다...")
    except Exception as e:
        print(f"오류 발생: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 정리
        stop_motor()
        set_servo_angle(SERVO_ANGLE_CENTER)
        picam2.stop()
        cv2.destroyAllWindows()
        motor_pwm.stop()
        servo_pwm.stop()
        GPIO.cleanup()
        print("시스템 종료 완료")

if __name__ == "__main__":
    main()

