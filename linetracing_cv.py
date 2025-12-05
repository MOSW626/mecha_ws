#!/usr/bin/env python3
# opencv 를 활용해 line tracing 을 합니다.
# 카메라 이미지를 처리하여 라인을 찾습니다.
# 매우 천천히 주행.
# left, right, forward, noline 을 판단해서 주행.
# red에서는 정지.
# green에서는 주행.
# 하얀 선의 각도와 중앙 위치를 기반으로 제어합니다.

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

# 속도 설정 (매우 천천히)
SPEED_SLOW = 35  # 매우 천천히 주행
SERVO_ANGLE_CENTER = 90
SERVO_ANGLE_MAX = 135
SERVO_ANGLE_MIN = 45

# ==================== 이미지 처리 설정 ====================
IMG_WIDTH = 320
IMG_HEIGHT = 240
ROI_TOP = 0.4
ROI_BOTTOM = 1.0

# 라인 검출 설정
WHITE_THRESHOLD = 200
MIN_LINE_WIDTH = 2
MAX_LINE_WIDTH = 20

# 제어 파라미터
Kp_center = 0.6  # 중앙 위치 오차에 대한 게인
Kp_angle = 0.3   # 각도 오차에 대한 게인
Kd = 0.1         # 미분 게인

# ==================== GPIO 초기화 ====================
GPIO.setmode(GPIO.BCM)
GPIO.setup([DIR_PIN, PWM_PIN, SERVO_PIN], GPIO.OUT)

motor_pwm = GPIO.PWM(PWM_PIN, MOTOR_FREQ)
servo_pwm = GPIO.PWM(SERVO_PIN, SERVO_FREQ)
motor_pwm.start(0)
servo_pwm.start(0)

# 제어 변수
prev_center_error = 0
prev_angle = SERVO_ANGLE_CENTER
prev_line_angle = 0.0

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
    """이미지 전처리 (대비 향상 포함)"""
    img = cv2.resize(frame, (IMG_WIDTH, IMG_HEIGHT))

    # 대비 향상 (CLAHE 사용 - 적응형 히스토그램 균등화)
    # LAB 색공간으로 변환하여 L 채널에만 적용 (색상 보존)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l_channel, a, b = cv2.split(lab)

    # CLAHE 적용 (대비 향상)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)

    # 다시 RGB로 변환
    img = cv2.merge([l_channel, a, b])
    img = cv2.cvtColor(img, cv2.COLOR_LAB2RGB)

    # ROI 추출
    h, w = img.shape[:2]
    roi_top = int(h * ROI_TOP)
    roi_bottom = int(h * ROI_BOTTOM)
    roi = img[roi_top:roi_bottom, :]
    return roi, roi_top

def detect_line_with_angle(roi):
    """라인 검출 및 각도 계산"""
    gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, WHITE_THRESHOLD, 255, cv2.THRESH_BINARY)

    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    h, w = binary.shape

    # 하단과 상단에서 라인 중심 찾기
    bottom_center = find_line_center(binary, int(h * 0.8))
    top_center = find_line_center(binary, int(h * 0.2))

    # 라인 각도 계산
    line_angle = 0.0
    if bottom_center is not None and top_center is not None:
        # 두 점 사이의 각도 계산 (라디안)
        dy = h * 0.6  # 상단과 하단 사이의 거리
        dx = bottom_center - top_center
        line_angle = np.arctan2(dy, dx) * 180.0 / np.pi  # 도 단위로 변환
        # -90 ~ 90도 범위로 정규화
        if line_angle > 90:
            line_angle = line_angle - 180
        elif line_angle < -90:
            line_angle = line_angle + 180
    elif bottom_center is not None:
        # 하단만 있으면 이전 각도 유지하거나 0으로 설정
        line_angle = prev_line_angle

    return binary, top_center, bottom_center, line_angle

def find_line_center(binary, y_pos):
    """특정 y 위치에서 라인의 중심 x 좌표 찾기"""
    row = binary[y_pos, :]
    white_pixels = np.where(row > 128)[0]

    if len(white_pixels) == 0:
        return None

    center = int(np.mean(white_pixels))
    line_width = white_pixels[-1] - white_pixels[0]

    if line_width < MIN_LINE_WIDTH or line_width > MAX_LINE_WIDTH:
        return None

    return center

def calculate_control_output(bottom_center, line_angle, img_center):
    """중앙 위치 오차와 각도를 기반으로 제어 출력 계산"""
    global prev_center_error, prev_angle, prev_line_angle

    if bottom_center is None:
        return None, None

    # 중앙 위치 오차 계산
    center_error = bottom_center - img_center

    # 각도 오차 (라인이 기울어진 정도)
    angle_error = line_angle  # 라인이 기울어진 각도

    # 중앙 위치 보정
    max_error = IMG_WIDTH / 2
    center_correction = (center_error / max_error) * 45  # 최대 45도

    # 각도 보정 (라인이 기울어진 정도에 따라)
    angle_correction = angle_error * Kp_angle

    # 최종 각도 계산
    angle_offset = center_correction + angle_correction

    # 미분 항 추가 (변화율 고려)
    center_derivative = center_error - prev_center_error
    derivative_correction = center_derivative * Kd

    angle_offset += derivative_correction

    # 이전 각도와의 차이를 고려한 보정
    angle_change = angle_offset
    new_angle = SERVO_ANGLE_CENTER - angle_offset
    new_angle = max(SERVO_ANGLE_MIN, min(SERVO_ANGLE_MAX, new_angle))

    # 이전 값 업데이트
    prev_center_error = center_error
    prev_angle = new_angle
    prev_line_angle = line_angle

    return new_angle, center_error

def detect_traffic_light(frame):
    """트래픽 라이트 감지"""
    h, w = frame.shape[:2]
    roi = frame[0:int(h*0.3), :]

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

    red_pixels = cv2.countNonZero(red_mask)
    green_pixels = cv2.countNonZero(green_mask)
    threshold = 100

    if red_pixels > threshold:
        return 'red'
    elif green_pixels > threshold:
        return 'green'
    else:
        return None

def judge_direction(bottom_center, line_angle, img_center):
    """left, right, forward, noline 판단"""
    if bottom_center is None:
        return 'noline'

    center_error = abs(bottom_center - img_center)
    threshold = IMG_WIDTH * 0.15  # 15% 임계값

    if center_error < threshold and abs(line_angle) < 10:
        return 'forward'
    elif bottom_center < img_center - threshold:
        return 'left'
    elif bottom_center > img_center + threshold:
        return 'right'
    else:
        return 'forward'

# ==================== 메인 함수 ====================
def main():
    global prev_center_error, prev_angle, prev_line_angle

    print("=" * 60)
    print("OpenCV 기반 라인트레이싱 (각도 기반 제어)")
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
    max_lost_count = 10

    # 초기화
    prev_center_error = 0
    prev_angle = SERVO_ANGLE_CENTER
    prev_line_angle = 0.0

    try:
        while True:
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
                        time.sleep(0.5)
                        break
                    time.sleep(0.1)

            # 이미지 전처리
            roi, roi_top = preprocess_image(frame_rgb)

            # 라인 검출 (각도 포함)
            binary, top_center, bottom_center, line_angle = detect_line_with_angle(roi)

            # 방향 판단
            direction = judge_direction(bottom_center, line_angle, img_center)

            # 제어 출력 계산
            angle, center_error = calculate_control_output(bottom_center, line_angle, img_center)

            if direction == 'noline':
                lost_line_count += 1
                if lost_line_count > max_lost_count:
                    print("⚠ 라인을 찾을 수 없습니다 - 정지")
                    stop_motor()
                else:
                    # 이전 각도 유지하며 느리게 진행
                    move_forward(SPEED_SLOW)
            else:
                lost_line_count = 0

                if angle is not None:
                    set_servo_angle(angle)
                    move_forward(SPEED_SLOW)

                    # 디버그 출력
                    print(f"Direction: {direction}, Center Error: {center_error:.1f}, "
                          f"Line Angle: {line_angle:.1f}°, Servo Angle: {angle:.1f}°")

            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\n키보드 인터럽트로 종료합니다...")
    except Exception as e:
        print(f"오류 발생: {e}")
        import traceback
        traceback.print_exc()
    finally:
        stop_motor()
        set_servo_angle(SERVO_ANGLE_CENTER)
        picam2.stop()
        motor_pwm.stop()
        servo_pwm.stop()
        GPIO.cleanup()
        print("시스템 종료 완료")

if __name__ == "__main__":
    main()
