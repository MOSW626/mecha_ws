#!/usr/bin/env python3
"""
통합 자율주행 시스템
- Phase 1: 라인트레이싱 (카메라 + AI 모델) - 초록불 감지 시 Phase 2로 전환, 빨간불 감지 시 정지
- Phase 2: 초음파 센서 기반 고속 레이싱
"""

import RPi.GPIO as GPIO
import time
import sys
import numpy as np
import cv2
import tflite_runtime.interpreter as tflite
from picamera2 import Picamera2

# ==================== GPIO 설정 ====================
# PID Gains (초음파 모드용 - 고속 레이싱을 위해 조정)
Kp = 0.65  # 증가: 더 빠른 반응
Ki = 0.0
Kd = 0.03  # 증가: 더 안정적인 제어

base_angle = 90
prev_error = 0
integral = 0

# GPIO pin locations
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

# Speed 설정 (Phase별로 다름)
SPEED_MIN_LINETRACE = 40  # 라인트레이싱 모드 최소 속도
SPEED_MAX_LINETRACE = 50  # 라인트레이싱 모드 최대 속도
SPEED_MIN_ULTRASONIC = 60  # 초음파 모드 최소 속도 (더 빠름)
SPEED_MAX_ULTRASONIC = 80  # 초음파 모드 최대 속도 (더 빠름)

MOTOR_SPEED = SPEED_MIN_LINETRACE

# Distance Clipping values
MIN_CM, MAX_CM = 3.0, 150.0
ALPHA = 0.85

# ==================== AI 모델 설정 ====================
IMG = 240
MODEL_PATH = "./model.tflite"
# infer_source (1).py의 labels 사용
LABELS = ["green", "left", "middle", "noline", "red", "right"]

# ==================== GPIO 초기화 ====================
GPIO.setmode(GPIO.BCM)
GPIO.setup([DIR_PIN, PWM_PIN, SERVO_PIN], GPIO.OUT)
GPIO.setup([TRIG_LEFT, TRIG_RIGHT], GPIO.OUT)
GPIO.setup([ECHO_LEFT, ECHO_RIGHT], GPIO.IN)

motor_pwm = GPIO.PWM(PWM_PIN, MOTOR_FREQ)
servo_pwm = GPIO.PWM(SERVO_PIN, SERVO_FREQ)
motor_pwm.start(0)
servo_pwm.start(0)

# ==================== 초음파 센서 함수 ====================
def sample_distance(trig, echo):
    GPIO.output(trig, True)
    time.sleep(0.00001)  # 10us로 단축 (더 빠른 샘플링)
    GPIO.output(trig, False)

    t0 = time.time()
    while GPIO.input(echo) == 0:
        if time.time() - t0 > 0.02:
            return None
    start = time.time()

    while GPIO.input(echo) == 1:
        if time.time() - start > 0.02:
            return 8787
    end = time.time()

    dist = (end - start) * 34300 / 2.0
    dist = max(MIN_CM, min(dist, MAX_CM))
    return dist

def read_stable(trig, echo):
    val = sample_distance(trig, echo)
    time.sleep(0.0005)  # 샘플링 간격 단축
    return val

def smooth(prev_value, new_value, alpha=ALPHA):
    if new_value == 8787:
        return 150
    if new_value is None:
        return prev_value
    if prev_value is None:
        return new_value
    return alpha * new_value + (1 - alpha) * prev_value

# ==================== 모터 제어 함수 ====================
def set_servo_angle(degree):
    degree = max(45, min(135, degree))
    duty = SERVO_MIN_DUTY + (degree * (SERVO_MAX_DUTY - SERVO_MIN_DUTY) / 180.0)
    servo_pwm.ChangeDutyCycle(duty)
    # time.sleep 제거하여 더 빠른 반응

def move_forward(speed):
    GPIO.output(DIR_PIN, GPIO.HIGH)
    motor_pwm.ChangeDutyCycle(speed)

def move_backward():
    GPIO.output(DIR_PIN, GPIO.LOW)
    motor_pwm.ChangeDutyCycle(MOTOR_SPEED)

def stop_motor():
    motor_pwm.ChangeDutyCycle(0)

def speed_from_angle(angle, amin=45, amid=90, amax=135, vmin=None, vmax=None):
    if vmin is None:
        vmin = SPEED_MIN_ULTRASONIC
    if vmax is None:
        vmax = SPEED_MAX_ULTRASONIC

    if angle <= amid:
        t = (angle - amin) / (amid - amin)
        t = max(0.0, min(1.0, t))
        if t != 0:
            t = 1 / t * 3
        t = min(15, t)
        return vmin + (vmax - vmin) * t * 0.3  # 0.25 -> 0.3으로 증가 (더 빠른 속도)
    else:
        t = (amax - angle) / (amax - amid)
        t = max(0.0, min(1.0, t))
        if t != 0:
            t = 1 / t * 3
        t = min(15, t)
        return vmin + (vmax - vmin) * t * 0.3

# ==================== AI 모델 함수 ====================
def preprocess_frame_for_model(frame):
    """카메라 프레임을 모델 입력 형식으로 변환"""
    # RGB로 변환 (Picamera2는 RGB 반환)
    if frame.ndim == 3:
        if frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
        elif frame.shape[2] == 1:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)

    # 리사이즈 및 정규화
    frame = cv2.resize(frame, (IMG, IMG), interpolation=cv2.INTER_AREA)
    frame = frame.astype(np.float32) / 255.0
    return frame[None, ...]

# ==================== 라인트레이싱 모드 ====================
def line_tracing_mode(picam2, interpreter, inp, out):
    """라인트레이싱 모드: 카메라와 AI 모델을 사용하여 라인을 따라감"""
    global MOTOR_SPEED

    print("=== 라인트레이싱 모드 시작 ===")
    MOTOR_SPEED = SPEED_MIN_LINETRACE

    # 초기 서보 각도 설정
    set_servo_angle(90)
    time.sleep(0.1)

    last_prediction = "forward"
    prediction_count = {"left": 0, "right": 0, "forward": 0}

    try:
        while True:
            # 카메라 프레임 캡처
            frame_rgb = picam2.capture_array()

            # 모델 입력 전처리
            x = preprocess_frame_for_model(frame_rgb)

            # 모델 추론
            interpreter.set_tensor(inp["index"], x)
            interpreter.invoke()

            probs = interpreter.get_tensor(out["index"])[0]
            pred_id = int(np.argmax(probs))
            pred_label = LABELS[pred_id]
            confidence = probs[pred_id]

            # 신호등 감지 (높은 신뢰도 필요)
            if pred_label == "green" and confidence > 0.7:
                print(f"🟢 초록불 감지! (신뢰도: {confidence:.2f}) - 초음파 모드로 전환")
                time.sleep(0.1)  # 짧은 대기
                return "ultrasonic"  # Phase 2로 전환

            elif pred_label == "red" and confidence > 0.7:
                print(f"🔴 빨간불 감지! (신뢰도: {confidence:.2f}) - 정지")
                stop_motor()
                set_servo_angle(90)
                return "stop"  # 정지

            # 라인트레이싱 제어 (신호등이 아닐 때만)
            if pred_label in ["left", "right", "forward"]:
                # 예측 안정화를 위한 카운팅
                prediction_count[pred_label] = prediction_count.get(pred_label, 0) + 1

                # 연속된 예측이 일정 횟수 이상일 때만 동작 변경
                if prediction_count[pred_label] >= 2:
                    last_prediction = pred_label
                    prediction_count = {"left": 0, "right": 0, "forward": 0}

            # 모터 제어
            if last_prediction == "left":
                set_servo_angle(60)  # 왼쪽으로 회전
                MOTOR_SPEED = SPEED_MIN_LINETRACE + 5
            elif last_prediction == "right":
                set_servo_angle(120)  # 오른쪽으로 회전
                MOTOR_SPEED = SPEED_MIN_LINETRACE + 5
            else:  # forward
                set_servo_angle(90)  # 직진
                MOTOR_SPEED = SPEED_MAX_LINETRACE

            move_forward(MOTOR_SPEED)

            # 디버그 출력 (선택적)
            # print(f"예측: {pred_label} (신뢰도: {confidence:.2f}) | 동작: {last_prediction} | 속도: {MOTOR_SPEED:.0f}")

            time.sleep(0.02)  # 20ms 간격 (50Hz)

    except KeyboardInterrupt:
        return "stop"
    except Exception as e:
        print(f"라인트레이싱 모드 오류: {e}")
        return "stop"

# ==================== 초음파 센서 모드 (고속 레이싱) ====================
def ultrasonic_racing_mode():
    """초음파 센서만을 사용한 고속 레이싱 모드"""
    global prev_error, integral, MOTOR_SPEED

    print("=== 초음파 센서 고속 레이싱 모드 시작 ===")
    print("최대 속도로 레이싱을 시작합니다!")

    # PID 초기화
    prev_error = 0
    integral = 0
    MOTOR_SPEED = SPEED_MIN_ULTRASONIC

    last_left = None
    last_right = None

    # 초기 서보 각도 설정
    set_servo_angle(90)
    time.sleep(0.05)

    try:
        loop_count = 0
        while True:
            loop_count += 1

            # 초음파 센서 읽기
            raw_left = read_stable(TRIG_LEFT, ECHO_LEFT)
            raw_right = read_stable(TRIG_RIGHT, ECHO_RIGHT)

            # 스무딩
            left = smooth(last_left, raw_left)
            right = smooth(last_right, raw_right)
            last_left, last_right = left, right

            if left is None or right is None:
                continue

            # PID 제어
            error = left - right * 2.1
            integral += error
            # 적분 제한 (windup 방지)
            integral = max(-100, min(100, integral))
            derivative = error - prev_error

            output = Kp * error + Ki * integral + Kd * derivative
            angle = max(45, min(135, base_angle - output))

            # 각도에 따른 속도 조정
            MOTOR_SPEED = speed_from_angle(angle, vmin=SPEED_MIN_ULTRASONIC, vmax=SPEED_MAX_ULTRASONIC)

            # 각도 클리핑 및 반올림
            angle1 = max(50, min(130, base_angle - output))
            angle = round(angle1, 0)

            # 규칙 기반 로직: 벽에 너무 가까우면 회피
            if left <= 7:
                set_servo_angle(130)
                MOTOR_SPEED = SPEED_MIN_ULTRASONIC  # 위험 시 속도 감소
            elif right <= 7:
                set_servo_angle(50)
                MOTOR_SPEED = SPEED_MIN_ULTRASONIC
            else:
                set_servo_angle(angle)

            move_forward(MOTOR_SPEED)

            # 주기적 디버그 출력 (너무 자주 출력하지 않음)
            if loop_count % 50 == 0:
                print(f"L: {left:.1f}cm R: {right:.1f}cm Err: {error:.1f} "
                      f"Angle: {angle:.1f}° Speed: {MOTOR_SPEED:.0f}%")

            # 매우 짧은 대기 (최대 속도)
            time.sleep(0.0001)

            prev_error = error

    except KeyboardInterrupt:
        print("초음파 모드 중단")
    except Exception as e:
        print(f"초음파 모드 오류: {e}")

# ==================== 메인 함수 ====================
def main():
    """메인 실행 함수"""
    print("=" * 50)
    print("통합 자율주행 시스템 시작")
    print("=" * 50)
    print("Phase 1: 라인트레이싱 모드")
    print("Phase 2: 초음파 센서 고속 레이싱 모드")
    print("=" * 50)

    # AI 모델 로드
    try:
        interpreter = tflite.Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors()
        inp = interpreter.get_input_details()[0]
        out = interpreter.get_output_details()[0]
        print(f"✓ AI 모델 로드 완료: {MODEL_PATH}")
    except Exception as e:
        print(f"✗ AI 모델 로드 실패: {e}")
        print("초음파 센서 모드만 사용합니다.")
        ultrasonic_racing_mode()
        return

    # 카메라 초기화
    try:
        picam2 = Picamera2()
        picam2.configure(picam2.create_preview_configuration({"main": {"size": (640, 480)}}))
        picam2.start()
        time.sleep(1)  # 카메라 워밍업
        print("✓ 카메라 초기화 완료")
    except Exception as e:
        print(f"✗ 카메라 초기화 실패: {e}")
        print("초음파 센서 모드만 사용합니다.")
        ultrasonic_racing_mode()
        return

    try:
        # Phase 1: 라인트레이싱 모드
        result = line_tracing_mode(picam2, interpreter, inp, out)

        if result == "ultrasonic":
            # 카메라 종료
            picam2.stop()
            print("카메라 종료 - 초음파 센서 모드로 전환")
            time.sleep(0.5)

            # Phase 2: 초음파 센서 고속 레이싱 모드
            ultrasonic_racing_mode()

        elif result == "stop":
            print("시스템 정지")

    except KeyboardInterrupt:
        print("\n사용자에 의해 중단됨")
    except Exception as e:
        print(f"오류 발생: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 정리
        try:
            picam2.stop()
        except:
            pass
        stop_motor()
        set_servo_angle(90)
        motor_pwm.stop()
        servo_pwm.stop()
        GPIO.cleanup()
        print("시스템 종료 완료")

if __name__ == "__main__":
    main()

