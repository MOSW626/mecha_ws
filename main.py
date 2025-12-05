#!/usr/bin/env python3
# 여기는 다음과 같이 실행
# python3 main.py -l : 라인트레이싱만
# python3 main.py -d : 주행만
# python3 main.py : 둘 다 사용하는 모드
# 둘 다 사용하는 경우, 라인트레이싱에서 주행모드로 초록불에서 바꾸면 됨.
# 초록이 됐을 때 직진으로 1.3초 정도 최고 속도(100)으로 달리면서 교체.
# 자연스럽게 쓰레드 활용. (라즈베리파이 4B)

import argparse
import threading
import time
import sys
import os
from picamera2 import Picamera2
import RPi.GPIO as GPIO

# 다른 파일들을 모듈로 import
try:
    import linetracing_cv
    import linetracing_ml
    import driving
except ImportError as e:
    print(f"모듈 import 오류: {e}")
    sys.exit(1)

try:
    import tflite_runtime.interpreter as tflite
    USE_TFLITE = True
except ImportError:
    USE_TFLITE = False

# ==================== GPIO 설정 ====================
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

# 전역 변수
mode_lock = threading.Lock()
current_mode = "linetracing"  # "linetracing" or "driving"
should_stop = False

# ==================== 모터 제어 함수 ====================
def set_servo_angle(degree):
    """서보 모터 각도 설정"""
    degree = max(45, min(135, degree))
    duty = SERVO_MIN_DUTY + (degree * (SERVO_MAX_DUTY - SERVO_MIN_DUTY) / 180.0)
    servo_pwm.ChangeDutyCycle(duty)

def move_forward(speed):
    """전진"""
    GPIO.output(DIR_PIN, GPIO.HIGH)
    motor_pwm.ChangeDutyCycle(speed)

def stop_motor():
    """정지"""
    motor_pwm.ChangeDutyCycle(0)

# ==================== 모드 전환 ====================
def switch_to_driving():
    """라인트레이싱에서 주행 모드로 전환"""
    global current_mode

    print("🟢 초록불 감지 - 주행 모드로 전환")

    # 1.3초 동안 직진 최고 속도(100)로 달리기
    set_servo_angle(90)  # 직진
    move_forward(100)  # 최고 속도
    time.sleep(1.3)

    with mode_lock:
        current_mode = "driving"

    print("주행 모드로 전환 완료")

# ==================== 라인트레이싱 쓰레드 ====================
def line_tracing_thread(picam2):
    """라인트레이싱 쓰레드 - linetracing_cv의 로직 사용"""
    global current_mode, should_stop

    # linetracing_cv의 함수들을 사용
    img_center = linetracing_cv.IMG_WIDTH / 2
    lost_line_count = 0
    max_lost_count = 10

    # ML 모델 로드 (선택적)
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

    # 초기 설정
    set_servo_angle(linetracing_cv.SERVO_ANGLE_CENTER)
    time.sleep(0.1)

    try:
        while not should_stop:
            with mode_lock:
                if current_mode != "linetracing":
                    time.sleep(0.1)
                    continue

            # 프레임 캡처
            frame_rgb = picam2.capture_array()

            # 트래픽 라이트 감지
            traffic_light = linetracing_cv.detect_traffic_light(frame_rgb)

            # ML로도 확인
            if interpreter:
                try:
                    # linetracing_ml의 전처리 함수 사용
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
                print("🔴 빨간불 감지 - 정지")
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

            # 이미지 전처리 및 라인 검출
            roi, roi_top = linetracing_cv.preprocess_image(frame_rgb)
            binary, top_center, bottom_center, line_angle = linetracing_cv.detect_line_with_angle(roi)

            # 제어 출력 계산
            angle, center_error = linetracing_cv.calculate_control_output(
                bottom_center, line_angle, img_center
            )

            if bottom_center is None:
                lost_line_count += 1
                if lost_line_count > max_lost_count:
                    print("⚠ 라인을 찾을 수 없습니다 - 정지")
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
        print(f"라인트레이싱 쓰레드 오류: {e}")
        import traceback
        traceback.print_exc()

# ==================== 주행 쓰레드 ====================
def driving_thread():
    """주행 쓰레드 - driving 모듈의 로직 사용"""
    global current_mode, should_stop

    # driving 모듈의 초음파 센서 함수 사용
    last_left = None
    last_right = None

    # 초기 서보 각도 설정
    set_servo_angle(90)
    time.sleep(0.05)

    try:
        while not should_stop:
            with mode_lock:
                if current_mode != "driving":
                    time.sleep(0.1)
                    continue

            # 초음파 센서 읽기
            raw_left = driving.read_stable(driving.TRIG_LEFT, driving.ECHO_LEFT)
            raw_right = driving.read_stable(driving.TRIG_RIGHT, driving.ECHO_RIGHT)

            # 스무딩
            left = driving.smooth(last_left, raw_left)
            right = driving.smooth(last_right, raw_right)
            last_left, last_right = left, right

            if left is None or right is None:
                continue

            # driving 모듈의 제어 로직 사용
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
        print(f"주행 쓰레드 오류: {e}")
        import traceback
        traceback.print_exc()

# ==================== 메인 함수 ====================
def line_tracing_only():
    """라인트레이싱만 실행"""
    linetracing_cv.main()

def driving_only():
    """주행만 실행"""
    driving.driving_mode()

def both_modes():
    """라인트레이싱과 주행을 함께 사용"""
    global should_stop

    print("=" * 60)
    print("통합 모드: 라인트레이싱 + 주행")
    print("=" * 60)

    # 카메라 초기화
    print("카메라 초기화 중...")
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"format": "RGB888", "size": (640, 480)}
    )
    picam2.configure(config)
    picam2.start()
    time.sleep(1)
    print("✓ 카메라 초기화 완료\n")

    # 쓰레드 시작
    lt_thread = threading.Thread(target=line_tracing_thread, args=(picam2,), daemon=True)
    dr_thread = threading.Thread(target=driving_thread, daemon=True)

    lt_thread.start()
    dr_thread.start()

    print("라인트레이싱 쓰레드 시작")
    print("주행 쓰레드 시작")
    print("통합 모드 실행 중... (Ctrl+C로 종료)\n")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n종료 중...")
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
        print("시스템 종료 완료")

def main():
    parser = argparse.ArgumentParser(description='자율주행 시스템')
    parser.add_argument('-l', '--linetracing', action='store_true', help='라인트레이싱만 실행')
    parser.add_argument('-d', '--driving', action='store_true', help='주행만 실행')

    args = parser.parse_args()

    if args.linetracing:
        print("라인트레이싱 모드로 실행합니다.")
        line_tracing_only()
    elif args.driving:
        print("주행 모드로 실행합니다.")
        driving_only()
    else:
        print("통합 모드로 실행합니다.")
        both_modes()

if __name__ == "__main__":
    main()
