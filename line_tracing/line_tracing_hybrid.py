#!/usr/bin/env python3
"""
하이브리드 라인트레이싱 (CV 주도 + ML 보조)
CV 방식을 주로 사용하고, ML은 특정 상황에서만 사용합니다.

전략:
- 기본: CV 방식으로 빠른 제어 (대부분의 경우)
- ML 사용 시점:
  1. CV가 라인을 찾지 못할 때 (noline 감지)
  2. 트래픽 라이트 감지 (더 정확함)
  3. 복잡한 곡선 구간 (선택적)

사용법:
    python3 line_tracing_hybrid.py
"""

import cv2
import numpy as np
import time
import os
from picamera2 import Picamera2
import RPi.GPIO as GPIO

# ML 모델 사용 여부
try:
    import tflite_runtime.interpreter as tflite
    USE_ML = True
    MODEL_PATH = "model.tflite"
    LABELS = ["green", "left", "middle", "noline", "red", "right"]
except ImportError:
    USE_ML = False
    print("⚠ TFLite를 사용할 수 없습니다. CV 방식만 사용합니다.")

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
SPEED_FAST = 60
SERVO_ANGLE_CENTER = 90
SERVO_ANGLE_MAX = 135
SERVO_ANGLE_MIN = 45

# ==================== CV 이미지 처리 설정 ====================
IMG_WIDTH = 320  # CV 처리용 (빠름)
IMG_HEIGHT = 240
ROI_TOP = 0.4
ROI_BOTTOM = 1.0

# ML 이미지 처리 설정
ML_IMG_SIZE = 224  # ML 모델 입력 크기

# 라인 검출 설정
WHITE_THRESHOLD = 200
MIN_LINE_WIDTH = 2
MAX_LINE_WIDTH = 20

# PID 제어 설정
Kp = 0.8
Ki = 0.0
Kd = 0.1

# 하이브리드 전략 설정
CV_CONFIDENCE_THRESHOLD = 0.7  # CV 신뢰도 임계값
ML_USE_INTERVAL = 3  # ML을 몇 프레임마다 사용할지 (3 = 3프레임마다 1번)
ML_CONFIDENCE_THRESHOLD = 0.6  # ML 신뢰도 임계값

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

# ML 관련 변수
ml_interpreter = None
ml_inp = None
ml_out = None
frame_count = 0

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

# ==================== CV 이미지 처리 함수 ====================
def preprocess_image_cv(frame):
    """CV용 이미지 전처리"""
    img = cv2.resize(frame, (IMG_WIDTH, IMG_HEIGHT))
    h, w = img.shape[:2]
    roi_top = int(h * ROI_TOP)
    roi_bottom = int(h * ROI_BOTTOM)
    roi = img[roi_top:roi_bottom, :]
    return roi, roi_top

def detect_line_cv(roi):
    """CV 방식 라인 검출"""
    gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, WHITE_THRESHOLD, 255, cv2.THRESH_BINARY)

    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    h, w = binary.shape
    bottom_center = find_line_center(binary, int(h * 0.8))
    top_center = find_line_center(binary, int(h * 0.2))

    # 신뢰도 계산 (라인 폭과 픽셀 수 기반)
    confidence = 0.0
    if bottom_center is not None:
        row = binary[int(h * 0.8), :]
        white_pixels = np.where(row > 128)[0]
        if len(white_pixels) > 0:
            line_width = white_pixels[-1] - white_pixels[0]
            # 라인 폭이 적절하면 높은 신뢰도
            if MIN_LINE_WIDTH <= line_width <= MAX_LINE_WIDTH:
                confidence = 0.9
            else:
                confidence = 0.5

    return binary, top_center, bottom_center, confidence

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

def calculate_error_cv(bottom_center, top_center, img_center):
    """CV 방식 에러 계산"""
    if bottom_center is None:
        return None, 0.0

    error = bottom_center - img_center

    if top_center is not None:
        direction = top_center - bottom_center
        error = error + direction * 0.3

    confidence = 0.9 if bottom_center is not None else 0.0
    return error, confidence

# ==================== ML 이미지 처리 함수 ====================
def preprocess_image_ml(frame_rgb):
    """ML용 이미지 전처리"""
    img = cv2.resize(frame_rgb, (ML_IMG_SIZE, ML_IMG_SIZE), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    return img[None, ...]

def predict_ml(frame_rgb):
    """ML 모델로 예측"""
    global ml_interpreter, ml_inp, ml_out

    if not USE_ML or ml_interpreter is None:
        return None, 0.0

    try:
        x = preprocess_image_ml(frame_rgb)
        ml_interpreter.set_tensor(ml_inp["index"], x)
        ml_interpreter.invoke()
        probs = ml_interpreter.get_tensor(ml_out["index"])[0]
        pred_id = int(np.argmax(probs))
        pred_label = LABELS[pred_id]
        confidence = probs[pred_id]
        return pred_label, confidence
    except Exception as e:
        print(f"⚠ ML 예측 오류: {e}")
        return None, 0.0

# ==================== 트래픽 라이트 감지 ====================
def detect_traffic_light_cv(frame):
    """CV 방식 트래픽 라이트 감지"""
    h, w = frame.shape[:2]
    roi = frame[0:int(h*0.3), :]

    hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)

    # 빨간색
    red_lower1 = np.array([0, 50, 50])
    red_upper1 = np.array([10, 255, 255])
    red_lower2 = np.array([170, 50, 50])
    red_upper2 = np.array([180, 255, 255])
    red_mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
    red_mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)

    # 초록색
    green_lower = np.array([40, 50, 50])
    green_upper = np.array([80, 255, 255])
    green_mask = cv2.inRange(hsv, green_lower, green_upper)

    red_pixels = cv2.countNonZero(red_mask)
    green_pixels = cv2.countNonZero(green_mask)
    threshold = 100

    if red_pixels > threshold:
        return 'red', 0.8
    elif green_pixels > threshold:
        return 'green', 0.8
    else:
        return None, 0.0

# ==================== PID 제어 ====================
def pid_control(error):
    """PID 제어로 서보 각도 계산"""
    global prev_error, integral

    if error is None:
        return None

    integral += error
    integral = max(-100, min(100, integral))
    derivative = error - prev_error

    output = Kp * error + Ki * integral + Kd * derivative

    max_error = IMG_WIDTH / 2
    angle_offset = (error / max_error) * 45
    angle = SERVO_ANGLE_CENTER - angle_offset
    angle = max(SERVO_ANGLE_MIN, min(SERVO_ANGLE_MAX, angle))

    prev_error = error
    return angle

# ==================== 메인 함수 ====================
def main():
    global ml_interpreter, ml_inp, ml_out, frame_count

    print("=" * 60)
    print("하이브리드 라인트레이싱 (CV 주도 + ML 보조)")
    print("=" * 60)

    # ML 모델 로드
    if USE_ML and os.path.exists(MODEL_PATH):
        print(f"ML 모델 로드 중: {MODEL_PATH}")
        try:
            ml_interpreter = tflite.Interpreter(model_path=MODEL_PATH)
            ml_interpreter.allocate_tensors()
            ml_inp = ml_interpreter.get_input_details()[0]
            ml_out = ml_interpreter.get_output_details()[0]
            print("✓ ML 모델 로드 완료")
        except Exception as e:
            print(f"⚠ ML 모델 로드 실패: {e}")
            ml_interpreter = None
    else:
        print("⚠ ML 모델을 찾을 수 없습니다. CV 방식만 사용합니다.")
        ml_interpreter = None

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

    # 초기 설정
    set_servo_angle(SERVO_ANGLE_CENTER)
    time.sleep(0.1)

    img_center = IMG_WIDTH / 2
    lost_line_count = 0
    max_lost_count = 10
    last_ml_prediction = "middle"
    ml_prediction_buffer = []

    print("라인트레이싱 시작! (q 키로 종료)")
    print("=" * 60)

    try:
        while True:
            frame_count += 1
            start_time = time.time()

            # 프레임 캡처
            frame_rgb = picam2.capture_array()

            # 트래픽 라이트 감지 (CV + ML)
            traffic_light_cv, cv_conf = detect_traffic_light_cv(frame_rgb)
            traffic_light = traffic_light_cv

            # ML로도 확인 (주기적으로)
            if ml_interpreter and frame_count % ML_USE_INTERVAL == 0:
                ml_pred, ml_conf = predict_ml(frame_rgb)
                if ml_pred in ['red', 'green'] and ml_conf > ML_CONFIDENCE_THRESHOLD:
                    traffic_light = ml_pred
                    print(f"🟢 ML 트래픽 라이트 감지: {ml_pred} (신뢰도: {ml_conf:.2f})")

            if traffic_light == 'red':
                print("🔴 빨간불 감지 - 정지")
                stop_motor()
                set_servo_angle(SERVO_ANGLE_CENTER)
                while True:
                    frame_rgb = picam2.capture_array()
                    traffic_light_cv, _ = detect_traffic_light_cv(frame_rgb)
                    if ml_interpreter and frame_count % 2 == 0:
                        ml_pred, ml_conf = predict_ml(frame_rgb)
                        if ml_pred == 'green' and ml_conf > ML_CONFIDENCE_THRESHOLD:
                            traffic_light_cv = 'green'
                    if traffic_light_cv == 'green':
                        print("🟢 초록불 감지 - 재시작")
                        time.sleep(0.5)
                        break
                    time.sleep(0.1)

            # CV 방식으로 라인 검출 (주로 사용)
            roi, roi_top = preprocess_image_cv(frame_rgb)
            binary, top_center, bottom_center, cv_confidence = detect_line_cv(roi)
            error_cv, cv_conf = calculate_error_cv(bottom_center, top_center, img_center)

            # CV 신뢰도가 낮으면 ML 사용
            use_ml = False
            ml_pred = None
            ml_conf = 0.0

            if cv_confidence < CV_CONFIDENCE_THRESHOLD and ml_interpreter:
                if frame_count % ML_USE_INTERVAL == 0:
                    ml_pred, ml_conf = predict_ml(frame_rgb)
                    ml_prediction_buffer.append(ml_pred)
                    if len(ml_prediction_buffer) > 3:
                        ml_prediction_buffer.pop(0)

                    if ml_conf > ML_CONFIDENCE_THRESHOLD:
                        use_ml = True
                        last_ml_prediction = ml_pred

            # 제어 결정
            if use_ml and ml_pred:
                # ML 사용
                if ml_pred == "left":
                    set_servo_angle(60)
                    move_forward(SPEED_NORMAL)
                elif ml_pred == "right":
                    set_servo_angle(120)
                    move_forward(SPEED_NORMAL)
                elif ml_pred == "middle":
                    set_servo_angle(SERVO_ANGLE_CENTER)
                    move_forward(SPEED_NORMAL)
                elif ml_pred == "noline":
                    # 라인 없으면 이전 방향 유지
                    if last_ml_prediction == "left":
                        set_servo_angle(60)
                    elif last_ml_prediction == "right":
                        set_servo_angle(120)
                    else:
                        set_servo_angle(SERVO_ANGLE_CENTER)
                    move_forward(SPEED_SLOW)
            elif error_cv is not None:
                # CV 사용 (주로 사용)
                lost_line_count = 0
                angle = pid_control(error_cv)
                if angle is not None:
                    set_servo_angle(angle)
                    move_forward(SPEED_NORMAL)
            else:
                # 라인을 찾지 못함
                lost_line_count += 1
                if lost_line_count > max_lost_count:
                    print("⚠ 라인을 찾을 수 없습니다 - 정지")
                    stop_motor()
                else:
                    # ML로 재시도
                    if ml_interpreter:
                        ml_pred, ml_conf = predict_ml(frame_rgb)
                        if ml_pred and ml_conf > ML_CONFIDENCE_THRESHOLD:
                            if ml_pred == "left":
                                set_servo_angle(60)
                            elif ml_pred == "right":
                                set_servo_angle(120)
                            else:
                                set_servo_angle(SERVO_ANGLE_CENTER)
                            move_forward(SPEED_SLOW)
                    else:
                        move_forward(SPEED_SLOW)

            # 화면 표시
            display_frame = frame_rgb.copy()
            h, w = display_frame.shape[:2]
            roi_top_px = int(h * ROI_TOP)
            cv2.rectangle(display_frame, (0, roi_top_px), (w, h), (0, 255, 0), 2)

            if bottom_center is not None:
                scale_x = w / IMG_WIDTH
                center_x = int(bottom_center * scale_x)
                center_y = int((h * ROI_TOP + h * 0.8) * (h / IMG_HEIGHT))
                cv2.circle(display_frame, (center_x, center_y), 5, (255, 0, 0), -1)
                cv2.line(display_frame, (w//2, center_y), (center_x, center_y), (0, 0, 255), 2)

            # 정보 표시
            mode_text = "ML" if use_ml else "CV"
            info_text = f"Mode: {mode_text} | Error: {error_cv:.1f}" if error_cv is not None else f"Mode: {mode_text} | No line"
            cv2.putText(display_frame, info_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            if traffic_light:
                cv2.putText(display_frame, f"Traffic: {traffic_light}", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                           (0, 0, 255) if traffic_light == 'red' else (0, 255, 0), 2)

            if ml_pred and use_ml:
                cv2.putText(display_frame, f"ML: {ml_pred} ({ml_conf:.2f})", (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            cv2.imshow("Hybrid Line Tracing", display_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("종료합니다...")
                break

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
        cv2.destroyAllWindows()
        motor_pwm.stop()
        servo_pwm.stop()
        GPIO.cleanup()
        print("시스템 종료 완료")

if __name__ == "__main__":
    main()

