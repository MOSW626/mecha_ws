#!/usr/bin/env python3
# cnn.tflite 파일을 사용하여 라인트레이싱을 합니다.
# 카메라 이미지를 처리하여 라인을 찾습니다.
# 매우 천천히 주행.
# left, right, forward, noline 을 판단해서 주행.
# red에서는 정지.
# green에서는 주행.

import cv2
import numpy as np
import time
import os
from picamera2 import Picamera2
import RPi.GPIO as GPIO

# Keras 모델 지원
try:
    import tensorflow as tf
    from tensorflow import keras
    USE_KERAS = True
except ImportError:
    USE_KERAS = False
    print("⚠ TensorFlow/Keras를 사용할 수 없습니다.")

# TFLite 모델 지원 (Keras가 없을 때 사용)
try:
    import tflite_runtime.interpreter as tflite
    USE_TFLITE = True
except ImportError:
    USE_TFLITE = False
    print("⚠ TFLite를 사용할 수 없습니다.")

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

# ==================== ML 모델 설정 ====================
# cnn 폴더의 모델 사용 (Keras 우선, 없으면 TFLite)
MODEL_DIR = "./cnn"
KERAS_MODEL_PATH = os.path.join(MODEL_DIR, "cnn_model.keras")
H5_MODEL_PATH = os.path.join(MODEL_DIR, "cnn_model.h5")
TFLITE_MODEL_PATH = os.path.join(MODEL_DIR, "cnn_model.tflite")

# 학습된 모델의 클래스 순서 (train_model.py와 동일)
LABELS = ["forward", "green", "left", "non", "red", "right"]
IMG_SIZE = 256  # 학습 시 사용한 이미지 크기

# ==================== GPIO 초기화 ====================
GPIO.setmode(GPIO.BCM)
GPIO.setup([DIR_PIN, PWM_PIN, SERVO_PIN], GPIO.OUT)

motor_pwm = GPIO.PWM(PWM_PIN, MOTOR_FREQ)
servo_pwm = GPIO.PWM(SERVO_PIN, SERVO_FREQ)
motor_pwm.start(0)
servo_pwm.start(0)

# ML 관련 변수
model = None  # Keras 모델
interpreter = None  # TFLite 인터프리터
inp = None
out = None
use_keras = False  # True면 Keras, False면 TFLite

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

# ==================== ML 이미지 처리 함수 ====================
def preprocess_frame(frame_rgb):
    """ML용 이미지 전처리"""
    img = cv2.resize(frame_rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    return img[None, ...]

def predict_ml(frame_rgb):
    """ML 모델로 예측"""
    global model, interpreter, inp, out, use_keras

    if model is None and interpreter is None:
        return None, 0.0

    try:
        x = preprocess_frame(frame_rgb)

        if use_keras and model is not None:
            # Keras 모델 사용
            probs = model.predict(x, verbose=0)[0]
        elif interpreter is not None:
            # TFLite 모델 사용
            interpreter.set_tensor(inp["index"], x)
            interpreter.invoke()
            probs = interpreter.get_tensor(out["index"])[0]
        else:
            return None, 0.0

        pred_id = int(np.argmax(probs))
        pred_label = LABELS[pred_id]
        confidence = float(probs[pred_id])
        return pred_label, confidence
    except Exception as e:
        print(f"⚠ ML 예측 오류: {e}")
        return None, 0.0

def map_label_to_direction(label):
    """ML 라벨을 방향으로 매핑"""
    if label == "left":
        return "left"
    elif label == "right":
        return "right"
    elif label == "forward":
        return "forward"
    elif label == "non":  # noline 대신 non 사용
        return "noline"
    else:
        return None

# ==================== 메인 함수 ====================
def main():
    global model, interpreter, inp, out, use_keras

    print("=" * 60)
    print("CNN 모델 기반 라인트레이싱")
    print("=" * 60)

    # ML 모델 로드 (Keras 우선, 없으면 TFLite)
    model_loaded = False

    # 1. Keras 모델 시도 (.keras 우선, 없으면 .h5)
    if USE_KERAS:
        if os.path.exists(KERAS_MODEL_PATH):
            print(f"Keras 모델 로드 시도: {KERAS_MODEL_PATH}")
            try:
                model = keras.models.load_model(KERAS_MODEL_PATH)
                use_keras = True
                model_loaded = True
                print("✓ Keras 모델 로드 완료 (.keras)")
            except Exception as e:
                print(f"⚠ .keras 모델 로드 실패: {e}")

        if not model_loaded and os.path.exists(H5_MODEL_PATH):
            print(f"Keras 모델 로드 시도: {H5_MODEL_PATH}")
            try:
                model = keras.models.load_model(H5_MODEL_PATH)
                use_keras = True
                model_loaded = True
                print("✓ Keras 모델 로드 완료 (.h5)")
            except Exception as e:
                print(f"⚠ .h5 모델 로드 실패: {e}")

    # 2. TFLite 모델 시도 (Keras가 없을 때)
    if not model_loaded and USE_TFLITE:
        if os.path.exists(TFLITE_MODEL_PATH):
            print(f"TFLite 모델 로드 시도: {TFLITE_MODEL_PATH}")
            try:
                interpreter = tflite.Interpreter(model_path=TFLITE_MODEL_PATH)
                interpreter.allocate_tensors()
                inp = interpreter.get_input_details()[0]
                out = interpreter.get_output_details()[0]
                use_keras = False
                model_loaded = True
                print("✓ TFLite 모델 로드 완료")
            except Exception as e:
                print(f"⚠ TFLite 모델 로드 실패: {e}")

    if not model_loaded:
        print("✗ 사용 가능한 모델 파일을 찾을 수 없습니다.")
        print(f"  시도한 경로:")
        if USE_KERAS:
            print(f"    - {KERAS_MODEL_PATH}")
            print(f"    - {H5_MODEL_PATH}")
        if USE_TFLITE:
            print(f"    - {TFLITE_MODEL_PATH}")
        return

    print(f"\n사용 모델: {'Keras' if use_keras else 'TFLite'}")
    print(f"이미지 크기: {IMG_SIZE}x{IMG_SIZE}")
    print(f"클래스: {LABELS}\n")

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

    print("라인트레이싱 시작!\n")

    # 예측 안정화를 위한 변수
    last_prediction = "middle"
    prediction_buffer = []
    buffer_size = 3

    try:
        while True:
            # 프레임 캡처
            frame_rgb = picam2.capture_array()

            # ML 예측
            pred_label, confidence = predict_ml(frame_rgb)

            if pred_label is None:
                continue

            # 예측 버퍼에 추가
            prediction_buffer.append(pred_label)
            if len(prediction_buffer) > buffer_size:
                prediction_buffer.pop(0)

            # 버퍼에서 가장 많이 나온 예측 선택
            from collections import Counter
            most_common = Counter(prediction_buffer).most_common(1)[0][0]

            # 신호등 처리
            if most_common == "red" and confidence > 0.7:
                print(f"🔴 빨간불 감지! (신뢰도: {confidence:.2f}) - 정지")
                stop_motor()
                set_servo_angle(SERVO_ANGLE_CENTER)
                # 빨간불이 꺼질 때까지 대기
                while True:
                    frame_rgb = picam2.capture_array()
                    pred_label, confidence = predict_ml(frame_rgb)
                    if pred_label == "green" and confidence > 0.7:
                        print(f"🟢 초록불 감지! (신뢰도: {confidence:.2f}) - 재시작")
                        time.sleep(0.5)
                        break
                    time.sleep(0.1)
                continue

            # 방향 판단
            direction = map_label_to_direction(most_common)

            if direction is None:
                continue

            # 주행 제어
            if direction == "noline":
                # 라인 없으면 이전 방향 유지
                if last_prediction == "left":
                    set_servo_angle(60)
                elif last_prediction == "right":
                    set_servo_angle(120)
                else:
                    set_servo_angle(SERVO_ANGLE_CENTER)
                move_forward(SPEED_SLOW)
            elif direction == "left":
                set_servo_angle(60)
                move_forward(SPEED_SLOW)
                last_prediction = "left"
            elif direction == "right":
                set_servo_angle(120)
                move_forward(SPEED_SLOW)
                last_prediction = "right"
            elif direction == "forward":
                set_servo_angle(SERVO_ANGLE_CENTER)
                move_forward(SPEED_SLOW)
                last_prediction = "forward"

            # 디버그 출력
            print(f"Direction: {direction}, Label: {most_common}, Confidence: {confidence:.2f}")

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
