#!/usr/bin/env python3
"""
라인트레이싱 실행 스크립트
학습된 모델을 사용하여 라인트레이싱을 수행합니다.

사용법:
    python3 run_line_tracing.py
"""

import os
import time
import numpy as np
import cv2
import tflite_runtime.interpreter as tflite
from picamera2 import Picamera2
import RPi.GPIO as GPIO

# ==================== 설정 ====================
IMG_SIZE = 224  # Transfer Learning 모델 표준 크기
MODEL_PATH = "model.tflite"  # 또는 "line_tracing_model.h5" (Keras 모델)
USE_KERAS_MODEL = False  # True로 변경하면 Keras 모델 사용 (더 빠름, Pi에서 가능한 경우)
LABELS = ["green", "left", "middle", "noline", "red", "right"]  # camera_main_gpt.py와 동일한 순서

# GPIO 설정
DIR_PIN = 16
PWM_PIN = 12
SERVO_PIN = 13

MOTOR_FREQ = 1000
SERVO_FREQ = 50
SERVO_MAX_DUTY = 12
SERVO_MIN_DUTY = 3

# 속도 설정
SPEED_FORWARD = 50
SPEED_TURN = 40
SERVO_ANGLE_FORWARD = 90
SERVO_ANGLE_LEFT = 60
SERVO_ANGLE_RIGHT = 120

# ==================== GPIO 초기화 ====================
GPIO.setmode(GPIO.BCM)
GPIO.setup([DIR_PIN, PWM_PIN, SERVO_PIN], GPIO.OUT)

motor_pwm = GPIO.PWM(PWM_PIN, MOTOR_FREQ)
servo_pwm = GPIO.PWM(SERVO_PIN, SERVO_FREQ)
motor_pwm.start(0)
servo_pwm.start(0)

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

def preprocess_frame(frame_rgb):
    """프레임 전처리"""
    img = cv2.resize(frame_rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    return img[None, ...]

def main():
    print("=" * 60)
    print("라인트레이싱 실행")
    print("=" * 60)

    # 모델 로드
    keras_model_path = "line_tracing_model.h5"

    if USE_KERAS_MODEL and os.path.exists(keras_model_path):
        # Keras 모델 사용 (더 빠름)
        print(f"Keras 모델 로드 중: {keras_model_path}")
        from tensorflow import keras
        model = keras.models.load_model(keras_model_path)
        print("✓ Keras 모델 로드 완료")
        use_keras = True
    elif os.path.exists(MODEL_PATH):
        # TFLite 모델 사용
        print(f"TFLite 모델 로드 중: {MODEL_PATH}")
        interpreter = tflite.Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors()
        inp = interpreter.get_input_details()[0]
        out = interpreter.get_output_details()[0]
        print("✓ TFLite 모델 로드 완료")
        use_keras = False
    else:
        print(f"✗ 오류: 모델 파일을 찾을 수 없습니다.")
        print(f"  - Keras 모델: {keras_model_path}")
        print(f"  - TFLite 모델: {MODEL_PATH}")
        print("먼저 train_model.py를 실행하여 모델을 학습하세요.")
        return

    # 카메라 초기화
    print("카메라 초기화 중...")
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"format": "RGB888", "size": (640, 480)}
    )
    picam2.configure(config)
    picam2.start()
    time.sleep(1)
    print("✓ 카메라 초기화 완료")

    # 초기 설정
    set_servo_angle(SERVO_ANGLE_FORWARD)
    time.sleep(0.1)

    print("\n라인트레이싱 시작! (q 키로 종료)")
    print("=" * 60)

    # 예측 안정화를 위한 변수
    last_prediction = "forward"
    prediction_buffer = []
    buffer_size = 3

    try:
        while True:
            # 프레임 캡처
            frame_rgb = picam2.capture_array()

            # 전처리
            x = preprocess_frame(frame_rgb)

            # 추론
            if use_keras:
                probs = model.predict(x, verbose=0)[0]
            else:
                interpreter.set_tensor(inp["index"], x)
                interpreter.invoke()
                probs = interpreter.get_tensor(out["index"])[0]

            pred_id = int(np.argmax(probs))
            pred_label = LABELS[pred_id]
            confidence = probs[pred_id]

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
                set_servo_angle(SERVO_ANGLE_FORWARD)
                break

            elif most_common == "green" and confidence > 0.7:
                print(f"🟢 초록불 감지! (신뢰도: {confidence:.2f}) - 계속 진행")
                # 초록불은 그냥 통과 (또는 다음 단계로 전환하는 로직 추가 가능)

            # 라인트레이싱 제어
            if most_common == "middle":
                set_servo_angle(SERVO_ANGLE_FORWARD)
                move_forward(SPEED_FORWARD)
                last_prediction = "forward"

            elif most_common == "left":
                set_servo_angle(SERVO_ANGLE_LEFT)
                move_forward(SPEED_TURN)
                last_prediction = "left"

            elif most_common == "right":
                set_servo_angle(SERVO_ANGLE_RIGHT)
                move_forward(SPEED_TURN)
                last_prediction = "right"

            elif most_common == "noline":
                # 라인이 없으면 마지막 방향 유지
                if last_prediction == "left":
                    set_servo_angle(SERVO_ANGLE_LEFT)
                elif last_prediction == "right":
                    set_servo_angle(SERVO_ANGLE_RIGHT)
                else:
                    set_servo_angle(SERVO_ANGLE_FORWARD)
                move_forward(SPEED_TURN)

            # 디버그 출력 (선택적)
            # print(f"예측: {most_common} (신뢰도: {confidence:.2f})")

            # 화면 표시 (선택적)
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            text = f"{most_common} ({confidence:.2f})"
            cv2.putText(frame_bgr, text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow("Line Tracing", frame_bgr)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("종료합니다...")
                break

            time.sleep(0.02)  # 20ms 간격

    except KeyboardInterrupt:
        print("\n키보드 인터럽트로 종료합니다...")
    except Exception as e:
        print(f"오류 발생: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 정리
        stop_motor()
        set_servo_angle(SERVO_ANGLE_FORWARD)
        picam2.stop()
        cv2.destroyAllWindows()
        motor_pwm.stop()
        servo_pwm.stop()
        GPIO.cleanup()
        print("시스템 종료 완료")

if __name__ == "__main__":
    main()

