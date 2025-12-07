#!/usr/bin/env python3
# linetracing_ml.py
# infer_source.py의 로직을 이식하여 ML 판단을 수행합니다.

import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
import os

# ==================== Settings ====================
MODEL_PATH = "./model.tflite"
# infer_source.py에 있던 라벨 순서 그대로 사용
LABELS = ["middle", "green", "left", "right", "red", "noline"]

interpreter = None
inp = None
out = None

def init_ml():
    """ML 모델을 로드하고 초기화합니다."""
    global interpreter, inp, out

    if not os.path.exists(MODEL_PATH):
        print(f"✗ ML Model not found: {MODEL_PATH}")
        return False

    try:
        interpreter = tflite.Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors()
        inp = interpreter.get_input_details()[0]
        out = interpreter.get_output_details()[0]
        print("✓ TFLite model loaded successfully (infer_source logic)")
        return True
    except Exception as e:
        print(f"⚠ ML Init Error: {e}")
        return False

def preprocess_frame_for_model(frame_rgb):
    """
    infer_source.py의 전처리 로직을 그대로 사용합니다.
    frame_rgb: Picamera2에서 받은 RGB 이미지 (H, W, 3)
    """
    global inp

    # [중요] Picamera2는 RGB를 주지만, 모델 학습이 BGR(OpenCV)로 되었다면 변환 필요
    # infer_source.py에서 성공했던 핵심 포인트입니다.
    f = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    expected_shape = inp["shape"]
    expected_c = int(expected_shape[-1])
    expected_dtype = np.dtype(inp["dtype"])

    # Resize to model spatial size
    f = cv2.resize(f, (expected_shape[2], expected_shape[1]), interpolation=cv2.INTER_AREA)

    # Convert to correct dtype and scaling
    # infer_source.py 로직: float32 모델이면 255로 나눔
    if expected_dtype == np.uint8:
        out_img = (f).astype(np.uint8)
    else:
        out_img = (f.astype(np.float32) / 255.0).astype(np.float32)

    # Add batch dim
    return out_img[None, ...]

def judge_ml(frame_rgb):
    """
    이미지를 받아 ML 추론 후 라벨(문자열)을 반환합니다.
    """
    global interpreter, inp, out, LABELS

    if interpreter is None:
        return None

    try:
        # 전처리
        x = preprocess_frame_for_model(frame_rgb)

        # 추론 수행
        interpreter.set_tensor(inp["index"], x)
        interpreter.invoke()

        probs = interpreter.get_tensor(out["index"])[0]
        pred_id = int(np.argmax(probs))
        pred_label = LABELS[pred_id]

        # 디버깅용 출력 (필요시 주석 처리)
        # print(f"ML Output: {pred_label} (prob: {probs[pred_id]:.2f})")

        return pred_label

    except Exception as e:
        print(f"⚠ ML Prediction Error: {e}")
        return None
