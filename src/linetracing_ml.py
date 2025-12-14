#!/usr/bin/env python3
# linetracing_ml.py
# infer_source.py의 로직을 이식하여 ML 판단을 수행합니다.

import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
import os

# ==================== Settings ====================
MODEL_PATH = "../models/gpu_model_lite.tflite"
# infer_source.py에 있던 라벨 순서 그대로 사용
LABELS = ["cv", "green", "red"]

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
# linetracing_ml.py 수정

def preprocess_frame_for_model(frame_rgb):
    """
    이미지의 상단 60%만 잘라서 모델에 넣습니다.
    (바닥에 있는 흰색 라인을 신호등으로 착각하는 것 방지)
    """
    global inp

    # 1. 이미지 자르기 (ROI)
    h, w, c = frame_rgb.shape
    # 상단 0% ~ 40%까지만 사용 (필요에 따라 0.5 ~ 0.7로 조절)
    roi_h = int(h * 0.4)
    frame_cropped = frame_rgb[0:roi_h, :]

    # 2. BGR 변환 (infer_source 방식 유지)
    f = cv2.cvtColor(frame_cropped, cv2.COLOR_RGB2BGR)

    # 3. 모델 입력 크기에 맞춰 리사이즈
    expected_shape = inp["shape"]
    f = cv2.resize(f, (expected_shape[2], expected_shape[1]), interpolation=cv2.INTER_AREA)

    # ... (이하 동일: 정규화 및 배치 차원 추가) ...
    expected_dtype = np.dtype(inp["dtype"])
    if expected_dtype == np.uint8:
        out_img = (f).astype(np.uint8)
    else:
        out_img = (f.astype(np.float32) / 255.0).astype(np.float32)

    return out_img[None, ...]
# linetracing_ml.py 수정

# [설정 추가] 0.0 ~ 1.0 사이. 0.8 추천 (80% 이상 확실할 때만 인정)
CONFIDENCE_THRESHOLD = 0.9

def judge_ml(frame_rgb):
    global interpreter, inp, out, LABELS

    if interpreter is None:
        return None

    try:
        x = preprocess_frame_for_model(frame_rgb)
        interpreter.set_tensor(inp["index"], x)
        interpreter.invoke()

        probs = interpreter.get_tensor(out["index"])[0]
        pred_id = int(np.argmax(probs))
        confidence = probs[pred_id]  # 가장 높은 확률값 가져오기

        # [수정된 부분] 확률이 낮으면 무시하고 None(또는 noline) 리턴
        if confidence < CONFIDENCE_THRESHOLD:
            # 디버깅용: 낮은 확률로 감지된 것은 무엇인지 확인
            # print(f"Ignored low confidence: {LABELS[pred_id]} ({confidence:.2f})")
            return "noline"

        pred_label = LABELS[pred_id]
        return pred_label

    except Exception as e:
        print(f"⚠ ML Prediction Error: {e}")
        return None
