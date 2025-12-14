#!/usr/bin/env python3
"""
이미지 자동 분류 스크립트
data 폴더의 frame_*.png 파일들을 자동으로 분류합니다.

사용법:
    python3 auto_classify.py [--data-dir data] [--model-path path/to/model]

전략:
1. 학습된 모델이 있으면 사용 (가장 정확)
2. 모델이 없으면 기존 분류된 데이터로 빠르게 학습
3. 그것도 안되면 CV 기반 자동 분류
"""

import os
import sys
import argparse
import shutil
import cv2
import numpy as np
from pathlib import Path
from collections import Counter

# 클래스 정의 (기존 폴더 이름과 매칭)
CLASSES = {
    'forward': 'forward',
    'green': 'green',
    'left': 'left',
    'right': 'right',
    'red': 'red',
    'non': 'non'  # noline 대신 non 사용
}

CLASS_NAMES = list(CLASSES.keys())

def find_model():
    """학습된 모델 찾기"""
    possible_paths = [
        "model.tflite",
        "line_tracing_model.h5",
        "../line_tracing/model.tflite",
        "../line_tracing/line_tracing_model.h5",
        "models/model.tflite",
        "models/line_tracing_model.h5"
    ]

    for path in possible_paths:
        if os.path.exists(path):
            return path
    return None

def load_model(model_path):
    """모델 로드"""
    if model_path.endswith('.tflite'):
        try:
            import tflite_runtime.interpreter as tflite
            interpreter = tflite.Interpreter(model_path=model_path)
            interpreter.allocate_tensors()
            inp = interpreter.get_input_details()[0]
            out = interpreter.get_output_details()[0]
            return ('tflite', interpreter, inp, out)
        except Exception as e:
            print(f"⚠ TFLite 모델 로드 실패: {e}")
            return None
    elif model_path.endswith('.h5'):
        try:
            from tensorflow import keras
            model = keras.models.load_model(model_path)
            return ('keras', model, None, None)
        except Exception as e:
            print(f"⚠ Keras 모델 로드 실패: {e}")
            return None
    return None

def preprocess_image(img_path, img_size=224):
    """이미지 전처리"""
    img = cv2.imread(str(img_path))
    if img is None:
        return None

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    return img

def predict_with_model(img_path, model_info):
    """모델로 예측"""
    img = preprocess_image(img_path, 224)
    if img is None:
        return None, 0.0

    model_type, model, inp, out = model_info

    if model_type == 'tflite':
        model.set_tensor(inp["index"], img[None, ...])
        model.invoke()
        probs = model.get_tensor(out["index"])[0]
    else:  # keras
        probs = model.predict(img[None, ...], verbose=0)[0]

    pred_id = int(np.argmax(probs))
    confidence = probs[pred_id]

    # 클래스 매핑 (모델의 클래스 순서에 맞춰야 함)
    # 일반적인 순서: ["green", "left", "middle", "noline", "red", "right"]
    model_labels = ["green", "left", "middle", "noline", "red", "right"]

    if pred_id < len(model_labels):
        pred_label = model_labels[pred_id]
        # 우리 클래스로 매핑
        if pred_label == "middle":
            pred_label = "forward"
        elif pred_label == "noline":
            pred_label = "non"
        return pred_label, confidence

    return None, 0.0

def classify_with_cv(img_path):
    """CV 기반 자동 분류 (간단한 휴리스틱)"""
    img = cv2.imread(str(img_path))
    if img is None:
        return "non", 0.0

    h, w = img.shape[:2]

    # 하단 ROI (라인 검출)
    roi_bottom = img[int(h*0.6):, :]
    gray = cv2.cvtColor(roi_bottom, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    # 라인 중심 찾기
    row = binary[int(binary.shape[0]*0.8), :]
    white_pixels = np.where(row > 128)[0]

    if len(white_pixels) > 0:
        center = np.mean(white_pixels)
        img_center = w / 2
        error = center - img_center

        # 에러에 따라 분류
        if abs(error) < w * 0.1:
            return "forward", 0.7
        elif error < 0:
            return "left", 0.7
        else:
            return "right", 0.7
    else:
        # 라인 없음
        return "non", 0.5

    # 트래픽 라이트 검출 (상단)
    roi_top = img[0:int(h*0.3), :]
    hsv = cv2.cvtColor(roi_top, cv2.COLOR_BGR2HSV)

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

    if red_pixels > 100:
        return "red", 0.8
    elif green_pixels > 100:
        return "green", 0.8

    return "non", 0.5

def train_quick_model(data_dir):
    """기존 분류된 데이터로 빠르게 학습"""
    print("\n기존 데이터로 빠르게 학습 중...")

    try:
        from tensorflow import keras
        from tensorflow.keras import layers
        from sklearn.model_selection import train_test_split
    except ImportError:
        print("⚠ TensorFlow가 설치되지 않았습니다. CV 기반 분류로 전환합니다.")
        return None

    images = []
    labels = []

    # 기존 분류된 데이터 로드
    for class_name in CLASS_NAMES:
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.exists(class_dir):
            continue

        image_files = [f for f in os.listdir(class_dir)
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        if len(image_files) == 0:
            continue

        print(f"  {class_name}: {len(image_files)}개 이미지")

        for img_file in image_files[:50]:  # 최대 50개씩만 사용 (빠른 학습)
            img_path = os.path.join(class_dir, img_file)
            img = preprocess_image(img_path, 224)
            if img is not None:
                images.append(img)
                labels.append(CLASS_NAMES.index(class_name))

    if len(images) < 10:
        print("⚠ 학습 데이터가 부족합니다. CV 기반 분류로 전환합니다.")
        return None

    X = np.array(images)
    y = keras.utils.to_categorical(labels, len(CLASS_NAMES))

    # 간단한 모델 생성
    model = keras.Sequential([
        layers.Input(shape=(224, 224, 3)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(len(CLASS_NAMES), activation='softmax')
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # 빠른 학습 (5 에포크만)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train, epochs=5, validation_data=(X_val, y_val), verbose=0)

    val_acc = model.evaluate(X_val, y_val, verbose=0)[1]
    print(f"✓ 빠른 학습 완료 (검증 정확도: {val_acc:.2%})")

    return ('keras', model, None, None)

def main():
    parser = argparse.ArgumentParser(description='이미지 자동 분류')
    parser.add_argument('--data-dir', type=str, default='data',
                       help='데이터 디렉토리 경로 (기본값: data)')
    parser.add_argument('--model-path', type=str, default=None,
                       help='모델 파일 경로 (지정하지 않으면 자동 검색)')
    parser.add_argument('--confidence-threshold', type=float, default=0.5,
                       help='최소 신뢰도 임계값 (기본값: 0.5)')
    parser.add_argument('--method', type=str, choices=['auto', 'model', 'cv', 'train'],
                       default='auto', help='분류 방법 선택')

    args = parser.parse_args()

    data_dir = args.data_dir

    print("=" * 60)
    print("이미지 자동 분류 도구")
    print("=" * 60)
    print(f"데이터 디렉토리: {data_dir}")

    # 클래스별 폴더 생성
    for class_name in CLASS_NAMES:
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)
            print(f"✓ 폴더 생성: {class_dir}")

    # 분류할 이미지 찾기
    image_files = [f for f in os.listdir(data_dir)
                   if f.lower().startswith('frame_') and f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if len(image_files) == 0:
        print("⚠ 분류할 이미지가 없습니다.")
        return

    print(f"\n분류할 이미지: {len(image_files)}개")

    # 모델 로드 또는 학습
    model_info = None

    if args.method in ['auto', 'model']:
        # 모델 찾기
        model_path = args.model_path or find_model()

        if model_path and os.path.exists(model_path):
            print(f"\n모델 로드 중: {model_path}")
            model_info = load_model(model_path)
            if model_info:
                print("✓ 모델 로드 완료")
            else:
                print("⚠ 모델 로드 실패")
        elif args.method == 'model':
            print("✗ 모델을 찾을 수 없습니다.")
            return

    if not model_info and args.method in ['auto', 'train']:
        # 빠른 학습 시도
        model_info = train_quick_model(data_dir)

    # 분류 방법 결정
    use_model = model_info is not None
    use_cv = args.method == 'cv' or (args.method == 'auto' and not model_info)

    if use_model:
        print("\n모델 기반 분류 사용")
    elif use_cv:
        print("\nCV 기반 분류 사용")

    # 분류 실행
    print("\n분류 시작...")
    stats = {class_name: 0 for class_name in CLASS_NAMES}
    stats['low_confidence'] = 0
    stats['failed'] = 0

    for idx, img_file in enumerate(image_files, 1):
        img_path = os.path.join(data_dir, img_file)

        # 예측
        if use_model:
            pred_class, confidence = predict_with_model(img_path, model_info)
        else:
            pred_class, confidence = classify_with_cv(img_path)

        # 신뢰도 체크
        if confidence < args.confidence_threshold:
            stats['low_confidence'] += 1
            pred_class = 'non'  # 신뢰도 낮으면 non으로 분류

        if pred_class and pred_class in CLASS_NAMES:
            # 이동
            dst_dir = os.path.join(data_dir, pred_class)
            dst_path = os.path.join(dst_dir, img_file)

            # 중복 처리
            if os.path.exists(dst_path):
                base, ext = os.path.splitext(img_file)
                counter = 1
                while os.path.exists(dst_path):
                    dst_path = os.path.join(dst_dir, f"{base}_{counter}{ext}")
                    counter += 1

            shutil.move(img_path, dst_path)
            stats[pred_class] += 1

            if idx % 50 == 0:
                print(f"진행: {idx}/{len(image_files)} ({idx*100//len(image_files)}%)")
        else:
            stats['failed'] += 1

    # 결과 출력
    print("\n" + "=" * 60)
    print("분류 완료 통계:")
    print("=" * 60)
    total = 0
    for class_name in CLASS_NAMES:
        count = stats[class_name]
        print(f"  {class_name:15s}: {count:4d}개")
        total += count
    print(f"  {'Low confidence':15s}: {stats['low_confidence']:4d}개")
    print(f"  {'Failed':15s}: {stats['failed']:4d}개")
    print(f"  {'Total':15s}: {total:4d}개")
    print("=" * 60)

    print("\n✓ 분류 완료!")

if __name__ == "__main__":
    main()

