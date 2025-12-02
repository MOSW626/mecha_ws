#!/usr/bin/env python3
"""
라인트레이싱 모델 학습 스크립트 (Mac M2 최적화)
Transfer Learning을 사용하여 더 나은 성능의 모델을 학습합니다.

사용법:
    python3 train_model.py
"""

import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import MobileNetV2, EfficientNetB0
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# 설정
DATA_DIR = "data"
IMG_SIZE = 224  # MobileNet/EfficientNet 표준 크기
BATCH_SIZE = 32
EPOCHS = 50
MODEL_SAVE_PATH = "line_tracing_model.h5"
TFLITE_SAVE_PATH = "model.tflite"  # 선택사항: Raspberry Pi에서 사용할 경우

# 클래스 정의 (순서 중요! camera_main_gpt.py와 동일)
CLASSES = ["green", "left", "middle", "noline", "red", "right"]
NUM_CLASSES = len(CLASSES)

# Transfer Learning 모델 선택 (MobileNetV2 또는 EfficientNetB0)
USE_EFFICIENTNET = False  # True로 변경하면 EfficientNet 사용 (더 정확하지만 느림)

def load_data():
    """데이터 로드 및 전처리"""
    print("데이터 로딩 중...")

    images = []
    labels = []

    for class_idx, class_name in enumerate(CLASSES):
        class_dir = os.path.join(DATA_DIR, class_name)

        if not os.path.exists(class_dir):
            print(f"⚠ 경고: {class_dir} 디렉토리가 없습니다. 건너뜁니다.")
            continue

        image_files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        print(f"  {class_name}: {len(image_files)}개 이미지")

        for img_file in image_files:
            img_path = os.path.join(class_dir, img_file)
            try:
                # 이미지 로드 (BGR -> RGB)
                img = cv2.imread(img_path)
                if img is None:
                    continue

                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # 리사이즈
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)

                # 정규화 (Transfer Learning 모델은 보통 [0, 1] 범위 사용)
                img = img.astype(np.float32) / 255.0

                images.append(img)
                labels.append(class_idx)
            except Exception as e:
                print(f"  ⚠ 이미지 로드 실패: {img_path} - {e}")
                continue

    if len(images) == 0:
        raise ValueError("로드된 이미지가 없습니다. 데이터를 먼저 수집하세요.")

    X = np.array(images)
    y = np.array(labels)

    # 원-핫 인코딩
    y = keras.utils.to_categorical(y, NUM_CLASSES)

    print(f"\n총 {len(X)}개 이미지 로드 완료")
    print(f"클래스별 분포:")
    for i, class_name in enumerate(CLASSES):
        count = np.sum(np.argmax(y, axis=1) == i)
        print(f"  {class_name}: {count}개")

    return X, y

def create_model():
    """Transfer Learning을 사용한 모델 생성"""
    print(f"\n{'EfficientNetB0' if USE_EFFICIENTNET else 'MobileNetV2'} 기반 모델 생성 중...")

    # Base 모델 선택
    if USE_EFFICIENTNET:
        base_model = EfficientNetB0(
            input_shape=(IMG_SIZE, IMG_SIZE, 3),
            include_top=False,
            weights='imagenet'
        )
    else:
        base_model = MobileNetV2(
            input_shape=(IMG_SIZE, IMG_SIZE, 3),
            include_top=False,
            weights='imagenet'
        )

    # Base 모델의 가중치를 고정 (처음에는 학습하지 않음)
    base_model.trainable = False

    # 모델 구성
    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))

    # 데이터 증강 레이어 (학습 중에만 적용)
    x = layers.RandomFlip("horizontal")(inputs)
    x = layers.RandomRotation(0.1)(x)
    x = layers.RandomZoom(0.1)(x)
    x = layers.RandomBrightness(0.1)(x)
    x = layers.RandomContrast(0.1)(x)

    # Base 모델 통과
    x = base_model(x, training=False)

    # Global Average Pooling
    x = layers.GlobalAveragePooling2D()(x)

    # Dropout
    x = layers.Dropout(0.3)(x)

    # Dense 레이어
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)

    # 출력 레이어
    outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)

    model = Model(inputs, outputs)

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy', 'top_k_categorical_accuracy']
    )

    return model, base_model

def fine_tune_model(model, base_model, X_train, y_train, X_val, y_val):
    """Fine-tuning: Base 모델의 일부 레이어를 학습 가능하게 설정"""
    print("\nFine-tuning 시작...")

    # Base 모델의 상위 레이어만 학습 가능하게 설정
    base_model.trainable = True

    # 하위 레이어는 고정 (과적합 방지)
    if USE_EFFICIENTNET:
        # EfficientNet의 경우 마지막 몇 개 블록만 학습
        for layer in base_model.layers[:-20]:
            layer.trainable = False
    else:
        # MobileNetV2의 경우 마지막 몇 개 블록만 학습
        for layer in base_model.layers[:-30]:
            layer.trainable = False

    # 학습률을 낮춰서 미세 조정
    model.compile(
        optimizer=Adam(learning_rate=0.0001),  # 더 낮은 학습률
        loss='categorical_crossentropy',
        metrics=['accuracy', 'top_k_categorical_accuracy']
    )

    return model

def plot_history(history):
    """학습 히스토리 시각화"""
    try:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Accuracy
        axes[0].plot(history.history['accuracy'], label='Train Accuracy')
        axes[0].plot(history.history['val_accuracy'], label='Val Accuracy')
        axes[0].set_title('Model Accuracy')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()
        axes[0].grid(True)

        # Loss
        axes[1].plot(history.history['loss'], label='Train Loss')
        axes[1].plot(history.history['val_loss'], label='Val Loss')
        axes[1].set_title('Model Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True)

        plt.tight_layout()
        plt.savefig('training_history.png', dpi=150)
        print("✓ 학습 히스토리 그래프 저장: training_history.png")
        plt.close()
    except Exception as e:
        print(f"⚠ 그래프 저장 실패: {e}")

def main():
    print("=" * 60)
    print("라인트레이싱 모델 학습 (Transfer Learning)")
    print("=" * 60)

    # 데이터 로드
    X, y = load_data()

    # 학습/검증 데이터 분할
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\n학습 데이터: {len(X_train)}개")
    print(f"검증 데이터: {len(X_val)}개")

    # 모델 생성
    model, base_model = create_model()
    model.summary()

    # 콜백 설정
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            MODEL_SAVE_PATH,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
    ]

    # 1단계: Base 모델 고정 상태에서 학습
    print("\n" + "=" * 60)
    print("1단계: Base 모델 고정 상태에서 학습")
    print("=" * 60)
    history1 = model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )

    # 2단계: Fine-tuning
    print("\n" + "=" * 60)
    print("2단계: Fine-tuning (Base 모델 일부 레이어 학습)")
    print("=" * 60)
    model = fine_tune_model(model, base_model, X_train, y_train, X_val, y_val)

    # Fine-tuning 콜백 (더 작은 patience)
    fine_tune_callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            MODEL_SAVE_PATH,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]

    history2 = model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=20,  # Fine-tuning은 적은 에포크
        validation_data=(X_val, y_val),
        callbacks=fine_tune_callbacks,
        verbose=1
    )

    # 히스토리 병합
    combined_history = {
        'accuracy': history1.history['accuracy'] + history2.history['accuracy'],
        'val_accuracy': history1.history['val_accuracy'] + history2.history['val_accuracy'],
        'loss': history1.history['loss'] + history2.history['loss'],
        'val_loss': history1.history['val_loss'] + history2.history['val_loss']
    }

    # 최종 평가
    print("\n" + "=" * 60)
    print("최종 평가")
    print("=" * 60)
    val_loss, val_accuracy, val_top_k = model.evaluate(X_val, y_val, verbose=0)
    print(f"검증 정확도: {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")
    print(f"검증 Top-K 정확도: {val_top_k:.4f} ({val_top_k*100:.2f}%)")
    print(f"검증 손실: {val_loss:.4f}")

    # 학습 히스토리 시각화
    plot_history(type('obj', (object,), {'history': combined_history})())

    # 모델 저장
    model.save(MODEL_SAVE_PATH)
    print(f"\n✓ Keras 모델 저장 완료: {MODEL_SAVE_PATH}")

    # TensorFlow Lite 변환 (선택사항 - Raspberry Pi에서 사용할 경우)
    try:
        import tensorflow as tf
        print("\nTensorFlow Lite 모델로 변환 중...")
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()

        with open(TFLITE_SAVE_PATH, 'wb') as f:
            f.write(tflite_model)
        print(f"✓ TFLite 모델 저장 완료: {TFLITE_SAVE_PATH} (Raspberry Pi용)")
    except Exception as e:
        print(f"⚠ TFLite 변환 실패 (선택사항): {e}")

    print("\n" + "=" * 60)
    print("학습 완료!")
    print(f"최종 검증 정확도: {val_accuracy*100:.2f}%")
    print("=" * 60)

if __name__ == "__main__":
    main()
