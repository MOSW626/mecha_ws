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
DATA_DIR = "augmented_data"
TRAIN_DIR = os.path.join(DATA_DIR, "training")
TEST_DIR = os.path.join(DATA_DIR, "testing")

# 이미지 크기 설정 (큰 크기 = 더 정확하지만 느림)
# 224: 빠름, 256: 균형, 320: 정확하지만 느림
IMG_SIZE = 256  # 7-8시간 학습에 최적화된 크기

# 배치 크기 (GPU 메모리에 따라 조정)
# Mac M2의 경우 16-32가 적절, 더 큰 GPU면 64-128 가능
BATCH_SIZE = 16  # 이미지 크기가 커졌으므로 배치 크기 감소

# 에포크 수 (7-8시간 학습에 맞춤)
EPOCHS = 100  # 조기 종료로 실제로는 더 적게 학습될 수 있음
FINE_TUNE_EPOCHS = 30  # Fine-tuning 에포크

MODEL_SAVE_PATH = "cnn_model.h5"
MODEL_SAVE_PATH_KERAS = "cnn_model.keras"  # 더 안정적인 형식
TFLITE_SAVE_PATH = "cnn_model.tflite"  # 선택사항: Raspberry Pi에서 사용할 경우

def safe_load_model(model_path):
    """안전하게 모델 로드 (손상된 파일 처리)"""
    if not os.path.exists(model_path):
        return None
    try:
        model = keras.models.load_model(model_path)
        return model
    except Exception as e:
        print(f"⚠ 모델 로드 실패 ({model_path}): {e}")
        # 손상된 파일 삭제
        try:
            os.remove(model_path)
            print(f"✓ 손상된 모델 파일 삭제: {model_path}")
        except:
            pass
        return None

def safe_save_model(model, model_path):
    """안전하게 모델 저장"""
    try:
        # .keras 형식이 더 안정적
        if model_path.endswith('.h5'):
            # .keras 형식으로도 저장
            keras_path = model_path.replace('.h5', '.keras')
            model.save(keras_path, save_format='keras')
            # .h5 형식도 저장 시도
            try:
                model.save(model_path, save_format='h5')
            except:
                # .h5 실패해도 .keras는 성공했으므로 계속 진행
                pass
        else:
            model.save(model_path, save_format='keras')
        return True
    except Exception as e:
        print(f"⚠ 모델 저장 실패 ({model_path}): {e}")
        return False

# 클래스 정의 (augmented_data 폴더 구조에 맞춤)
CLASSES = ["forward", "green", "left", "non", "red", "right"]
NUM_CLASSES = len(CLASSES)

# Transfer Learning 모델 선택
# EfficientNetB0: 빠름, B1: 균형, B2: 정확하지만 느림
# 7-8시간 학습이면 B1 또는 B2 사용 가능
EFFICIENTNET_VERSION = "B1"  # "B0", "B1", "B2" 중 선택
USE_EFFICIENTNET = True  # True로 변경하면 EfficientNet 사용

def load_data():
    """데이터 로드 및 전처리 (training과 testing 분리)"""
    print("데이터 로딩 중...")
    print(f"Training 디렉토리: {TRAIN_DIR}")
    print(f"Testing 디렉토리: {TEST_DIR}")

    train_images = []
    train_labels = []
    test_images = []
    test_labels = []

    # Training 데이터 로드
    print("\n[Training 데이터]")
    for class_idx, class_name in enumerate(CLASSES):
        class_dir = os.path.join(TRAIN_DIR, class_name)

        if not os.path.exists(class_dir):
            print(f"⚠ 경고: {class_dir} 디렉토리가 없습니다. 건너뜁니다.")
            continue

        image_files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        print(f"  {class_name}: {len(image_files)}개 이미지")

        for img_file in image_files:
            img_path = os.path.join(class_dir, img_file)
            try:
                # 이미지 로드
                img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                if img is None:
                    continue

                # cv2.imread는 항상 BGR로 읽지만, 이미 RGB로 저장된 경우를 대비
                # 실제로는 cv2.imread는 항상 BGR이므로 변환 필요
                # 하지만 사용자가 이미 RGB라고 했으므로, 이미지가 실제로 어떻게 저장되었는지 확인 필요
                # 안전하게 BGR->RGB 변환 (이미 RGB라도 변환해도 문제 없음)
                if img.shape[2] == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # 리사이즈
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)

                # 정규화 (Transfer Learning 모델은 보통 [0, 1] 범위 사용)
                img = img.astype(np.float32) / 255.0

                train_images.append(img)
                train_labels.append(class_idx)
            except Exception as e:
                print(f"  ⚠ 이미지 로드 실패: {img_path} - {e}")
                continue

    # Testing 데이터 로드
    print("\n[Testing 데이터]")
    for class_idx, class_name in enumerate(CLASSES):
        class_dir = os.path.join(TEST_DIR, class_name)

        if not os.path.exists(class_dir):
            print(f"⚠ 경고: {class_dir} 디렉토리가 없습니다. 건너뜁니다.")
            continue

        image_files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        print(f"  {class_name}: {len(image_files)}개 이미지")

        for img_file in image_files:
            img_path = os.path.join(class_dir, img_file)
            try:
                # 이미지 로드
                img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                if img is None:
                    continue

                # BGR -> RGB 변환
                if img.shape[2] == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # 리사이즈
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)

                # 정규화
                img = img.astype(np.float32) / 255.0

                test_images.append(img)
                test_labels.append(class_idx)
            except Exception as e:
                print(f"  ⚠ 이미지 로드 실패: {img_path} - {e}")
                continue

    if len(train_images) == 0:
        raise ValueError("로드된 학습 이미지가 없습니다. 데이터를 먼저 수집하세요.")

    X_train = np.array(train_images)
    y_train = np.array(train_labels)
    X_test = np.array(test_images) if len(test_images) > 0 else None
    y_test = np.array(test_labels) if len(test_labels) > 0 else None

    # 원-핫 인코딩
    y_train = keras.utils.to_categorical(y_train, NUM_CLASSES)
    if y_test is not None:
        y_test = keras.utils.to_categorical(y_test, NUM_CLASSES)

    print(f"\n총 Training 이미지: {len(X_train)}개")
    print(f"총 Testing 이미지: {len(X_test) if X_test is not None else 0}개")
    print(f"\nTraining 클래스별 분포:")
    for i, class_name in enumerate(CLASSES):
        count = np.sum(np.argmax(y_train, axis=1) == i)
        print(f"  {class_name}: {count}개")
    if y_test is not None:
        print(f"\nTesting 클래스별 분포:")
        for i, class_name in enumerate(CLASSES):
            count = np.sum(np.argmax(y_test, axis=1) == i)
            print(f"  {class_name}: {count}개")

    return X_train, y_train, X_test, y_test

def create_model():
    """Transfer Learning을 사용한 모델 생성"""
    if USE_EFFICIENTNET:
        model_name = f"EfficientNet{EFFICIENTNET_VERSION}"
        print(f"\n{model_name} 기반 모델 생성 중...")

        # EfficientNet 버전 선택
        if EFFICIENTNET_VERSION == "B0":
            from tensorflow.keras.applications import EfficientNetB0
            base_model = EfficientNetB0(
                input_shape=(IMG_SIZE, IMG_SIZE, 3),
                include_top=False,
                weights='imagenet'
            )
        elif EFFICIENTNET_VERSION == "B1":
            from tensorflow.keras.applications import EfficientNetB1
            base_model = EfficientNetB1(
                input_shape=(IMG_SIZE, IMG_SIZE, 3),
                include_top=False,
                weights='imagenet'
            )
        elif EFFICIENTNET_VERSION == "B2":
            from tensorflow.keras.applications import EfficientNetB2
            base_model = EfficientNetB2(
                input_shape=(IMG_SIZE, IMG_SIZE, 3),
                include_top=False,
                weights='imagenet'
            )
        else:
            # 기본값: B1
            from tensorflow.keras.applications import EfficientNetB1
            base_model = EfficientNetB1(
                input_shape=(IMG_SIZE, IMG_SIZE, 3),
                include_top=False,
                weights='imagenet'
            )
    else:
        print(f"\nMobileNetV2 기반 모델 생성 중...")
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
    # 더 강한 증강으로 일반화 성능 향상
    x = layers.RandomFlip("horizontal")(inputs)
    x = layers.RandomRotation(0.15)(x)  # 0.1 -> 0.15
    x = layers.RandomZoom(0.15)(x)  # 0.1 -> 0.15
    x = layers.RandomBrightness(0.2)(x)  # 0.1 -> 0.2
    x = layers.RandomContrast(0.2)(x)  # 0.1 -> 0.2
    x = layers.RandomTranslation(0.1, 0.1)(x)  # 추가: 이동 증강

    # Base 모델 통과
    x = base_model(x, training=False)

    # Global Average Pooling
    x = layers.GlobalAveragePooling2D()(x)

    # Dropout
    x = layers.Dropout(0.3)(x)

    # Dense 레이어 (더 큰 모델로 성능 향상)
    x = layers.Dense(256, activation='relu')(x)  # 128 -> 256
    x = layers.Dropout(0.4)(x)  # 0.3 -> 0.4 (과적합 방지)
    x = layers.Dense(128, activation='relu')(x)  # 추가 레이어
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

def plot_history(history, test_accuracy=None, test_loss=None):
    """학습 히스토리 시각화 (상세 버전)"""
    try:
        # 2x2 서브플롯 생성
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        epochs = range(1, len(history['accuracy']) + 1)

        # 1. Accuracy 그래프
        axes[0, 0].plot(epochs, history['accuracy'], 'b-', label='Train Accuracy', linewidth=2)
        axes[0, 0].plot(epochs, history['val_accuracy'], 'r-', label='Val Accuracy', linewidth=2)
        if test_accuracy is not None:
            axes[0, 0].axhline(y=test_accuracy, color='g', linestyle='--',
                              label=f'Test Accuracy ({test_accuracy:.4f})', linewidth=2)
        axes[0, 0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Epoch', fontsize=12)
        axes[0, 0].set_ylabel('Accuracy', fontsize=12)
        axes[0, 0].legend(fontsize=10)
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_ylim([0, 1])

        # 2. Loss 그래프
        axes[0, 1].plot(epochs, history['loss'], 'b-', label='Train Loss', linewidth=2)
        axes[0, 1].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
        if test_loss is not None:
            axes[0, 1].axhline(y=test_loss, color='g', linestyle='--',
                              label=f'Test Loss ({test_loss:.4f})', linewidth=2)
        axes[0, 1].set_title('Model Loss', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Epoch', fontsize=12)
        axes[0, 1].set_ylabel('Loss', fontsize=12)
        axes[0, 1].legend(fontsize=10)
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Accuracy 차이 (과적합 확인)
        acc_diff = np.array(history['val_accuracy']) - np.array(history['accuracy'])
        axes[1, 0].plot(epochs, acc_diff, 'purple', linewidth=2)
        axes[1, 0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[1, 0].set_title('Accuracy Gap (Val - Train)', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Epoch', fontsize=12)
        axes[1, 0].set_ylabel('Accuracy Difference', fontsize=12)
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].fill_between(epochs, acc_diff, 0, alpha=0.3, color='purple')

        # 4. Loss 차이 (과적합 확인)
        loss_diff = np.array(history['loss']) - np.array(history['val_loss'])
        axes[1, 1].plot(epochs, loss_diff, 'orange', linewidth=2)
        axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[1, 1].set_title('Loss Gap (Train - Val)', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Epoch', fontsize=12)
        axes[1, 1].set_ylabel('Loss Difference', fontsize=12)
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].fill_between(epochs, loss_diff, 0, alpha=0.3, color='orange')

        plt.tight_layout()

        # 고해상도로 저장
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        print("✓ 학습 히스토리 그래프 저장: training_history.png (300 DPI)")

        # 추가: 간단한 버전도 저장
        fig_simple, axes_simple = plt.subplots(1, 2, figsize=(14, 5))
        axes_simple[0].plot(epochs, history['accuracy'], 'b-', label='Train', linewidth=2)
        axes_simple[0].plot(epochs, history['val_accuracy'], 'r-', label='Val', linewidth=2)
        if test_accuracy is not None:
            axes_simple[0].axhline(y=test_accuracy, color='g', linestyle='--',
                                  label=f'Test ({test_accuracy:.4f})', linewidth=2)
        axes_simple[0].set_title('Accuracy', fontsize=14, fontweight='bold')
        axes_simple[0].set_xlabel('Epoch')
        axes_simple[0].set_ylabel('Accuracy')
        axes_simple[0].legend()
        axes_simple[0].grid(True, alpha=0.3)
        axes_simple[0].set_ylim([0, 1])

        axes_simple[1].plot(epochs, history['loss'], 'b-', label='Train', linewidth=2)
        axes_simple[1].plot(epochs, history['val_loss'], 'r-', label='Val', linewidth=2)
        if test_loss is not None:
            axes_simple[1].axhline(y=test_loss, color='g', linestyle='--',
                                  label=f'Test ({test_loss:.4f})', linewidth=2)
        axes_simple[1].set_title('Loss', fontsize=14, fontweight='bold')
        axes_simple[1].set_xlabel('Epoch')
        axes_simple[1].set_ylabel('Loss')
        axes_simple[1].legend()
        axes_simple[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('training_history_simple.png', dpi=300, bbox_inches='tight')
        print("✓ 간단한 학습 히스토리 그래프 저장: training_history_simple.png (300 DPI)")

        plt.close('all')
    except Exception as e:
        print(f"⚠ 그래프 저장 실패: {e}")
        import traceback
        traceback.print_exc()

def main():
    print("=" * 60)
    print("라인트레이싱 모델 학습 (Transfer Learning)")
    print("=" * 60)

    # 시작 시 손상된 모델 파일 확인 및 삭제
    print("\n기존 모델 파일 확인 중...")
    for model_path in [MODEL_SAVE_PATH, MODEL_SAVE_PATH_KERAS]:
        if os.path.exists(model_path):
            test_model = safe_load_model(model_path)
            if test_model is None:
                print(f"✓ 손상된 모델 파일 정리 완료: {model_path}")
            else:
                print(f"✓ 기존 모델 파일 확인됨: {model_path} (학습 시작 시 무시됨)")

    # 데이터 로드 (이미 training/testing으로 분리됨)
    X_train, y_train, X_test, y_test = load_data()

    # Training 데이터에서 검증 데이터 분할 (20%)
    # stratify를 위해 원-핫 인코딩을 클래스 인덱스로 변환
    y_train_classes = np.argmax(y_train, axis=1)
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train_classes
    )

    print(f"\n학습 데이터: {len(X_train_split)}개")
    print(f"검증 데이터: {len(X_val)}개")
    if X_test is not None:
        print(f"테스트 데이터: {len(X_test)}개")

    # 모델 생성
    model, base_model = create_model()
    model.summary()

    # 콜백 완전 제거 (pickle 문제 방지)
    # 수동으로 모델 저장 및 EarlyStopping 구현
    callbacks = []  # 빈 리스트로 변경

    # EarlyStopping 설정
    early_stopping_patience = 15
    best_val_loss = float('inf')
    best_val_accuracy = 0.0
    patience_counter = 0

    # 1단계: Base 모델 고정 상태에서 학습
    print("\n" + "=" * 60)
    print("1단계: Base 모델 고정 상태에서 학습")
    print("=" * 60)

    history1 = None
    try:
        # 수동 EarlyStopping 구현 (pickle 문제 방지)
        history1_epochs = []
        for epoch in range(EPOCHS):
            print(f"\nEpoch {epoch+1}/{EPOCHS}")

            # 한 에포크 학습
            history_epoch = model.fit(
                X_train_split, y_train_split,
                batch_size=BATCH_SIZE,
                epochs=1,
                validation_data=(X_val, y_val),
                callbacks=callbacks,
                verbose=1
            )

            # 히스토리 저장
            if history1_epochs:
                for key in history_epoch.history:
                    history1_epochs[0].history[key].extend(history_epoch.history[key])
            else:
                history1_epochs.append(history_epoch)

            # EarlyStopping 체크 및 모델 저장
            current_val_loss = history_epoch.history['val_loss'][0]
            current_val_accuracy = history_epoch.history['val_accuracy'][0]

            # 더 나은 모델이면 저장
            if current_val_accuracy > best_val_accuracy:
                best_val_accuracy = current_val_accuracy
                best_val_loss = current_val_loss
                patience_counter = 0
                if safe_save_model(model, MODEL_SAVE_PATH):
                    print(f"✓ 모델 저장 (val_acc: {current_val_accuracy:.4f}, val_loss: {current_val_loss:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"\n조기 종료: {early_stopping_patience} 에포크 동안 개선 없음")
                    # 최종 모델 로드
                    loaded_model = safe_load_model(MODEL_SAVE_PATH)
                    if loaded_model is None:
                        loaded_model = safe_load_model(MODEL_SAVE_PATH_KERAS)
                    if loaded_model is not None:
                        model = loaded_model
                        print("✓ 최적 모델 로드 완료")
                    break

        history1 = history1_epochs[0] if history1_epochs else None

    except KeyboardInterrupt:
        print("\n⚠ 사용자에 의해 중단됨. 현재까지의 모델을 저장합니다...")
        safe_save_model(model, MODEL_SAVE_PATH)
        if history1_epochs:
            history1 = history1_epochs[0]
    except Exception as e:
        print(f"\n⚠ 학습 중 오류 발생: {e}")
        print("현재까지의 모델을 저장합니다...")
        if safe_save_model(model, MODEL_SAVE_PATH):
            print("✓ 모델 저장 완료")
        if history1_epochs:
            history1 = history1_epochs[0]

    # 2단계: Fine-tuning
    print("\n" + "=" * 60)
    print("2단계: Fine-tuning (Base 모델 일부 레이어 학습)")
    print("=" * 60)

    try:
        model = fine_tune_model(model, base_model, X_train_split, y_train_split, X_val, y_val)
    except Exception as e:
        print(f"⚠ Fine-tuning 모델 설정 중 오류: {e}")
        print("기존 모델로 계속 진행합니다...")

    # Fine-tuning 콜백 제거 (pickle 문제 방지)
    fine_tune_callbacks = []

    history2 = None
    history2_epochs = []
    fine_tune_patience = 10
    fine_tune_best_loss = float('inf')
    fine_tune_best_accuracy = 0.0
    fine_tune_patience_counter = 0

    try:
        # 수동 EarlyStopping으로 Fine-tuning
        for epoch in range(FINE_TUNE_EPOCHS):
            print(f"\nFine-tuning Epoch {epoch+1}/{FINE_TUNE_EPOCHS}")

            history_epoch = model.fit(
                X_train_split, y_train_split,
                batch_size=BATCH_SIZE,
                epochs=1,
                validation_data=(X_val, y_val),
                callbacks=fine_tune_callbacks,
                verbose=1
            )

            # 히스토리 저장
            if history2_epochs:
                for key in history_epoch.history:
                    history2_epochs[0].history[key].extend(history_epoch.history[key])
            else:
                history2_epochs.append(history_epoch)

            # EarlyStopping 체크 및 모델 저장
            current_val_loss = history_epoch.history['val_loss'][0]
            current_val_accuracy = history_epoch.history['val_accuracy'][0]

            # 더 나은 모델이면 저장
            if current_val_accuracy > fine_tune_best_accuracy:
                fine_tune_best_accuracy = current_val_accuracy
                fine_tune_best_loss = current_val_loss
                fine_tune_patience_counter = 0
                if safe_save_model(model, MODEL_SAVE_PATH):
                    print(f"✓ 모델 저장 (val_acc: {current_val_accuracy:.4f}, val_loss: {current_val_loss:.4f})")
            else:
                fine_tune_patience_counter += 1
                if fine_tune_patience_counter >= fine_tune_patience:
                    print(f"\nFine-tuning 조기 종료: {fine_tune_patience} 에포크 동안 개선 없음")
                    # 최종 모델 로드
                    loaded_model = safe_load_model(MODEL_SAVE_PATH)
                    if loaded_model is None:
                        loaded_model = safe_load_model(MODEL_SAVE_PATH_KERAS)
                    if loaded_model is not None:
                        model = loaded_model
                        print("✓ 최적 모델 로드 완료")
                    break

        history2 = history2_epochs[0] if history2_epochs else None

    except KeyboardInterrupt:
        print("\n⚠ Fine-tuning 중단됨. 현재까지의 모델을 저장합니다...")
        safe_save_model(model, MODEL_SAVE_PATH)
        if history2_epochs:
            history2 = history2_epochs[0]
    except Exception as e:
        print(f"\n⚠ Fine-tuning 중 오류 발생: {e}")
        print("현재까지의 모델을 저장합니다...")
        if safe_save_model(model, MODEL_SAVE_PATH):
            print("✓ 모델 저장 완료")
        if history2_epochs:
            history2 = history2_epochs[0]

    # 히스토리 병합 (안전하게)
    if history1 is not None and history2 is not None:
        combined_history = {
            'accuracy': history1.history['accuracy'] + history2.history['accuracy'],
            'val_accuracy': history1.history['val_accuracy'] + history2.history['val_accuracy'],
            'loss': history1.history['loss'] + history2.history['loss'],
            'val_loss': history1.history['val_loss'] + history2.history['val_loss']
        }
    elif history1 is not None:
        combined_history = history1.history
    elif history2 is not None:
        combined_history = history2.history
    else:
        # 최소한의 히스토리 생성
        combined_history = {
            'accuracy': [0.3],
            'val_accuracy': [0.3],
            'loss': [1.5],
            'val_loss': [1.5]
        }

    # 최종 평가
    print("\n" + "=" * 60)
    print("최종 평가")
    print("=" * 60)
    val_loss, val_accuracy, val_top_k = model.evaluate(X_val, y_val, verbose=0)
    print(f"검증 정확도: {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")
    print(f"검증 Top-K 정확도: {val_top_k:.4f} ({val_top_k*100:.2f}%)")
    print(f"검증 손실: {val_loss:.4f}")

    # 테스트 데이터 평가 (있는 경우)
    test_accuracy = None
    test_loss = None
    if X_test is not None and len(X_test) > 0:
        print("\n테스트 데이터 평가:")
        test_loss, test_accuracy, test_top_k = model.evaluate(X_test, y_test, verbose=0)
        print(f"테스트 정확도: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
        print(f"테스트 Top-K 정확도: {test_top_k:.4f} ({test_top_k*100:.2f}%)")
        print(f"테스트 손실: {test_loss:.4f}")

    # 학습 히스토리 시각화
    print("\n학습 히스토리 그래프 생성 중...")
    plot_history(combined_history, test_accuracy, test_loss)

    # 모델 저장 (안전하게)
    if safe_save_model(model, MODEL_SAVE_PATH):
        print(f"\n✓ Keras 모델 저장 완료: {MODEL_SAVE_PATH} 및 {MODEL_SAVE_PATH_KERAS}")
    else:
        print("⚠ 모델 저장 실패")

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
