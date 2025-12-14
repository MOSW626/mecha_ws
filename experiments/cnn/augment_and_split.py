#!/usr/bin/env python3
"""
데이터 증강 및 train/test 분할 스크립트
- 데이터 증강: 회전, 이동, 밝기, 대비, 노이즈 등 적용
- train/test 분할: 80/20 비율로 분할

사용법:
    python3 augment_and_split.py [--data-dir data] [--output-dir augmented_data] [--augment-ratio 3]
"""

import os
import sys
import argparse
import shutil
import cv2
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import random

# 클래스 정의
CLASSES = ['forward', 'green', 'left', 'non', 'red', 'right']

def augment_image(img, augment_type='random'):
    """
    이미지 증강 적용

    Args:
        img: 입력 이미지 (BGR)
        augment_type: 증강 타입 ('random', 'rotation', 'brightness', 'contrast', 'noise', 'blur', 'shift')

    Returns:
        증강된 이미지
    """
    if augment_type == 'random':
        augment_type = random.choice(['rotation', 'brightness', 'contrast', 'noise', 'blur', 'shift', 'flip'])

    h, w = img.shape[:2]

    if augment_type == 'rotation':
        # 작은 회전 (-2도 ~ +2도)
        angle = random.uniform(-2, 2)
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

    elif augment_type == 'brightness':
        # 밝기 조정 (-40 ~ +40)
        brightness = random.randint(-40, 40)
        img = cv2.convertScaleAbs(img, alpha=1, beta=brightness)

    elif augment_type == 'contrast':
        # 대비 조정 (0.6 ~ 1.4)
        contrast = random.uniform(0.6, 1.4)
        img = cv2.convertScaleAbs(img, alpha=contrast, beta=0)

    elif augment_type == 'noise':
        # 가우시안 노이즈 추가
        noise = np.random.normal(0, 5, img.shape).astype(np.uint8)
        img = cv2.add(img, noise)

    elif augment_type == 'blur':
        # 약간의 블러 (모션 블러 효과)
        kernel_size = random.choice([3, 5])
        img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

    elif augment_type == 'shift':
        # 작은 이동 (-5 ~ +5 픽셀)
        tx = random.randint(-5, 5)
        ty = random.randint(-5, 5)
        M = np.float32([[1, 0, tx], [0, 1, ty]])
        img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

    elif augment_type == 'flip':
        # 수평 뒤집기 (일부 클래스에만 적용)
        img = cv2.flip(img, 1)

    elif augment_type == 'gamma':
        # 감마 보정 (밝기 조정의 다른 방법)
        gamma = random.uniform(0.8, 1.2)
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        img = cv2.LUT(img, table)

    elif augment_type == 'saturation':
        # 채도 조정
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv[:, :, 1] = cv2.multiply(hsv[:, :, 1], random.uniform(0.8, 1.2))
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return img

def augment_dataset(data_dir, output_dir, augment_ratio=3, train_ratio=0.9):
    """
    데이터셋 증강 및 train/test 분할

    Args:
        data_dir: 원본 데이터 디렉토리
        output_dir: 출력 디렉토리
        augment_ratio: 원본 이미지당 생성할 증강 이미지 수
        train_ratio: 학습 데이터 비율 (기본 0.9 = 90%)
    """
    print("=" * 60)
    print("데이터 증강 및 train/test 분할")
    print("=" * 60)
    print(f"원본 데이터: {data_dir}")
    print(f"출력 디렉토리: {output_dir}")
    print(f"증강 비율: {augment_ratio}배")
    print(f"Train/Test 비율: {train_ratio*100:.0f}% / {(1-train_ratio)*100:.0f}%")
    print("=" * 60)

    # 출력 디렉토리 생성
    train_dir = os.path.join(output_dir, 'training')
    test_dir = os.path.join(output_dir, 'testing')

    for split_dir in [train_dir, test_dir]:
        if not os.path.exists(split_dir):
            os.makedirs(split_dir)
        for class_name in CLASSES:
            class_dir = os.path.join(split_dir, class_name)
            if not os.path.exists(class_dir):
                os.makedirs(class_dir)

    # 각 클래스별로 처리
    total_original = 0
    total_augmented = 0
    total_train = 0
    total_test = 0

    for class_name in CLASSES:
        print(f"\n[{class_name}] 처리 중...")

        class_dir = os.path.join(data_dir, class_name)
        if not os.path.exists(class_dir):
            print(f"  ⚠ {class_name} 폴더가 없습니다. 건너뜁니다.")
            continue

        # 이미지 파일 목록
        image_files = [f for f in os.listdir(class_dir)
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        if len(image_files) == 0:
            print(f"  ⚠ {class_name}에 이미지가 없습니다. 건너뜁니다.")
            continue

        print(f"  원본 이미지: {len(image_files)}개")
        total_original += len(image_files)

        # train/test 분할
        train_files, test_files = train_test_split(
            image_files,
            test_size=1-train_ratio,
            random_state=42
        )

        print(f"  Train: {len(train_files)}개, Test: {len(test_files)}개")

        # 원본 이미지 복사 및 증강
        all_images = []

        # Train 데이터 처리
        for img_file in train_files:
            src_path = os.path.join(class_dir, img_file)
            dst_path = os.path.join(train_dir, class_name, img_file)

            # 원본 복사
            shutil.copy2(src_path, dst_path)
            all_images.append(('train', img_file, src_path))

        # Test 데이터 처리 (증강 없이 원본만)
        for img_file in test_files:
            src_path = os.path.join(class_dir, img_file)
            dst_path = os.path.join(test_dir, class_name, img_file)

            # 원본 복사
            shutil.copy2(src_path, dst_path)

        # Train 데이터 증강
        augmented_count = 0
        for img_file in train_files:
            src_path = os.path.join(class_dir, img_file)
            img = cv2.imread(src_path)

            if img is None:
                continue

            # 증강 이미지 생성
            for i in range(augment_ratio):
                # 여러 증강 타입 조합
                augmented_img = img.copy()

                # 랜덤하게 1-2개의 증강 적용
                num_augments = random.randint(1, 2)
                augment_types = random.sample(
                    ['rotation', 'brightness', 'contrast', 'noise', 'blur', 'shift', 'gamma', 'saturation'],
                    num_augments
                )

                for aug_type in augment_types:
                    augmented_img = augment_image(augmented_img, aug_type)

                # 저장
                base_name, ext = os.path.splitext(img_file)
                aug_filename = f"{base_name}_aug_{i+1}{ext}"
                aug_path = os.path.join(train_dir, class_name, aug_filename)
                cv2.imwrite(aug_path, augmented_img)
                augmented_count += 1

        print(f"  증강 이미지 생성: {augmented_count}개")
        total_augmented += augmented_count
        total_train += len(train_files) + augmented_count
        total_test += len(test_files)

    # 최종 통계
    print("\n" + "=" * 60)
    print("완료 통계")
    print("=" * 60)
    print(f"원본 이미지: {total_original}개")
    print(f"증강 이미지: {total_augmented}개")
    print(f"총 Train 데이터: {total_train}개")
    print(f"총 Test 데이터: {total_test}개")
    print(f"총 데이터: {total_train + total_test}개")
    print("=" * 60)

    print(f"\n✓ 데이터 증강 및 분할 완료!")
    print(f"  - Training: {train_dir}")
    print(f"  - Testing: {test_dir}")

def main():
    parser = argparse.ArgumentParser(description='데이터 증강 및 train/test 분할')
    parser.add_argument('--data-dir', type=str, default='data',
                       help='원본 데이터 디렉토리 (기본값: data)')
    parser.add_argument('--output-dir', type=str, default='augmented_data',
                       help='출력 디렉토리 (기본값: augmented_data)')
    parser.add_argument('--augment-ratio', type=int, default=3,
                       help='원본 이미지당 생성할 증강 이미지 수 (기본값: 3)')
    parser.add_argument('--train-ratio', type=float, default=0.8,
                       help='학습 데이터 비율 (기본값: 0.8 = 80%%)')

    args = parser.parse_args()

    if not os.path.exists(args.data_dir):
        print(f"✗ 오류: 데이터 디렉토리를 찾을 수 없습니다: {args.data_dir}")
        return

    augment_dataset(
        args.data_dir,
        args.output_dir,
        args.augment_ratio,
        args.train_ratio
    )

if __name__ == "__main__":
    main()
