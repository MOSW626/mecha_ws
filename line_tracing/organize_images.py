#!/usr/bin/env python3
"""
폰으로 찍은 이미지를 클래스별로 분류하는 스크립트
이미지 파일을 보면서 키보드로 클래스를 지정하여 분류합니다.

사용법:
    1. 폰으로 찍은 이미지들을 하나의 폴더에 모으기 (예: photos/)
    2. python3 organize_images.py --input photos/
    3. 이미지를 보면서 키보드로 클래스 지정
    4. 자동으로 data/ 클래스별 폴더로 이동
"""

import os
import sys
import argparse
import cv2
import shutil
from pathlib import Path

# 클래스 정의 (camera_main_gpt.py와 동일한 순서)
CLASSES = {
    'g': 'green',
    'l': 'left',
    'm': 'middle',
    'n': 'noline',
    's': 'red',
    'r': 'right'
}

# 데이터 저장 경로
DATA_DIR = "data"
IMG_SIZE = 224  # 모델 입력 크기

def setup_directories():
    """클래스별 디렉토리 생성"""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    for class_name in CLASSES.values():
        class_dir = os.path.join(DATA_DIR, class_name)
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)
            print(f"✓ 디렉토리 생성: {class_dir}")

def get_image_files(input_dir):
    """이미지 파일 목록 가져오기"""
    image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
    image_files = []

    for ext in image_extensions:
        image_files.extend(Path(input_dir).glob(f'*{ext}'))

    return sorted(image_files)

def resize_and_save_image(src_path, dst_path):
    """이미지를 리사이즈하여 저장"""
    img = cv2.imread(str(src_path))
    if img is None:
        return False

    # 리사이즈
    resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(dst_path), resized)
    return True

def main():
    parser = argparse.ArgumentParser(description='폰으로 찍은 이미지를 클래스별로 분류')
    parser.add_argument('--input', '-i', type=str, required=True,
                       help='이미지가 있는 폴더 경로')
    parser.add_argument('--move', action='store_true',
                       help='원본 이미지를 이동 (기본값: 복사)')

    args = parser.parse_args()

    input_dir = args.input
    if not os.path.exists(input_dir):
        print(f"✗ 오류: 입력 디렉토리를 찾을 수 없습니다: {input_dir}")
        return

    print("=" * 60)
    print("이미지 분류 도구")
    print("=" * 60)
    print(f"입력 폴더: {input_dir}")
    print(f"모드: {'이동' if args.move else '복사'}")
    print("\n키보드 입력:")
    print("  'g' - green (초록불)")
    print("  'l' - left (좌회전)")
    print("  'm' - middle (중앙/직진)")
    print("  'n' - noline (라인 없음)")
    print("  's' - red (빨간불/정지)")
    print("  'r' - right (우회전)")
    print("  'd' - 삭제 (이미지 건너뛰기)")
    print("  'q' - 종료")
    print("=" * 60)

    # 디렉토리 설정
    setup_directories()

    # 이미지 파일 목록
    image_files = get_image_files(input_dir)

    if len(image_files) == 0:
        print(f"✗ 오류: {input_dir}에 이미지 파일이 없습니다.")
        return

    print(f"\n총 {len(image_files)}개 이미지 파일 발견")
    print("이미지를 하나씩 보여드립니다. 키보드로 클래스를 선택하세요.\n")

    # 통계
    counts = {class_name: 0 for class_name in CLASSES.values()}
    skipped = 0
    processed = 0

    try:
        for idx, img_path in enumerate(image_files, 1):
            # 이미지 로드
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"⚠ 이미지 로드 실패: {img_path}")
                continue

            # 화면에 표시
            display_img = img.copy()

            # 정보 표시
            info_text = f"[{idx}/{len(image_files)}] {img_path.name}"
            cv2.putText(display_img, info_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

            cv2.putText(display_img, "Press key to classify (g/l/m/n/s/r/d/q)",
                       (10, display_img.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

            # 창 크기 조정 (너무 크면 리사이즈)
            max_width = 1200
            if display_img.shape[1] > max_width:
                scale = max_width / display_img.shape[1]
                new_width = int(display_img.shape[1] * scale)
                new_height = int(display_img.shape[0] * scale)
                display_img = cv2.resize(display_img, (new_width, new_height))

            cv2.imshow("Image Classification", display_img)

            # 키 입력 대기
            while True:
                key = cv2.waitKey(0) & 0xFF

                if key == ord('q'):
                    print("\n종료합니다...")
                    return

                elif key == ord('d'):
                    print(f"[{idx}/{len(image_files)}] 건너뛰기: {img_path.name}")
                    skipped += 1
                    break

                elif chr(key).lower() in CLASSES:
                    class_name = CLASSES[chr(key).lower()]

                    # 대상 경로
                    filename = img_path.name
                    dst_dir = os.path.join(DATA_DIR, class_name)
                    dst_path = os.path.join(dst_dir, filename)

                    # 파일명 중복 처리
                    counter = 1
                    while os.path.exists(dst_path):
                        name, ext = os.path.splitext(filename)
                        dst_path = os.path.join(dst_dir, f"{name}_{counter}{ext}")
                        counter += 1

                    # 이미지 리사이즈 및 저장
                    if resize_and_save_image(img_path, dst_path):
                        counts[class_name] += 1
                        processed += 1
                        print(f"[{idx}/{len(image_files)}] ✓ {class_name}: {img_path.name} -> {dst_path}")

                        # 원본 이동/삭제 옵션
                        if args.move:
                            os.remove(img_path)
                    else:
                        print(f"⚠ 저장 실패: {img_path.name}")

                    break

            cv2.destroyAllWindows()

    except KeyboardInterrupt:
        print("\n키보드 인터럽트로 종료합니다...")
    finally:
        cv2.destroyAllWindows()

        # 최종 통계 출력
        print("\n" + "=" * 60)
        print("분류 완료 통계:")
        print("=" * 60)
        total = 0
        for class_name, count in counts.items():
            print(f"  {class_name:10s}: {count:4d}개")
            total += count
        print(f"  {'Processed':10s}: {processed:4d}개")
        print(f"  {'Skipped':10s}: {skipped:4d}개")
        print(f"  {'Total':10s}: {total:4d}개")
        print("=" * 60)

if __name__ == "__main__":
    main()

