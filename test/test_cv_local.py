#!/usr/bin/env python3
"""
로컬에서 CV 알고리즘을 테스트하는 스크립트
이미지 파일을 입력받아 CV 판단 결과를 시각화하여 보여줍니다.
"""

import cv2
import numpy as np
import os
import sys
import argparse
from pathlib import Path

# linetracing_cv 모듈 import
import linetracing_cv

def test_single_image(image_path, save_debug=False, output_dir="test_output"):
    """단일 이미지에 대해 CV 판단을 수행하고 결과를 시각화"""

    # 이미지 로드
    if not os.path.exists(image_path):
        print(f"❌ 이미지 파일을 찾을 수 없습니다: {image_path}")
        return None

    # 이미지 로드 (BGR로 로드됨)
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print(f"❌ 이미지를 로드할 수 없습니다: {image_path}")
        return None

    # BGR을 RGB로 변환 (카메라에서 받은 것처럼)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    print(f"\n{'='*60}")
    print(f"테스트 이미지: {image_path}")
    print(f"{'='*60}")

    # CV 판단 (디버그 정보 포함)
    result, debug_info = linetracing_cv.judge_cv(img_rgb, return_debug=True)

    print(f"\n📊 판단 결과:")
    print(f"  CV 결과: {result}")
    if debug_info.get('traffic_light'):
        print(f"  신호등: {debug_info['traffic_light']}")
    if debug_info.get('bottom_center') is not None:
        print(f"  하단 중심: {debug_info['bottom_center']:.1f}px")
    else:
        print(f"  하단 중심: None")
    if debug_info.get('top_center') is not None:
        print(f"  상단 중심: {debug_info['top_center']:.1f}px")
    else:
        print(f"  상단 중심: None")
    if debug_info.get('line_angle') is not None:
        print(f"  라인 각도: {debug_info['line_angle']:.1f}deg")
    else:
        print(f"  라인 각도: None")

    # 시각화 이미지 생성
    vis_img = create_visualization(img_bgr, debug_info, result)

    if save_debug:
        os.makedirs(output_dir, exist_ok=True)
        base_name = Path(image_path).stem
        output_path = os.path.join(output_dir, f"{base_name}_cv_result.jpg")
        cv2.imwrite(output_path, vis_img)
        print(f"\n💾 결과 이미지 저장: {output_path}")

    return vis_img, result, debug_info

def create_visualization(img_bgr, debug_info, result):
    """디버그 정보를 시각화한 이미지 생성"""
    vis_img = img_bgr.copy()
    h, w = vis_img.shape[:2]

    # 원본 이미지를 CV 처리 크기로 리사이즈
    img_resized = cv2.resize(vis_img, (linetracing_cv.IMG_WIDTH, linetracing_cv.IMG_HEIGHT))
    h_resized, w_resized = img_resized.shape[:2]

    # ROI 표시
    roi_top = int(h_resized * linetracing_cv.ROI_TOP)
    roi_bottom = h_resized
    cv2.rectangle(img_resized, (0, roi_top), (w_resized, roi_bottom), (0, 255, 0), 2)

    # 라인 중심 표시
    if debug_info.get('bottom_center') is not None:
        bottom_y = int(roi_top + (roi_bottom - roi_top) * 0.8)
        bottom_x = int(debug_info['bottom_center'])
        cv2.circle(img_resized, (bottom_x, bottom_y), 8, (255, 0, 0), -1)
        cv2.line(img_resized, (w_resized // 2, bottom_y), (bottom_x, bottom_y), (0, 0, 255), 2)
        cv2.putText(img_resized, f"B:{bottom_x}", (bottom_x + 10, bottom_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    if debug_info.get('top_center') is not None:
        top_y = int(roi_top + (roi_bottom - roi_top) * 0.2)
        top_x = int(debug_info['top_center'])
        cv2.circle(img_resized, (top_x, top_y), 8, (0, 255, 255), -1)
        cv2.putText(img_resized, f"T:{top_x}", (top_x + 10, top_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # 이미지 중심선 표시
    cv2.line(img_resized, (w_resized // 2, 0), (w_resized // 2, h_resized), (0, 255, 0), 1)

    # 이진화 이미지 표시
    if debug_info.get('binary') is not None:
        binary = debug_info['binary']
        binary_colored = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        binary_h = int(h_resized * 0.3)
        binary_w = int(binary_colored.shape[1] * binary_h / binary_colored.shape[0])
        binary_resized = cv2.resize(binary_colored, (binary_w, binary_h))
        # 우측 상단에 배치
        x_offset = w_resized - binary_w
        img_resized[0:binary_h, x_offset:w_resized] = binary_resized[:, :min(binary_w, w_resized - x_offset)]
        cv2.rectangle(img_resized, (x_offset, 0), (w_resized, binary_h), (255, 255, 0), 2)

    # 텍스트 정보 표시
    info_y = 20
    cv2.putText(img_resized, f"CV: {result}", (10, info_y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    if debug_info.get('bottom_center') is not None:
        cv2.putText(img_resized, f"Bottom: {debug_info['bottom_center']:.1f}", (10, info_y + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        center_x = linetracing_cv.IMG_WIDTH / 2
        center_error = abs(debug_info['bottom_center'] - center_x)
        threshold = linetracing_cv.IMG_WIDTH * 0.15
        cv2.putText(img_resized, f"Error: {center_error:.1f} (thresh: {threshold:.1f})", (10, info_y + 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    else:
        cv2.putText(img_resized, "Bottom: None", (10, info_y + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    if debug_info.get('line_angle') is not None:
        cv2.putText(img_resized, f"Angle: {debug_info['line_angle']:.1f}deg", (10, info_y + 75),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    else:
        cv2.putText(img_resized, "Angle: None", (10, info_y + 75),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # 판단 로직 정보 표시
    if debug_info.get('bottom_center') is not None:
        center_x = linetracing_cv.IMG_WIDTH / 2
        center_error = abs(debug_info['bottom_center'] - center_x)
        threshold = linetracing_cv.IMG_WIDTH * 0.15
        angle = debug_info.get('line_angle', 0)

        # 판단 조건 표시
        condition1 = center_error < threshold
        condition2 = abs(angle) < 10
        cv2.putText(img_resized, f"Cond1 (err<{threshold:.1f}): {condition1}", (10, info_y + 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0) if condition1 else (0, 0, 255), 1)
        cv2.putText(img_resized, f"Cond2 (|angle|<10): {condition2}", (10, info_y + 120),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0) if condition2 else (0, 0, 255), 1)

    return img_resized

def test_directory(image_dir, save_debug=False, output_dir="test_output"):
    """디렉토리 내 모든 이미지에 대해 테스트"""
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []

    for ext in image_extensions:
        image_files.extend(Path(image_dir).glob(f"*{ext}"))
        image_files.extend(Path(image_dir).glob(f"*{ext.upper()}"))

    if not image_files:
        print(f"❌ 이미지 파일을 찾을 수 없습니다: {image_dir}")
        return

    image_files = sorted(image_files)
    print(f"\n📁 {len(image_files)}개의 이미지 파일을 찾았습니다.")

    results_summary = {}
    for img_path in image_files:
        if '_debug' in str(img_path):
            continue  # 디버그 이미지는 건너뛰기

        try:
            vis_img, result, debug_info = test_single_image(str(img_path), save_debug, output_dir)
            if result:
                results_summary[result] = results_summary.get(result, 0) + 1
        except Exception as e:
            print(f"❌ 오류 발생 ({img_path}): {e}")
            import traceback
            traceback.print_exc()

    # 결과 요약
    print(f"\n{'='*60}")
    print("📊 결과 요약:")
    print(f"{'='*60}")
    for result, count in sorted(results_summary.items()):
        print(f"  {result}: {count}개")
    print(f"{'='*60}")

def main():
    parser = argparse.ArgumentParser(description='로컬에서 CV 알고리즘 테스트')
    parser.add_argument('input', type=str, help='테스트할 이미지 파일 또는 디렉토리 경로')
    parser.add_argument('--save', action='store_true', help='결과 이미지 저장')
    parser.add_argument('--output', type=str, default='test_output', help='결과 이미지 저장 디렉토리 (기본: test_output)')
    parser.add_argument('--show', action='store_true', help='결과 이미지 표시 (GUI 필요)')

    args = parser.parse_args()

    # CV 모듈 초기화
    linetracing_cv.init_cv()

    if os.path.isfile(args.input):
        # 단일 파일
        vis_img, result, debug_info = test_single_image(args.input, args.save, args.output)
        if args.show and vis_img is not None:
            cv2.imshow('CV Test Result', vis_img)
            print("\n아무 키나 누르면 종료됩니다...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    elif os.path.isdir(args.input):
        # 디렉토리
        test_directory(args.input, args.save, args.output)
    else:
        print(f"❌ 파일 또는 디렉토리를 찾을 수 없습니다: {args.input}")
        sys.exit(1)

if __name__ == "__main__":
    main()

