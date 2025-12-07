#!/usr/bin/env python3
"""
로컬에서 ML 알고리즘을 테스트하는 스크립트
이미지 파일을 입력받아 ML 판단 결과를 시각화하여 보여줍니다.
"""

import cv2
import numpy as np
import os
import sys
import argparse
from pathlib import Path

# linetracing_ml 모듈 import
import linetracing_ml

def test_single_image(image_path, save_debug=False, output_dir="test_output"):
    """단일 이미지에 대해 ML 판단을 수행하고 결과를 시각화"""

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

    # ML 판단
    result = linetracing_ml.judge_ml(img_rgb)

    # 상세 정보를 얻기 위해 직접 추론 수행
    debug_info = get_ml_debug_info(img_rgb)

    print(f"\n📊 판단 결과:")
    print(f"  ML 결과 (mapped): {result}")
    if debug_info:
        print(f"  원본 레이블: {debug_info['original_label']}")
        print(f"  신뢰도: {debug_info['confidence']:.3f}")
        print(f"  추론 시간: {debug_info['inference_time']:.1f} ms")
        print(f"\n  확률 분포:")
        for i, (label, prob) in enumerate(zip(linetracing_ml.labels, debug_info['probabilities'])):
            marker = "✓" if i == debug_info['pred_id'] else " "
            print(f"    {marker} {label:10s}: {prob:.4f} ({prob*100:5.1f}%)")

    # 시각화 이미지 생성
    vis_img = create_visualization(img_bgr, debug_info, result)

    if save_debug:
        os.makedirs(output_dir, exist_ok=True)
        base_name = Path(image_path).stem
        output_path = os.path.join(output_dir, f"{base_name}_ml_result.jpg")
        cv2.imwrite(output_path, vis_img)
        print(f"\n💾 결과 이미지 저장: {output_path}")

    return vis_img, result, debug_info

def get_ml_debug_info(frame_rgb):
    """ML 모델의 상세 정보를 얻기 위해 직접 추론 수행"""
    if linetracing_ml.interpreter is None or linetracing_ml.inp is None or linetracing_ml.out is None:
        return None

    try:
        import time

        # prepare input for model
        x = linetracing_ml.preprocess_frame_for_model(frame_rgb)

        # set tensor and run
        t0 = time.time()
        linetracing_ml.interpreter.set_tensor(linetracing_ml.inp["index"], x)
        linetracing_ml.interpreter.invoke()
        dt = (time.time() - t0) * 1000  # ms

        probs = linetracing_ml.interpreter.get_tensor(linetracing_ml.out["index"])[0]
        pred_id = int(np.argmax(probs))
        pred_label = linetracing_ml.labels[pred_id]
        confidence = float(probs[pred_id])

        return {
            'probabilities': probs,
            'pred_id': pred_id,
            'original_label': pred_label,
            'confidence': confidence,
            'inference_time': dt,
            'model_input': x[0]  # 전처리된 입력 이미지
        }
    except Exception as e:
        print(f"⚠ ML debug info error: {e}")
        import traceback
        traceback.print_exc()
        return None

def create_visualization(img_bgr, debug_info, result):
    """디버그 정보를 시각화한 이미지 생성"""
    vis_img = img_bgr.copy()
    h, w = vis_img.shape[:2]

    # 원본 이미지 표시
    vis_img_resized = cv2.resize(vis_img, (640, 480))
    h_vis, w_vis = vis_img_resized.shape[:2]

    # 텍스트 정보 표시
    info_y = 20
    line_height = 25

    # 메인 결과
    cv2.putText(vis_img_resized, f"ML: {result}", (10, info_y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    if debug_info:
        info_y += line_height + 5
        cv2.putText(vis_img_resized, f"Original: {debug_info['original_label']}", (10, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        info_y += line_height
        cv2.putText(vis_img_resized, f"Confidence: {debug_info['confidence']:.3f} ({debug_info['confidence']*100:.1f}%)",
                   (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        info_y += line_height
        cv2.putText(vis_img_resized, f"Inference: {debug_info['inference_time']:.1f} ms",
                   (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # 확률 분포 표시 (우측에)
        prob_x = w_vis - 200
        prob_y = 20
        cv2.putText(vis_img_resized, "Probabilities:", (prob_x, prob_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        prob_y += line_height
        for i, (label, prob) in enumerate(zip(linetracing_ml.labels, debug_info['probabilities'])):
            if i == debug_info['pred_id']:
                color = (0, 255, 0)  # 초록색 (예측된 레이블)
                marker = "> "
            else:
                color = (200, 200, 200)  # 회색
                marker = "  "

            prob_text = f"{marker}{label:8s}: {prob*100:5.1f}%"
            cv2.putText(vis_img_resized, prob_text, (prob_x, prob_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            prob_y += line_height - 5

        # 모델 입력 이미지 표시 (우측 하단)
        if 'model_input' in debug_info:
            model_input = debug_info['model_input']
            # Convert to displayable format
            if model_input.dtype != np.uint8:
                model_input_disp = (model_input * 255).astype(np.uint8)
            else:
                model_input_disp = model_input.copy()

            # Convert to BGR if needed
            if len(model_input_disp.shape) == 3:
                if model_input_disp.shape[2] == 3:
                    model_input_bgr = cv2.cvtColor(model_input_disp, cv2.COLOR_RGB2BGR)
                elif model_input_disp.shape[2] == 1:
                    model_input_bgr = cv2.cvtColor(model_input_disp[..., 0], cv2.COLOR_GRAY2BGR)
                else:
                    model_input_bgr = model_input_disp[..., :3]
            else:
                model_input_bgr = cv2.cvtColor(model_input_disp, cv2.COLOR_GRAY2BGR)

            # Resize to fit in corner
            input_h, input_w = model_input_bgr.shape[:2]
            display_size = 150
            scale = min(display_size / input_w, display_size / input_h)
            new_w = int(input_w * scale)
            new_h = int(input_h * scale)
            model_input_resized = cv2.resize(model_input_bgr, (new_w, new_h))

            # Place in bottom right corner
            y_offset = h_vis - new_h - 10
            x_offset = w_vis - new_w - 10
            vis_img_resized[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = model_input_resized

            # Border
            cv2.rectangle(vis_img_resized, (x_offset-1, y_offset-1),
                         (x_offset+new_w, y_offset+new_h), (0, 255, 255), 2)
            cv2.putText(vis_img_resized, "Model Input", (x_offset, y_offset-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    else:
        info_y += line_height
        cv2.putText(vis_img_resized, "Debug info unavailable", (10, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    return vis_img_resized

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
        if '_debug' in str(img_path) or '_ml_result' in str(img_path):
            continue  # 디버그 이미지와 결과 이미지는 건너뛰기

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
    parser = argparse.ArgumentParser(description='로컬에서 ML 알고리즘 테스트')
    parser.add_argument('input', type=str, help='테스트할 이미지 파일 또는 디렉토리 경로')
    parser.add_argument('--save', action='store_true', help='결과 이미지 저장')
    parser.add_argument('--output', type=str, default='test_output', help='결과 이미지 저장 디렉토리 (기본: test_output)')
    parser.add_argument('--show', action='store_true', help='결과 이미지 표시 (GUI 필요)')
    parser.add_argument('--model', type=str, default=None, help='모델 파일 경로 (기본: linetracing_ml.py의 model_path 사용)')

    args = parser.parse_args()

    # 모델 경로 설정 (옵션이 제공된 경우)
    if args.model:
        linetracing_ml.model_path = args.model
        print(f"📦 사용할 모델: {args.model}")

    # ML 모듈 초기화
    print("🔧 ML 모델 초기화 중...")
    if not linetracing_ml.init_ml():
        print("❌ ML 모델 초기화 실패!")
        print(f"   모델 경로: {linetracing_ml.model_path}")
        sys.exit(1)
    print("✓ ML 모델 초기화 완료")

    if os.path.isfile(args.input):
        # 단일 파일
        vis_img, result, debug_info = test_single_image(args.input, args.save, args.output)
        if args.show and vis_img is not None:
            cv2.imshow('ML Test Result', vis_img)
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

