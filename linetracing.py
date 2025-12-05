#!/usr/bin/env python3
# linetracing_ml.py 와 linetracing_cv.py 을 같이 사용.
# cv 쪽과 ml의 판단 기준을 정하는 변수 설정.
# 기본값은 cv가 8할로 높게 설정.

import sys
import os

# linetracing_cv와 linetracing_ml 모듈 import
try:
    import linetracing_cv
    import linetracing_ml
except ImportError as e:
    print(f"모듈 import 오류: {e}")
    sys.exit(1)

# ==================== 판단 기준 설정 ====================
CV_WEIGHT = 0.8  # CV 8할
ML_WEIGHT = 0.2  # ML 2할

def main():
    """하이브리드 라인트레이싱 실행"""
    print("=" * 60)
    print("하이브리드 라인트레이싱 (CV + ML)")
    print(f"CV 가중치: {CV_WEIGHT}, ML 가중치: {ML_WEIGHT}")
    print("=" * 60)
    print("주의: 이 파일은 linetracing_cv.py와 linetracing_ml.py를")
    print("통합하여 실행하는 기능을 제공합니다.")
    print("=" * 60)
    print("\n현재는 개별 파일을 실행하거나 main.py를 사용하세요.")
    print("linetracing_cv.py: OpenCV 기반 라인트레이싱")
    print("linetracing_ml.py: ML 기반 라인트레이싱")
    print("main.py: 통합 모드 (라인트레이싱 + 주행)")

if __name__ == "__main__":
    main()
