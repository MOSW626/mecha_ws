#!/bin/bash
# 빠른 시작 스크립트 - 가상환경 활성화 및 학습 시작

cd "$(dirname "$0")"

echo "=========================================="
echo "강화학습 학습 시작"
echo "=========================================="

# 가상환경 활성화
source venv/bin/activate

echo "가상환경 활성화 완료"
echo ""

# 학습 시작
echo "학습 시작 중..."
echo "Ctrl+C로 중단할 수 있습니다."
echo ""

python3 train.py --mode train --timesteps 10000

