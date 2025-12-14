#!/bin/bash
# 강화학습 학습 환경 설치 스크립트

echo "=========================================="
echo "강화학습 학습 환경 설치"
echo "=========================================="

# Python 버전 확인
echo "Python 버전 확인 중..."
python3 --version

# pip 업그레이드
echo ""
echo "pip 업그레이드 중..."
python3 -m pip install --upgrade pip

# 의존성 설치
echo ""
echo "의존성 설치 중..."
python3 -m pip install -r requirements.txt

# 설치 확인
echo ""
echo "=========================================="
echo "설치 확인"
echo "=========================================="

python3 -c "import torch; print('✓ PyTorch:', torch.__version__); print('  CUDA 사용 가능:', torch.cuda.is_available())" 2>&1
python3 -c "import stable_baselines3; print('✓ Stable-Baselines3:', stable_baselines3.__version__)" 2>&1
python3 -c "import gymnasium; print('✓ Gymnasium:', gymnasium.__version__)" 2>&1
python3 -c "import onnx; print('✓ ONNX:', onnx.__version__)" 2>&1
python3 -c "import onnxruntime; print('✓ ONNX Runtime:', onnxruntime.__version__)" 2>&1

echo ""
echo "=========================================="
echo "설치 완료!"
echo "=========================================="
echo ""
echo "학습 시작:"
echo "  python3 train.py --mode train --timesteps 100000"
echo ""
echo "간단한 테스트 (10,000 타임스텝):"
echo "  python3 train.py --mode train --timesteps 10000"

