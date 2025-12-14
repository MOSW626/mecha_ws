#!/bin/bash
# KAIST ME203 - Mechatronics System Design
# Raspberry Pi Setup Script
# 자동으로 종료되면 안 되므로 에러 발생 시에도 계속 진행

set -e  # 에러 발생 시 스크립트 중단

echo "=========================================="
echo "Setting up Raspberry Pi for mecha_ws"
echo "=========================================="

# 1. 시스템 패키지 업데이트 및 설치
echo "[1/6] Updating system packages..."
sudo apt update
sudo apt upgrade -y

echo "[2/6] Installing system dependencies..."
sudo apt install -y \
    python3 \
    python3-pip \
    python3-venv \
    python3-opencv \
    python3-picamera2 \
    python3-gpiozero \
    git

# 2. 가상환경 생성
echo "[3/6] Creating Python virtual environment..."
VENV_PATH="$HOME/venvs/mecha"
if [ -d "$VENV_PATH" ]; then
    echo "Virtual environment already exists. Skipping..."
else
    python3 -m venv "$VENV_PATH" --system-site-packages
    echo "Virtual environment created at $VENV_PATH"
fi

# 3. 가상환경 활성화 및 패키지 설치
echo "[4/6] Installing Python dependencies..."
source "$VENV_PATH/bin/activate"

# pip 업그레이드
pip install --upgrade pip

# 필수 패키지 설치
pip install \
    tflite-runtime \
    numpy \
    opencv-python-headless

# 가상환경 활성화를 .bashrc에 추가 (중복 방지)
if ! grep -q "source $VENV_PATH/bin/activate" ~/.bashrc; then
    echo "" >> ~/.bashrc
    echo "# mecha_ws virtual environment" >> ~/.bashrc
    echo "source $VENV_PATH/bin/activate" >> ~/.bashrc
    echo "Added virtual environment activation to ~/.bashrc"
else
    echo "Virtual environment already in ~/.bashrc"
fi

# 4. 의존성 체크
echo "[5/6] Checking dependencies..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SCRIPT_DIR/Check_dependencies_pi.py" ]; then
    python3 "$SCRIPT_DIR/Check_dependencies_pi.py"
else
    echo "Warning: Check_dependencies_pi.py not found. Skipping..."
fi

# 5. 카메라 체크
echo "[6/6] Checking camera..."
if [ -f "$SCRIPT_DIR/cameracheck.py" ]; then
    python3 "$SCRIPT_DIR/cameracheck.py"
else
    echo "Warning: cameracheck.py not found. Skipping..."
fi

# 6. Git 설정 (선택사항 - 주석 처리)
# echo "Setting up Git..."
# git config --global user.name "Your Name"
# git config --global user.email "your.email@example.com"

echo ""
echo "=========================================="
echo "Setup completed successfully!"
echo "=========================================="
echo ""
echo "To activate the virtual environment, run:"
echo "  source $VENV_PATH/bin/activate"
echo ""
echo "Or restart your terminal (it will auto-activate)."
echo ""
echo "To run the main program:"
echo "  cd src"
echo "  python3 main.py"
echo ""
