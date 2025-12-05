#!/bin/bash
# 코드와 모델 파일만 커밋하고 푸시하는 스크립트
# frame 이미지는 자동으로 제외됩니다

cd /Users/an-yeonsu/Documents/GitHub/mecha_ws

echo "=== 1. .gitignore 파일 커밋 ==="
git add .gitignore cnn/.gitignore
git commit -m "Exclude frame images and .DS_Store from git" 2>&1

echo ""
echo "=== 2. 코드 파일 추가 ==="
git add cnn/*.py cnn/*.md cnn/*.txt 2>/dev/null
git add *.py 2>/dev/null
git add line_tracing/*.py line_tracing/*.md 2>/dev/null || true
git add simulation/*.py simulation/*.md simulation/*.sh 2>/dev/null || true
git add Yeonsu_track/*.py Yeonsu_track/*.md 2>/dev/null || true

echo ""
echo "=== 3. 모델 파일 확인 ==="
if [ -f "cnn/cnn_model.keras" ]; then
    echo "cnn_model.keras 발견 - 추가 중..."
    git add cnn/cnn_model.keras 2>/dev/null
fi
if [ -f "cnn/cnn_model.h5" ]; then
    echo "cnn_model.h5 발견 - 추가 중..."
    git add cnn/cnn_model.h5 2>/dev/null
fi
if [ -f "cnn/cnn_model.tflite" ]; then
    echo "cnn_model.tflite 발견 - 추가 중..."
    git add cnn/cnn_model.tflite 2>/dev/null
fi

echo ""
echo "=== 4. 스테이징된 파일 확인 ==="
git status --short | head -20

echo ""
echo "=== 5. 커밋 ==="
git commit -m "Update code and models, exclude frame images" 2>&1

echo ""
echo "=== 6. 푸시 시도 ==="
git push origin main 2>&1

echo ""
echo "완료!"

