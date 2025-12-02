# 라인트레이싱 프로젝트

라인트레이싱을 위한 머신러닝 모델 학습 및 실행 프로젝트입니다.

**참고**: ML 없이 컴퓨터 비전만 사용하는 방법도 제공됩니다. `line_tracing_cv.py`와 `README_CV.md`를 참조하세요.

## 프로젝트 구조

```
line_tracing/
├── collect_data.py      # 데이터 수집 스크립트
├── train_model.py       # 모델 학습 스크립트
├── run_line_tracing.py  # 라인트레이싱 실행 스크립트
├── data/                # 수집된 이미지 데이터
│   ├── green/
│   ├── left/
│   ├── middle/
│   ├── noline/
│   ├── red/
│   └── right/
├── model.tflite         # 학습된 모델 (학습 후 생성)
└── README.md
```

## 클래스 정의

- **green**: 초록불 (신호등)
- **left**: 좌회전 (라인이 왼쪽에 있음)
- **middle**: 중앙/직진 (라인이 중앙에 있음)
- **noline**: 라인 없음
- **red**: 빨간불/정지 (신호등)
- **right**: 우회전 (라인이 오른쪽에 있음)

**주의**: 클래스 순서는 `camera_main_gpt.py`와 동일하게 유지해야 합니다!

## 사용 방법

### 1. 데이터 수집

**방법 A: 카메라로 실시간 수집 (Raspberry Pi에서)**
```bash
python3 collect_data.py
```

**방법 B: 폰으로 찍은 이미지 분류 (Mac에서)**
```bash
# 1. 폰으로 이미지들을 하나의 폴더에 모으기 (예: photos/)
# 2. 이미지 분류 스크립트 실행
python3 organize_images.py --input photos/

# 옵션: --move 플래그를 사용하면 원본 이미지를 이동 (기본값: 복사)
python3 organize_images.py --input photos/ --move
```

**키보드 입력:**
- `g` - green (초록불) 선택
- `l` - left (좌회전) 선택
- `m` - middle (중앙/직진) 선택
- `n` - noline (라인 없음) 선택
- `r` - red (빨간불) 선택
- `f` - right (우회전) 선택
- `스페이스바` - 현재 프레임 저장
- `q` - 종료

**팁:**
- 각 클래스당 최소 100-200개 이상의 이미지를 수집하는 것을 권장합니다.
- 다양한 조명 조건, 각도, 거리에서 데이터를 수집하세요.
- 클래스별로 균형있게 데이터를 수집하세요.
- **폰으로 찍은 이미지도 전혀 문제없습니다!** 정지된 이미지로 학습해도 동영상에서 잘 작동합니다.

### 2. 모델 학습 (Mac M2에서)

```bash
# TensorFlow 설치 (Mac M2 최적화)
pip install tensorflow-macos tensorflow-metal

# 학습 실행
python3 train_model.py
```

학습이 완료되면 다음 파일이 생성됩니다:
- `line_tracing_model.h5` - Keras 모델 (주요 모델)
- `model.tflite` - TensorFlow Lite 모델 (Raspberry Pi용, 선택사항)
- `training_history.png` - 학습 히스토리 그래프

**학습 파라미터:**
- **Transfer Learning**: MobileNetV2 또는 EfficientNetB0 사용
- 이미지 크기: 224x224
- 배치 크기: 32
- 에포크: 50 (조기 종료 포함)
- 2단계 학습: Base 모델 고정 → Fine-tuning

**모델 선택:**
`train_model.py`에서 `USE_EFFICIENTNET = True`로 변경하면 EfficientNet 사용 (더 정확하지만 느림)

### 3. 라인트레이싱 실행

```bash
python3 run_line_tracing.py
```

학습된 모델을 사용하여 자동으로 라인트레이싱을 수행합니다.

**동작:**
- 카메라에서 프레임을 캡처
- 모델로 예측 수행
- 예측 결과에 따라 모터 제어
- 빨간불 감지 시 정지
- `q` 키로 종료

## 요구사항

### 하드웨어
- Raspberry Pi
- Pi Camera 2
- 모터 및 서보 모터 (GPIO 연결)

### 소프트웨어

**Mac M2 (학습용):**
```bash
pip install tensorflow-macos tensorflow-metal
pip install numpy opencv-python scikit-learn matplotlib
```

**Raspberry Pi (실행용):**
```bash
pip install numpy opencv-python picamera2 tflite-runtime RPi.GPIO
```

또는 가상환경 사용:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 문제 해결

### 데이터 수집 시
- 카메라가 인식되지 않으면: `sudo modprobe bcm2835-v4l2` 실행
- 이미지가 저장되지 않으면: `data/` 디렉토리 권한 확인

### 학습 시
- 메모리 부족: 배치 크기를 줄이거나 이미지 크기를 줄이세요
- 데이터 불균형: 클래스별 데이터 수를 비슷하게 맞추세요

### 실행 시
- 모델을 찾을 수 없음: `train_model.py`를 먼저 실행하세요
- GPIO 오류: `sudo` 권한으로 실행하거나 GPIO 권한 설정

## 성능 개선 팁

1. **데이터 품질**
   - 다양한 환경에서 데이터 수집
   - 노이즈가 있는 이미지 제거
   - 데이터 증강 활용

2. **모델 튜닝**
   - 학습률 조정
   - 모델 구조 변경
   - 정규화 추가

3. **실행 최적화**
   - 프레임 레이트 조정
   - 예측 버퍼 크기 조정
   - 신뢰도 임계값 조정

## 라이선스

프로젝트 라이선스를 참조하세요.

