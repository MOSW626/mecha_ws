# 강화학습 기반 자율주행 레이싱 카 프로젝트

PPO 알고리즘을 사용하여 자율주행 레이싱 카를 학습시키고, 라즈베리파이에 배포하는 프로젝트입니다.

## 프로젝트 구조

```
simulation/
├── env.py              # 커스텀 Gymnasium 환경 (랜덤 트랙 생성)
├── train.py            # PPO 학습 스크립트 및 ONNX 변환
├── run_pi.py           # 라즈베리파이용 경량 추론 스크립트
├── requirements.txt    # 필요한 라이브러리 목록
└── README.md           # 이 파일
```

## 주요 특징

### 1. Sim-to-Real 최적화
- **도메인 랜덤화**: 매 에피소드마다 새로운 랜덤 트랙 생성
- **센서 노이즈**: 가우시안 노이즈로 실제 센서 불확실성 시뮬레이션
- **마찰 변동**: 물리 파라미터 변동으로 다양한 환경 조건 학습

### 2. 커스텀 환경 (env.py)
- **랜덤 트랙 생성**: 스플라인을 사용한 부드러운 폐루프 트랙
- **운동학적 자전거 모델**: 현실적인 차량 동역학
- **레이캐스팅**: 초음파 센서 시뮬레이션
- **정규화된 관측값**: [초음파 거리들, 카메라 라인 에러]

### 3. 학습 (train.py)
- **PPO 알고리즘**: Stable-Baselines3 사용
- **자동 모델 저장**: 최고 성능 모델 자동 저장
- **ONNX 변환**: 라즈베리파이 배포를 위한 경량 모델 변환

### 4. 추론 (run_pi.py)
- **경량 설계**: gym, pygame, matplotlib 의존성 없음
- **ONNX Runtime**: 빠른 추론 성능
- **하드웨어 인터페이스**: GPIO 및 카메라 연동 준비

## 설치 방법

### 학습 환경 (Mac/PC/노트북)

**네, 노트북에서 학습할 수 있습니다!** CPU만 있어도 학습이 가능하며, GPU가 있으면 자동으로 사용합니다.

#### 방법 1: 자동 설치 스크립트 (권장)

```bash
cd simulation
chmod +x install.sh
./install.sh
```

#### 방법 2: 수동 설치

```bash
cd simulation

# 가상환경 생성 (선택사항, 권장)
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt
```

#### 설치 확인

```bash
python3 -c "import torch; print('PyTorch:', torch.__version__)"
python3 -c "import stable_baselines3; print('Stable-Baselines3 설치 완료')"
python3 -c "import gymnasium; print('Gymnasium 설치 완료')"
```

#### 빠른 테스트 (설치 확인용)

```bash
# 환경이 제대로 작동하는지 간단히 테스트
python3 -c "from env import RandomTrackEnv; env = RandomTrackEnv(); obs, _ = env.reset(); print('환경 테스트 성공! 관측값 shape:', obs.shape)"
```

### 라즈베리파이 환경

```bash
# 최소 의존성만 설치
pip install onnxruntime numpy

# GPIO 및 카메라 지원 (선택사항)
pip install RPi.GPIO picamera2
```

## 빠른 시작

```bash
# 1. 학습 (5분 정도)
cd simulation
source venv/bin/activate
python3 train.py --mode train --timesteps 50000

# 2. 테스트
python3 test_model.py

# 3. 자세한 사용법
cat USAGE.md
```

## 사용 방법

### 1. 모델 학습

**노트북에서 학습 시작하기:**

```bash
# 간단한 테스트 학습 (10,000 타임스텝, 약 5-10분)
python3 train.py --mode train --timesteps 10000

# 기본 학습 (1,000,000 타임스텝, 몇 시간 소요)
python3 train.py --mode train --timesteps 1000000

# 더 긴 학습 (5,000,000 타임스텝, 하루 이상 소요 가능)
python3 train.py --mode train --timesteps 5000000
```

**학습 시간 예상:**
- CPU만 사용 시: 10,000 타임스텝당 약 5-10분
- GPU 사용 시: 10,000 타임스텝당 약 1-2분
- 실제 학습에는 최소 100,000 타임스텝 이상 권장

학습 중 생성되는 파일:
- `models/best/best_model.zip`: 최고 성능 모델
- `models/ppo_racing_car_final.zip`: 최종 모델
- `models/racing_car_policy.onnx`: ONNX 모델 (라즈베리파이용)
- `logs/`: 학습 로그 및 TensorBoard 데이터

### 2. 학습된 모델 테스트

```bash
# 최고 모델 테스트
python train.py --mode test --model-path models/best/best_model

# 최종 모델 테스트
python train.py --mode test --model-path models/ppo_racing_car_final
```

### 3. 라즈베리파이에서 추론 실행

```bash
# 기본 실행
python run_pi.py --model racing_car_policy.onnx

# 커스텀 설정
python run_pi.py \
    --model models/racing_car_policy.onnx \
    --sensors 3 \
    --sensor-range 150.0 \
    --max-steering 20.0 \
    --freq 10.0
```

## 환경 파라미터 설정

`env.py`의 `RandomTrackEnv` 클래스에서 다음 파라미터를 조정할 수 있습니다:

- **센서 설정**:
  - `num_ultrasonic_sensors`: 초음파 센서 개수
  - `sensor_angles`: 센서 장착 각도 (도)
  - `sensor_max_range`: 센서 최대 감지 거리 (cm)
  - `sensor_noise_std_dev`: 센서 노이즈 표준편차 (Sim-to-Real)

- **트랙 설정**:
  - `track_width_min/max`: 트랙 폭 범위 (cm)
  - `track_length_min/max`: 트랙 길이 범위 (cm)

- **차량 설정**:
  - `max_steering_angle`: 최대 조향각 (도)
  - `max_speed`: 최대 속도 (cm/s)
  - `friction_variation`: 마찰 계수 변동 (Sim-to-Real)

## 하드웨어 연동

`run_pi.py`의 `HardwareInterface` 클래스를 실제 하드웨어에 맞게 수정하세요:

1. **GPIO 설정**: 서보 모터 및 DC 모터 핀 번호 설정
2. **초음파 센서**: TRIG/ECHO 핀 매핑
3. **카메라**: Picamera2 설정 및 라인 검출 로직

예시 코드는 주석으로 제공되어 있습니다.

## 성능 최적화

### 학습 속도 향상
- GPU 사용: PyTorch가 CUDA를 자동으로 감지하여 사용
- 벡터화된 환경: 여러 환경을 병렬로 실행 (코드 수정 필요)

### 추론 속도 향상
- ONNX Runtime 최적화: `run_pi.py`에서 이미 활성화됨
- CPU 스레드 수 조정: `sess_options.intra_op_num_threads` 조정
- 제어 루프 주파수: `--freq` 파라미터로 조정 (기본값: 10 Hz)

## 문제 해결

### ONNX 변환 실패
- PyTorch 버전 확인: `torch>=2.0.0` 권장
- ONNX opset 버전: `train.py`에서 `opset_version` 조정

### 추론 속도가 느림
- ONNX Runtime 버전 확인: 최신 버전 사용
- CPU 스레드 수 조정: 라즈베리파이 CPU 코어 수에 맞게 설정

### 학습이 수렴하지 않음
- 학습률 조정: `train.py`의 `learning_rate` 파라미터
- 리워드 함수 조정: `env.py`의 `_calculate_reward` 메서드
- 더 많은 타임스텝: `--timesteps` 증가

## 참고 자료

- [Stable-Baselines3 문서](https://stable-baselines3.readthedocs.io/)
- [Gymnasium 문서](https://gymnasium.farama.org/)
- [ONNX Runtime 문서](https://onnxruntime.ai/)

## 라이선스

이 프로젝트는 MIT 라이선스를 따릅니다.

