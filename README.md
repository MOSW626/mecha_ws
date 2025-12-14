# mecha_ws

자율주행 로봇 프로젝트 워크스페이스  
Autonomous Driving Robot Project Workspace

## 📚 프로젝트 정보 / Project Information

**과목 / Course**: KAIST Mechatronics System Design (ME203)  
**프로젝트 / Project**: 자율주행 로봇 시스템 개발 / Autonomous Driving Robot System Development

### 📊 최종 발표 자료 / Final Presentation

[프로젝트 발표 프레젠테이션 보기 / View Project Presentation](https://docs.google.com/presentation/d/e/2PACX-1vTjnZmDp63P8Fe85efcQ0TtqOuLPMxrtsNupx1o-mQ86d1k6RMgJLY8ttvDVztPpagZjweUCYdb9oF_/pub?start=false&loop=false&delayms=3000)

## 📁 프로젝트 구조 / Project Structure

```
mecha_ws/
├── src/                        # 메인 소스 코드 / Main source code
│   ├── main.py                 # 메인 실행 파일 / Main entry point
│   ├── linetracing.py          # 라인트레이싱 메인 로직 / Line tracing main logic
│   ├── linetracing_cv.py        # CV 기반 라인 감지 / CV-based line detection
│   ├── linetracing_ml.py        # ML 기반 신호등 감지 / ML-based traffic light detection
│   ├── linetracing_drive.py     # 모터/서보 제어 / Motor/servo control
│   ├── linetracing_Judgment.py  # CV/ML 판단 통합 / CV/ML judgment integration
│   └── low_defense.py           # 초음파 센서 기반 고속 주행 / Ultrasonic sensor-based high-speed driving
│
├── models/                     # ML 모델 파일 / ML model files
│   └── gpu_model_lite.tflite   # 최종 신호등 감지 모델 / Final traffic light detection model
│
├── test/                       # 테스트 파일들 / Test files
│   ├── README_TEST_CV.md
│   ├── test_cv_local.py
│   ├── test_ml_local.py
│   └── using_both_michan_decrease_speed_straight.py
│
├── settings/                   # 설정 및 유틸리티 / Settings and utilities
│   ├── cameracheck.py
│   ├── Check_dependencies_pi.py
│   └── setting.sh
│
├── logs/                       # 로그 파일 및 이미지 / Log files and images
│   ├── line_log/               # 라인트레이싱 이미지 로그 / Line tracing image logs
│   ├── linetracinglog.txt
│   ├── linetracinglog2.txt
│   ├── main.log
│   └── mllog
│
└── experiments/                # 실험 및 참고 코드 (사용 안 함) / Experimental and reference code (not used)
    ├── archive/                 # 사용하지 않는 파일들 / Unused files
    ├── cnn/                     # CNN 학습 실험 / CNN training experiments
    ├── line_tracing/            # 라인트레이싱 실험 코드 / Line tracing experimental code
    ├── reference/               # 수업 시간 참고 코드 / Class reference code
    ├── simulation/              # ML 시뮬레이션 실험 / ML simulation experiments
    ├── track_old/               # 기존 트랙 주행 코드 / Old track driving code
    └── Yeonsu_track/            # 벽 따라가기 실험 / Wall following experiments
```

## 🚀 실행 방법 / How to Run

### 메인 시스템 실행 / Run Main System

```bash
cd src
python3 main.py
```

또는 프로젝트 루트에서 / Or from project root:

```bash
python3 src/main.py
```

### 시스템 동작 흐름 / System Workflow

1. **Part 1: 라인트레이싱 / Line Tracing**
   - 카메라로 라인 감지 (CV + ML 하이브리드) / Line detection using camera (CV + ML hybrid)
   - ML 모델로 신호등 감지 (Red/Green) / Traffic light detection using ML model (Red/Green)
   - 초록불 감지 시 Part 2로 전환 / Switch to Part 2 when green light detected
   - 빨간불 감지 시 정지 후 초록불 대기 / Stop and wait for green light when red light detected

2. **Part 2: 초음파 센서 주행 / Ultrasonic Sensor Driving**
   - 초음파 센서 기반 고속 주행 / High-speed driving based on ultrasonic sensors
   - 좌우 벽 거리 측정하여 중앙 유지 / Maintain center by measuring left/right wall distances
   - 코너/직선 구간 자동 감지 및 속도 조절 / Automatic corner/straight detection and speed adjustment

## 🔧 주요 모듈 / Key Modules

### 라인트레이싱 모듈 (`src/`) / Line Tracing Module

- **linetracing.py**: 메인 로직, 신호등 감지 및 단계 전환 / Main logic, traffic light detection and stage transition
- **linetracing_cv.py**: OpenCV 기반 라인 감지 / OpenCV-based line detection
- **linetracing_ml.py**: TFLite 모델로 신호등 감지 (Red/Green/CV) / Traffic light detection using TFLite model (Red/Green/CV)
- **linetracing_drive.py**: 모터/서보 하드웨어 제어 / Motor/servo hardware control
- **linetracing_Judgment.py**: CV와 ML 판단 결과 통합 / Integration of CV and ML judgment results

### 초음파 주행 모듈 / Ultrasonic Driving Module

- **low_defense.py**: 초음파 센서 기반 고속 주행 제어 / Ultrasonic sensor-based high-speed driving control
  - 좌우 센서 거리 차이로 조향각 계산 / Calculate steering angle from left/right sensor distance difference
  - 코너/직선 구간 자동 감지 / Automatic corner/straight section detection
  - 구간별 속도 자동 조절 / Automatic speed adjustment by section

## 📦 의존성 / Dependencies

주요 라이브러리 / Main libraries:
- `RPi.GPIO`: GPIO 제어 / GPIO control
- `picamera2`: 카메라 제어 / Camera control
- `opencv-python-headless`: 이미지 처리 (Raspberry Pi용) / Image processing (for Raspberry Pi)
- `tflite-runtime`: ML 모델 추론 / ML model inference
- `numpy`: 수치 연산 / Numerical computation

### requirements.txt 사용 / Using requirements.txt

```bash
pip install -r requirements.txt
```

또는 가상환경에서 / Or in virtual environment:

```bash
source ~/venvs/mecha/bin/activate
pip install -r requirements.txt
```

## ⚙️ 설정 및 설치 / Setup and Installation

### 자동 설치 스크립트 / Automated Setup Script

Raspberry Pi에서 다음 명령어로 자동으로 환경을 설정할 수 있습니다:  
You can automatically set up the environment on Raspberry Pi with the following command:

```bash
cd settings
chmod +x setting.sh
./setting.sh
```

### 설치 스크립트 기능 / Setup Script Features

`settings/setting.sh` 스크립트는 다음 작업을 자동으로 수행합니다:  
The `settings/setting.sh` script automatically performs the following tasks:

1. **시스템 패키지 업데이트 / System Package Update**
   - `apt update` 및 필수 패키지 설치 / `apt update` and install essential packages
   - Python3, OpenCV, Picamera2, GPIO 라이브러리 설치 / Install Python3, OpenCV, Picamera2, GPIO libraries

2. **가상환경 생성 / Virtual Environment Creation**
   - `~/venvs/mecha` 경로에 가상환경 생성 / Create virtual environment at `~/venvs/mecha`
   - 시스템 사이트 패키지와 연동 / Link with system site packages
   - `.bashrc`에 자동 활성화 추가 / Add auto-activation to `.bashrc`

3. **Python 패키지 설치 / Python Package Installation**
   - pip 업그레이드 / Upgrade pip
   - 필수 패키지 설치: `tflite-runtime`, `numpy`, `opencv-python-headless` / Install essential packages

4. **의존성 확인 / Dependency Check**
   - `Check_dependencies_pi.py` 실행하여 설치 확인 / Run `Check_dependencies_pi.py` to verify installation

5. **카메라 확인 / Camera Check**
   - `cameracheck.py` 실행하여 카메라 동작 확인 / Run `cameracheck.py` to verify camera operation

### 수동 설치 / Manual Installation

자동 스크립트를 사용하지 않는 경우, 다음 명령어를 순서대로 실행하세요:  
If you prefer manual installation, run the following commands in order:

```bash
# 시스템 패키지 설치 / Install system packages
sudo apt update
sudo apt install -y python3 python3-pip python3-venv python3-opencv python3-picamera2 python3-gpiozero

# 가상환경 생성 / Create virtual environment
python3 -m venv ~/venvs/mecha --system-site-packages
source ~/venvs/mecha/bin/activate

# Python 패키지 설치 / Install Python packages
pip install --upgrade pip
pip install tflite-runtime numpy opencv-python-headless
```

### 가상환경 활성화 / Activate Virtual Environment

설치 후 가상환경을 활성화하세요:  
After installation, activate the virtual environment:

```bash
source ~/venvs/mecha/bin/activate
```

터미널을 재시작하면 자동으로 활성화됩니다.  
The virtual environment will auto-activate when you restart your terminal.

## 📝 참고사항 / Notes

- **최종 모델 / Final Model**: `models/gpu_model_lite.tflite` (신호등 감지용 / for traffic light detection)
- **로그 이미지 / Log Images**: `logs/line_log/` 폴더 (디버깅용, 필요시 활성화 / for debugging, enable when needed)
- **테스트 코드 / Test Code**: `test/` 폴더 참고 / See `test/` folder
- **실험 코드 / Experimental Code**: `experiments/` 폴더 (참고용, 사용 안 함 / for reference, not used)
