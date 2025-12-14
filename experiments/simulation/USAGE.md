# 사용 가이드

## 📚 목차
1. [학습하기](#학습하기)
2. [모델 테스트하기](#모델-테스트하기)
3. [학습 결과 확인하기](#학습-결과-확인하기)
4. [라즈베리파이 배포](#라즈베리파이-배포)

---

## 🎓 학습하기

### 기본 학습 (빠른 테스트)
```bash
cd simulation
source venv/bin/activate
python3 train.py --mode train --timesteps 50000
```

### 본격적인 학습
```bash
python3 train.py --mode train --timesteps 1000000
```

### 학습 옵션
```bash
# 렌더링 없이 빠르게 학습
python3 train.py --mode train --timesteps 50000 --no-render-eval --render-freq 0

# 주기적 렌더링 빈도 조정
python3 train.py --mode train --timesteps 50000 --render-freq 10000
```

### 학습 중 확인할 것들
- **터미널 출력**: `ep_rew_mean`이 점점 증가하는지 확인
- **학습 곡선 그래프**: 리워드가 상승하는지 확인
- **트랙 시각화**: 차량이 트랙을 따라가는지 확인

---

## 🧪 모델 테스트하기

### 간단한 테스트
```bash
# 기본 모델 테스트 (렌더링 포함)
python3 test_model.py

# 또는 train.py 직접 사용
python3 train.py --mode test
```

### 상세 테스트
```bash
# 여러 에피소드 테스트
python3 test_model.py --episodes 10

# 렌더링 없이 빠르게 테스트
python3 test_model.py --episodes 10 --no-render

# 특정 모델 테스트
python3 test_model.py --model models/best/best_model --episodes 5
```

### 테스트 결과 해석
- **평균 리워드**: 높을수록 좋음 (양수면 성공!)
- **에피소드 길이**: 길수록 더 오래 주행 (충돌 없이)
- **최고/최저 리워드**: 성능의 일관성 확인

---

## 📊 학습 결과 확인하기

### 1. 학습 곡선 그래프
```bash
# 학습 중 자동으로 생성됨
open logs/learning_curve.png
```

**확인 사항:**
- 리워드가 시간에 따라 증가하는가?
- 에피소드 길이가 증가하는가? (더 오래 살아남는다는 의미)

### 2. TensorBoard (고급)
```bash
tensorboard --logdir logs/tensorboard/
```
브라우저에서 `http://localhost:6006` 접속

### 3. 저장된 모델
```bash
ls models/
# models/
#   ├── best/              # 최고 성능 모델
#   │   └── best_model.zip
#   ├── checkpoints/       # 정기 저장 모델
#   └── ppo_racing_car_final.zip  # 최종 모델
```

---

## 🤖 라즈베리파이 배포

### 1. 모델 파일 전송
```bash
# 라즈베리파이로 모델 파일 전송
scp models/ppo_racing_car_final.zip pi@raspberrypi:/home/pi/racing_car/
```

### 2. 라즈베리파이에서 실행
```bash
# 라즈베리파이에서
cd ~/racing_car
unzip ppo_racing_car_final.zip

# run_pi.py 수정 필요:
# - HardwareInterface 클래스에 실제 GPIO 코드 추가
# - get_ultrasonic_distances() 구현
# - get_camera_line_error() 구현
# - set_servo_angle() 구현
# - set_motor_throttle() 구현

# 실행
python3 run_pi.py --model ppo_racing_car_final
```

### 3. ONNX 변환 (선택사항)
현재 ONNX 변환이 실패하는 경우:
```bash
# onnxscript 설치
pip install onnxscript

# 다시 학습 후 변환 시도
python3 train.py --mode train --timesteps 50000
```

또는 PyTorch 모델을 직접 사용:
- `run_pi.py`를 수정하여 PyTorch 모델 직접 로드
- 더 많은 메모리가 필요하지만 작동함

---

## 🎯 학습 성능 개선 팁

### 리워드가 낮을 때 (-80 이하)
1. **리워드 함수 확인**: `env.py`의 `_calculate_reward()` 확인
2. **학습 시간 증가**: `--timesteps`를 더 크게 (100000 이상)
3. **학습률 조정**: `train.py`의 `learning_rate` 파라미터 조정

### 학습이 느릴 때
1. **렌더링 비활성화**: `--no-render-eval --render-freq 0`
2. **트랙 크기 줄이기**: `env.py`의 `track_length_min/max` 줄이기
3. **GPU 사용**: GPU가 있으면 자동으로 사용됨

### 차량이 계속 충돌할 때
1. **트랙 폭 증가**: `track_width_min/max` 증가
2. **초기 속도 조정**: `reset()` 함수의 `initial_velocity` 조정
3. **리워드 함수 개선**: 충돌 페널티 완화

---

## 📝 자주 묻는 질문

### Q: 학습이 완료되었는데 어떻게 사용하나요?
A: `test_model.py`로 테스트하거나, `run_pi.py`를 수정하여 실제 하드웨어에 연결하세요.

### Q: 모델이 어디에 저장되나요?
A: `models/` 디렉토리에 저장됩니다. 최고 성능 모델은 `models/best/best_model.zip`입니다.

### Q: 학습 곡선이 보이지 않아요
A: 학습 중 matplotlib 창이 열립니다. 닫지 마세요! 또는 `logs/learning_curve.png`를 확인하세요.

### Q: ONNX 변환이 실패해요
A: `pip install onnxscript`를 실행하거나, PyTorch 모델을 직접 사용하세요.

### Q: 리워드가 여전히 낮아요
A: 더 많은 타임스텝으로 학습하세요 (최소 100,000 이상 권장).

---

## 🚀 빠른 시작 예제

```bash
# 1. 학습 (5분 정도)
cd simulation
source venv/bin/activate
python3 train.py --mode train --timesteps 50000

# 2. 테스트 (학습된 모델 확인)
python3 test_model.py --episodes 5

# 3. 결과 확인
open logs/learning_curve.png
ls models/
```

---

## 💡 다음 단계

1. **더 긴 학습**: 100,000 타임스텝 이상으로 학습
2. **하이퍼파라미터 튜닝**: `train.py`의 학습 파라미터 조정
3. **리워드 함수 개선**: `env.py`의 `_calculate_reward()` 수정
4. **실제 하드웨어 연결**: `run_pi.py`의 하드웨어 인터페이스 구현

