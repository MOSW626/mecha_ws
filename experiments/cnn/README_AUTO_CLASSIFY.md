# 이미지 자동 분류 도구

`data` 폴더에 있는 `frame_*.png` 파일들을 자동으로 분류합니다.

## 사용 방법

### 기본 사용 (자동 모드)
```bash
cd cnn
python3 auto_classify.py
```

자동으로:
1. 학습된 모델을 찾아서 사용
2. 모델이 없으면 기존 분류된 데이터로 빠르게 학습
3. 그것도 안되면 CV 기반으로 분류

### 옵션

```bash
# 특정 모델 경로 지정
python3 auto_classify.py --model-path ../line_tracing/model.tflite

# 신뢰도 임계값 조정 (기본값: 0.5)
python3 auto_classify.py --confidence-threshold 0.7

# 분류 방법 강제 지정
python3 auto_classify.py --method model    # 모델만 사용
python3 auto_classify.py --method cv      # CV만 사용
python3 auto_classify.py --method train    # 빠른 학습 후 사용
python3 auto_classify.py --method auto    # 자동 선택 (기본값)
```

## 작동 원리

### 1. 모델 기반 분류 (가장 정확)
- 학습된 모델이 있으면 사용
- `model.tflite` 또는 `line_tracing_model.h5` 파일을 찾음
- 여러 경로에서 자동 검색

### 2. 빠른 학습 (중간 정확도)
- 기존 분류된 데이터가 있으면 빠르게 학습
- 간단한 CNN 모델로 5 에포크만 학습
- 약 1-2분 소요

### 3. CV 기반 분류 (빠름, 낮은 정확도)
- 라인 위치를 분석하여 분류
- 트래픽 라이트 색상 검출
- 모델 없이도 작동

## 분류 결과

이미지들이 다음 폴더로 자동 이동됩니다:
- `forward/` - 직진
- `left/` - 좌회전
- `right/` - 우회전
- `green/` - 초록불
- `red/` - 빨간불
- `non/` - 라인 없음 / 신뢰도 낮음

## 주의사항

1. **백업**: 분류 전에 데이터를 백업하세요
2. **검토**: 자동 분류 후 일부 이미지는 수동으로 확인하는 것을 권장합니다
3. **신뢰도**: `--confidence-threshold`를 높이면 더 정확하지만, 분류되지 않는 이미지가 많아질 수 있습니다

## 문제 해결

### 모델을 찾을 수 없음
- `--method train` 옵션으로 빠른 학습 시도
- 또는 `--method cv`로 CV 기반 분류 사용

### 분류가 부정확함
- 신뢰도 임계값을 높이세요: `--confidence-threshold 0.7`
- 더 많은 학습 데이터로 모델을 재학습하세요

### TensorFlow 오류
- CV 기반 분류 사용: `--method cv`
- 또는 TensorFlow 설치: `pip install tensorflow`

