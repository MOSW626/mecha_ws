# Models Directory

이 폴더는 라인 트레이싱 시스템에서 사용하는 TensorFlow Lite 모델 파일들을 포함합니다.
This directory contains TensorFlow Lite model files used by the line tracing system.

## 모델 파일 목록 / Model Files

### 1. `gpu_model_lite.tflite` (ML1 모델 / ML1 Model)
- **용도 / Purpose**: 신호등 및 CV 모드 판단 / Traffic light and CV mode detection
- **출력 라벨 / Output Labels**: `["cv", "green", "red"]`
- **설명 / Description**:
  - 이미지 상단 40% 영역을 사용하여 신호등(빨강/초록) 또는 CV 모드 여부를 판단합니다
    Uses the top 40% of the image to detect traffic lights (red/green) or CV mode
  - `src/linetracing_ml.py`의 `judge_ml()` 함수에서 사용됩니다
    Used by the `judge_ml()` function in `src/linetracing_ml.py`
  - ML이 "cv"로 판단하면 CV 로직 또는 ML2+CV 가중치 결합 로직을 사용합니다
    When ML detects "cv", it uses CV logic or ML2+CV weighted combination logic

### 2. `check_line.tflite` (ML2 모델 / ML2 Model)
- **용도 / Purpose**: 라인 트레이싱 방향 판단 / Line tracing direction detection
- **출력 라벨 / Output Labels**: `["forward", "right", "left", "green", "red", "noline"]`
- **설명 / Description**:
  - 전체 이미지를 사용하여 라인 트레이싱 방향을 판단합니다
    Uses the full image to determine line tracing direction
  - `src/linetracing_ml.py`의 `judge_ml2()` 함수에서 사용됩니다
    Used by the `judge_ml2()` function in `src/linetracing_ml.py`
  - ML1이 "cv"로 판단할 때, ML2와 CV를 가중치로 결합하여 사용합니다 (ML2: 0.2, CV: 0.8)
    When ML1 detects "cv", combines ML2 and CV with weights (ML2: 0.2, CV: 0.8)
- **참고사항 / Notes**:
  - 파일 크기가 커서 Git에 업로드되지 않았습니다 (더미 파일로 대체됨)
    File size is too large for Git upload (replaced with dummy file)
  - 실제 모델 파일은 별도로 배포되거나 학습하여 생성해야 합니다
    Actual model file must be distributed separately or generated through training
  - 다른 프로젝트에서 학습에 참고할 수 있습니다
    Can be used as a reference for training in other projects

## 사용 방법 / Usage

모델 파일들은 `src/linetracing.py`의 `run_linetracing_sequence()` 함수에서 자동으로 로드됩니다.
Model files are automatically loaded by the `run_linetracing_sequence()` function in `src/linetracing.py`.

```python
# 초기화 / Initialization
linetracing_ml.init_ml()  # ML1과 ML2 모델 모두 로드 / Loads both ML1 and ML2 models

# ML1 판단 (신호등/CV 판단) / ML1 Detection (Traffic light/CV detection)
ml_label = linetracing_ml.judge_ml(frame_rgb)  # "cv", "green", "red" 중 하나 / One of "cv", "green", "red"

# ML2 판단 (방향 판단, ML1이 "cv"일 때 사용) / ML2 Detection (Direction detection, used when ML1 is "cv")
ml2_result = linetracing_ml.judge_ml2(frame_rgb)  # "forward", "right", "left" 등 / "forward", "right", "left", etc.
```

## 가중치 결합 로직 / Weighted Combination Logic

ML1이 "cv"로 판단할 때, ML2와 CV(Computer Vision)의 결과를 가중치로 결합합니다:
When ML1 detects "cv", it combines ML2 and CV (Computer Vision) results with weights:

- **ML2 가중치 / ML2 Weight**: 0.2
- **CV 가중치 / CV Weight**: 0.8

결합 로직은 `src/linetracing.py`의 `combine_ml2_cv_weighted()` 함수에서 구현되어 있습니다.
The combination logic is implemented in the `combine_ml2_cv_weighted()` function in `src/linetracing.py`.

## 모델 학습 / Model Training

모델 학습 관련 정보는 `experiments/cnn/` 디렉토리를 참고하세요.
For model training information, refer to the `experiments/cnn/` directory.

