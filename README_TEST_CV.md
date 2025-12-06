# CV 알고리즘 로컬 테스트 가이드

## 개요
`test_cv_local.py`는 로컬 컴퓨터에서 CV 알고리즘을 테스트할 수 있는 스크립트입니다.
이미지 파일을 입력받아 CV 판단 결과를 시각화하여 보여줍니다.

## 설치 요구사항
```bash
pip install opencv-python numpy pillow
```

## 사용 방법

### 단일 이미지 테스트
```bash
python3 test_cv_local.py <이미지_경로> [옵션]
```

### 디렉토리 내 모든 이미지 테스트
```bash
python3 test_cv_local.py <디렉토리_경로> [옵션]
```

### 옵션
- `--save`: 결과 이미지를 저장합니다 (기본 디렉토리: `test_output`)
- `--output <디렉토리>`: 결과 이미지 저장 디렉토리 지정
- `--show`: 결과 이미지를 화면에 표시합니다 (GUI 필요)

## 사용 예시

### 예시 1: 단일 이미지 테스트 및 결과 저장
```bash
python3 test_cv_local.py line_log/test2_0001.jpg --save
```

### 예시 2: 디렉토리 내 모든 이미지 테스트
```bash
python3 test_cv_local.py line_log/ --save --output test_results
```

### 예시 3: 결과 화면에 표시
```bash
python3 test_cv_local.py line_log/test2_0001.jpg --show
```

## 출력 정보

스크립트는 다음 정보를 출력합니다:
- CV 판단 결과 (forward, left, right, non, red, green)
- 하단 라인 중심 좌표
- 상단 라인 중심 좌표
- 라인 각도
- 판단 조건 (center_error < threshold, |angle| < 10 등)

## 시각화 정보

결과 이미지에는 다음이 표시됩니다:
- **초록색 사각형**: ROI (Region of Interest) 영역
- **파란색 점**: 하단 라인 중심
- **노란색 점**: 상단 라인 중심
- **빨간색 선**: 이미지 중심에서 라인 중심까지의 오프셋
- **우측 상단**: 이진화 결과
- **텍스트 정보**: 판단 결과, 좌표, 각도, 판단 조건

## 문제 해결

### RGB/BGR 문제
- 저장된 이미지는 BGR 형식일 수 있습니다
- 스크립트는 자동으로 BGR을 RGB로 변환합니다
- Picamera2는 RGB를 반환하므로 실제 로봇에서는 문제가 없습니다

### CV 알고리즘 문제
- `judge_direction` 함수가 개선되었습니다
- 각도 임계값이 10도에서 15도로 완화되었습니다
- 각도 정보를 더 잘 활용하도록 로직이 개선되었습니다

## 수정 사항

### 1. RGB/BGR 변환 수정
- 이미지 저장 시 RGB 형식 유지
- 테스트 스크립트에서 BGR→RGB 변환 자동 처리

### 2. CV 판단 로직 개선
- `judge_direction` 함수 개선:
  - 각도 임계값 완화 (10도 → 15도)
  - 각도 정보를 더 잘 활용
  - None 값 처리 개선

### 3. 디버깅 정보 추가
- 판단 조건 시각화
- center_error와 threshold 비교 표시
- 각도 정보 표시

