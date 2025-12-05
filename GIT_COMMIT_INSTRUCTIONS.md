# Git 커밋 및 푸시 가이드

## 완료된 작업
1. ✅ `.gitignore`에 frame 이미지 패턴 추가 (`**/frame_*.png`, `**/frame_*.jpeg`, `**/frame_*.jpg`)
2. ✅ `cnn/.gitignore`에도 frame 이미지 패턴 추가
3. ✅ `.DS_Store` 파일 제외 설정

## 다음 단계 (수동 실행)

터미널에서 다음 명령어를 순서대로 실행하세요:

```bash
cd /Users/an-yeonsu/Documents/GitHub/mecha_ws

# 1. .gitignore 파일 커밋
git add .gitignore cnn/.gitignore
git commit -m "Exclude frame images and .DS_Store from git"

# 2. 코드 파일만 추가 (frame 이미지 제외됨)
git add cnn/*.py cnn/*.md cnn/*.txt
git add *.py
git add line_tracing/*.py line_tracing/*.md 2>/dev/null
git add simulation/*.py simulation/*.md simulation/*.sh 2>/dev/null
git add Yeonsu_track/*.py Yeonsu_track/*.md 2>/dev/null

# 3. 모델 파일 추가 (선택사항 - 크기가 클 수 있음)
# git add cnn/cnn_model.keras cnn/cnn_model.h5 cnn/cnn_model.tflite

# 4. 상태 확인
git status --short

# 5. 커밋
git commit -m "Update code and models, exclude frame images"

# 6. 푸시 시도
git push origin main
```

## HTTP 500 에러 발생 시

원격 서버 오류가 발생하면:
1. 잠시 후 다시 시도: `git push origin main`
2. 또는 작은 단위로 나눠서 푸시
3. GitHub 웹 인터페이스에서 확인

## 확인 사항

- frame으로 시작하는 이미지들은 이제 git에 추가되지 않습니다
- .DS_Store 파일들도 무시됩니다
- 코드 파일과 모델 파일만 커밋됩니다

