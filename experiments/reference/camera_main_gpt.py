#!/home/mecha/venvs/tflite/bin/python
# camera_infer_main.py

import os
import time
import numpy as np
import cv2
import tflite_runtime.interpreter as tflite
from picamera2 import Picamera2

# ---------------- 기본 설정 ----------------
IMG = 240
model = "./model.tflite"
labels = ["green", "left", "middle", "noline", "red", "right"]  # 모델 학습 시 클래스 순서와 맞게

# TFLite 인터프리터 생성
interpreter = tflite.Interpreter(model_path=model)
interpreter.allocate_tensors()
inp = interpreter.get_input_details()[0]
out = interpreter.get_output_details()[0]

print("Input tensor info:", inp)
print("Output tensor info:", out)

# 채널 수 체크(마지막 차원)
input_shape = inp["shape"]
if len(input_shape) != 4 or input_shape[-1] != 3:
    raise RuntimeError(f"이 코드는 [1, {IMG}, {IMG}, 3] 형태의 입력을 가정합니다. 현재 입력 shape={input_shape}")

# 헤드리스 환경(SSH 등) 여부
HEADLESS = (os.environ.get("DISPLAY", "") == "")
if HEADLESS:
    print("HEADLESS 모드: DISPLAY가 없으므로 imshow를 사용하지 않습니다.")

# ---------------- 전처리 함수 ----------------
def preprocess_frame(frame_rgb):
    """
    frame_rgb: H x W x 3 (RGB)
    모델 입력 형태: [1, IMG, IMG, 3], float32, [0, 1]
    """
    img = cv2.resize(frame_rgb, (IMG, IMG), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    return img[None, ...]  # 배치 차원 추가


# ---------------- 카메라 설정 ----------------
picam2 = Picamera2()

# RGB888(3채널)로 받도록 설정
config = picam2.create_preview_configuration(
    main={"format": "RGB888", "size": (640, 480)}
)
picam2.configure(config)
picam2.start()

print("Camera started.")

# ---------------- 메인 루프 ----------------
try:
    while True:
        # 1) 카메라에서 프레임 읽기 (RGB888, 3채널)
        frame = picam2.capture_array()  # shape: (480, 640, 3), RGB

        # 2) 전처리 후 TFLite 입력으로 설정
        x = preprocess_frame(frame)
        interpreter.set_tensor(inp["index"], x)

        # 3) 추론 실행
        t0 = time.time()
        interpreter.invoke()
        dt = (time.time() - t0) * 1e3  # ms

        # 4) 결과 가져오기
        probs = interpreter.get_tensor(out["index"])[0]  # shape: (num_classes,)
        pred_id = int(np.argmax(probs))
        pred = labels[pred_id]

        # 5) 결과 출력 (터미널)
        print(f"pred={pred}  probs={probs}  ({dt:.1f} ms)")

        # 6) 분류 결과에 따라 동작 (지금은 print만, 나중에 GPIO 연동)
        if pred == "forward":
            print(">> forward 동작 (모터 직진)")
            # TODO: GPIO 코드 작성
        elif pred == "left":
            print(">> left 동작 (좌회전)")
            # TODO: GPIO 코드 작성
        elif pred == "right":
            print(">> right 동작 (우회전)")
            # TODO: GPIO 코드 작성
        else:
            print(">> color/기타 상태")

        # 7) 화면에 띄우기 (GUI 있을 때만)
        if not HEADLESS:
            # frame은 RGB → OpenCV imshow는 BGR 사용
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # 좌상단에 예측 결과 표시
            text = f"{pred} ({dt:.1f} ms)"
            cv2.putText(
                frame,
                text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow("Pi Camera (TFLite inference)", frame)

            # q 키로 종료
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("q 입력으로 종료합니다.")
                break

        # 너무 바쁘지 않게 약간 쉬어주기 (필요시 줄이거나 없애도 됨)
        time.sleep(0.05)

except KeyboardInterrupt:
    print("KeyboardInterrupt로 종료합니다.")

finally:
    picam2.stop()
    if not HEADLESS:
        cv2.destroyAllWindows()
    print("카메라 및 윈도우 정리 완료.")
