#!/usr/bin/env python3
"""
라즈베리파이용 경량 추론 스크립트
ONNX Runtime을 사용하여 학습된 정책을 실행
gym, pygame, matplotlib 등의 무거운 라이브러리 의존성 없음
"""

import numpy as np
import onnxruntime as ort
import time
from typing import Tuple, Optional


class HardwareInterface:
    """
    실제 하드웨어 인터페이스 (더미 클래스)
    실제 라즈베리파이에서는 GPIO 및 카메라 인터페이스로 교체
    """

    def __init__(self):
        """
        하드웨어 초기화
        실제 구현에서는 GPIO 및 카메라 설정
        """
        # GPIO 설정 예시 (주석 처리됨)
        # import RPi.GPIO as GPIO
        # GPIO.setmode(GPIO.BCM)
        # GPIO.setup([SERVO_PIN, MOTOR_PIN], GPIO.OUT)
        # self.servo_pwm = GPIO.PWM(SERVO_PIN, 50)
        # self.motor_pwm = GPIO.PWM(MOTOR_PIN, 1000)

        # 초음파 센서 핀 설정 예시
        # self.TRIG_PINS = [17, 5, 6]  # 왼쪽, 중앙, 오른쪽
        # self.ECHO_PINS = [27, 6, 13]

        # 카메라 설정 예시
        # from picamera2 import Picamera2
        # self.camera = Picamera2()
        # self.camera.start()

        print("[하드웨어] 하드웨어 인터페이스 초기화 완료 (시뮬레이션 모드)")

    def get_ultrasonic_distances(self, num_sensors: int = 2) -> np.ndarray:
        """
        초음파 센서로부터 거리 데이터 읽기

        Args:
            num_sensors: 센서 개수

        Returns:
            센서 거리 배열 (cm)
        """
        # 실제 구현 예시:
        # distances = []
        # for i in range(num_sensors):
        #     # TRIG 핀으로 펄스 전송
        #     GPIO.output(self.TRIG_PINS[i], True)
        #     time.sleep(0.00001)
        #     GPIO.output(self.TRIG_PINS[i], False)
        #
        #     # ECHO 핀에서 펄스 대기
        #     while GPIO.input(self.ECHO_PINS[i]) == 0:
        #         pulse_start = time.time()
        #     while GPIO.input(self.ECHO_PINS[i]) == 1:
        #         pulse_end = time.time()
        #
        #     pulse_duration = pulse_end - pulse_start
        #     distance = pulse_duration * 17150  # cm 단위
        #     distances.append(np.clip(distance, 3.0, 150.0))

        # 더미 데이터 (테스트용)
        distances = np.random.uniform(20.0, 100.0, num_sensors)
        return distances.astype(np.float32)

    def get_camera_line_error(self) -> float:
        """
        카메라로부터 라인 트레이싱 에러 계산

        Returns:
            정규화된 크로스 트랙 에러 [-1, 1]
            -1: 왼쪽으로 많이 벗어남
             0: 중심
            +1: 오른쪽으로 많이 벗어남
        """
        # 실제 구현 예시:
        # frame = self.camera.capture_array()
        #
        # # 이미지 전처리
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        #
        # # ROI 설정 (하단 부분만)
        # h, w = binary.shape
        # roi = binary[int(h*0.6):, :]
        #
        # # 라인 검출 (HoughLines 또는 컨투어)
        # # ... 라인 검출 로직 ...
        #
        # # 중심선과의 에러 계산
        # center_x = w / 2
        # line_x = detected_line_x
        # error = (line_x - center_x) / (w / 2)
        # error = np.clip(error, -1.0, 1.0)
        #
        # return error

        # 더미 데이터 (테스트용)
        error = np.random.uniform(-0.3, 0.3)
        return float(np.clip(error, -1.0, 1.0))

    def set_servo_angle(self, angle_degrees: float):
        """
        서보 모터 제어 (조향각 설정)

        Args:
            angle_degrees: 조향각 (도), -20 ~ +20 범위
        """
        # 실제 구현 예시:
        # angle_degrees = np.clip(angle_degrees, -20.0, 20.0)
        # duty_cycle = 7.5 + (angle_degrees / 90.0) * 5.0  # 90도 기준
        # self.servo_pwm.ChangeDutyCycle(duty_cycle)

        print(f"[제어] 조향각: {angle_degrees:.2f}도")

    def set_motor_throttle(self, throttle: float):
        """
        모터 제어 (스로틀 설정)

        Args:
            throttle: 스로틀 값 [0, 1], 0: 정지, 1: 최대 속도
        """
        # 실제 구현 예시:
        # throttle = np.clip(throttle, 0.0, 1.0)
        # duty_cycle = throttle * 100.0
        # self.motor_pwm.ChangeDutyCycle(duty_cycle)

        print(f"[제어] 스로틀: {throttle:.2%}")

    def cleanup(self):
        """
        하드웨어 정리
        """
        # GPIO.cleanup()
        # self.camera.stop()
        print("[하드웨어] 정리 완료")


class RacingCarInference:
    """
    ONNX 모델을 사용한 추론 클래스
    """

    def __init__(
        self,
        model_path: str,
        num_ultrasonic_sensors: int = 2,  # 실제 하드웨어: 2개 센서
        sensor_max_range: float = 150.0,
        max_steering_angle: float = 20.0,
    ):
        """
        초기화

        Args:
            model_path: ONNX 모델 파일 경로
            num_ultrasonic_sensors: 초음파 센서 개수
            sensor_max_range: 센서 최대 감지 거리 (cm)
            max_steering_angle: 최대 조향각 (도)
        """
        self.num_ultrasonic_sensors = num_ultrasonic_sensors
        self.sensor_max_range = sensor_max_range
        self.max_steering_angle = max_steering_angle

        # ONNX Runtime 세션 생성
        print(f"[모델] ONNX 모델 로드 중: {model_path}")
        try:
            # CPU 최적화 옵션 설정 (라즈베리파이 성능 향상)
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_options.intra_op_num_threads = 2  # CPU 코어 수에 맞게 조정

            self.session = ort.InferenceSession(
                model_path,
                sess_options=sess_options,
                providers=['CPUExecutionProvider']  # 라즈베리파이는 CPU만 사용
            )

            # 입력/출력 정보 확인
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            input_shape = self.session.get_inputs()[0].shape

            print(f"[모델] 모델 로드 완료")
            print(f"  - 입력 이름: {self.input_name}")
            print(f"  - 출력 이름: {self.output_name}")
            print(f"  - 입력 shape: {input_shape}")

        except Exception as e:
            raise RuntimeError(f"모델 로드 실패: {e}")

    def preprocess_observation(
        self,
        ultrasonic_distances: np.ndarray,
        camera_line_error: float
    ) -> np.ndarray:
        """
        관측값 전처리 및 정규화

        Args:
            ultrasonic_distances: 초음파 센서 거리 배열 (cm)
            camera_line_error: 카메라 라인 에러 [-1, 1]

        Returns:
            정규화된 관측값 벡터
        """
        # 거리 정규화 [0, 1]
        normalized_distances = ultrasonic_distances / self.sensor_max_range
        normalized_distances = np.clip(normalized_distances, 0.0, 1.0)

        # 관측값 결합
        observation = np.concatenate([
            normalized_distances,
            [camera_line_error]
        ]).astype(np.float32)

        # 배치 차원 추가 (1, N+1)
        observation = observation[np.newaxis, :]

        return observation

    def predict(self, observation: np.ndarray) -> Tuple[float, float]:
        """
        모델 추론 실행

        Args:
            observation: 정규화된 관측값 (1, N+1)

        Returns:
            (조향각, 스로틀) 튜플
            - 조향각: 정규화된 값 [-1, 1]
            - 스로틀: 정규화된 값 [0, 1]
        """
        # 추론 실행
        outputs = self.session.run(
            [self.output_name],
            {self.input_name: observation}
        )

        action = outputs[0][0]  # 배치 차원 제거

        steering_normalized = float(np.clip(action[0], -1.0, 1.0))
        throttle_normalized = float(np.clip(action[1], 0.0, 1.0))

        return steering_normalized, throttle_normalized

    def get_control_command(
        self,
        ultrasonic_distances: np.ndarray,
        camera_line_error: float
    ) -> Tuple[float, float]:
        """
        센서 데이터로부터 제어 명령 계산 (전체 파이프라인)

        Args:
            ultrasonic_distances: 초음파 센서 거리 배열 (cm)
            camera_line_error: 카메라 라인 에러 [-1, 1]

        Returns:
            (조향각_도, 스로틀) 튜플
        """
        # 전처리
        observation = self.preprocess_observation(
            ultrasonic_distances, camera_line_error
        )

        # 추론
        steering_normalized, throttle_normalized = self.predict(observation)

        # 조향각을 도 단위로 변환
        steering_angle = steering_normalized * self.max_steering_angle

        return steering_angle, throttle_normalized


def main(
    model_path: str = "racing_car_policy.onnx",
    num_sensors: int = 2,  # 실제 하드웨어: 2개 센서
    sensor_max_range: float = 150.0,
    max_steering_angle: float = 20.0,
    loop_freq: float = 10.0,  # Hz
    max_iterations: Optional[int] = None,
):
    """
    메인 추론 루프

    Args:
        model_path: ONNX 모델 경로
        num_sensors: 초음파 센서 개수
        sensor_max_range: 센서 최대 감지 거리 (cm)
        max_steering_angle: 최대 조향각 (도)
        loop_freq: 제어 루프 주파수 (Hz)
        max_iterations: 최대 반복 횟수 (None이면 무한 루프)
    """
    print("=" * 60)
    print("자율주행 레이싱 카 추론 시작")
    print("=" * 60)

    # 하드웨어 인터페이스 초기화
    hardware = HardwareInterface()

    # 추론 엔진 초기화
    try:
        inference = RacingCarInference(
            model_path=model_path,
            num_ultrasonic_sensors=num_sensors,
            sensor_max_range=sensor_max_range,
            max_steering_angle=max_steering_angle,
        )
    except Exception as e:
        print(f"[오류] 추론 엔진 초기화 실패: {e}")
        hardware.cleanup()
        return

    # 제어 루프
    dt = 1.0 / loop_freq
    iteration = 0
    total_inference_time = 0.0

    print(f"\n[제어 루프] 시작 (주파수: {loop_freq} Hz)")
    print("Ctrl+C로 종료할 수 있습니다.\n")

    try:
        while max_iterations is None or iteration < max_iterations:
            loop_start = time.time()

            # 센서 데이터 읽기
            ultrasonic_distances = hardware.get_ultrasonic_distances(num_sensors)
            camera_line_error = hardware.get_camera_line_error()

            # 추론 실행
            inference_start = time.time()
            steering_angle, throttle = inference.get_control_command(
                ultrasonic_distances, camera_line_error
            )
            inference_time = time.time() - inference_start
            total_inference_time += inference_time

            # 하드웨어 제어
            hardware.set_servo_angle(steering_angle)
            hardware.set_motor_throttle(throttle)

            # 통계 출력 (주기적으로)
            if iteration % 10 == 0:
                avg_inference_time = total_inference_time / (iteration + 1)
                print(f"\n[반복 {iteration}]")
                print(f"  센서 거리: {ultrasonic_distances}")
                print(f"  라인 에러: {camera_line_error:.3f}")
                print(f"  조향각: {steering_angle:.2f}도")
                print(f"  스로틀: {throttle:.2%}")
                print(f"  추론 시간: {inference_time*1000:.2f}ms (평균: {avg_inference_time*1000:.2f}ms)")

            iteration += 1

            # 주파수 유지
            elapsed = time.time() - loop_start
            sleep_time = dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                print(f"[경고] 루프 시간 초과: {elapsed*1000:.2f}ms > {dt*1000:.2f}ms")

    except KeyboardInterrupt:
        print("\n[종료] 사용자에 의해 중단되었습니다.")

    finally:
        # 정리
        print("\n[정리] 하드웨어 정리 중...")
        hardware.cleanup()

        if iteration > 0:
            avg_inference_time = total_inference_time / iteration
            print(f"\n[통계]")
            print(f"  총 반복 횟수: {iteration}")
            print(f"  평균 추론 시간: {avg_inference_time*1000:.2f}ms")
            print(f"  최대 루프 주파수: {1.0/avg_inference_time:.1f} Hz")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="라즈베리파이용 추론 스크립트")
    parser.add_argument(
        "--model",
        type=str,
        default="racing_car_policy.onnx",
        help="ONNX 모델 파일 경로",
    )
    parser.add_argument(
        "--sensors",
        type=int,
        default=2,
        help="초음파 센서 개수 (실제 하드웨어: 2개)",
    )
    parser.add_argument(
        "--sensor-range",
        type=float,
        default=150.0,
        help="센서 최대 감지 거리 (cm)",
    )
    parser.add_argument(
        "--max-steering",
        type=float,
        default=20.0,
        help="최대 조향각 (도)",
    )
    parser.add_argument(
        "--freq",
        type=float,
        default=10.0,
        help="제어 루프 주파수 (Hz)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="최대 반복 횟수 (기본값: 무한)",
    )

    args = parser.parse_args()

    main(
        model_path=args.model,
        num_sensors=args.sensors,
        sensor_max_range=args.sensor_range,
        max_steering_angle=args.max_steering,
        loop_freq=args.freq,
        max_iterations=args.iterations,
    )

