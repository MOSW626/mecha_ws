import RPi.GPIO as GPIO
import time
import os
from multiprocessing import Process, Value, Lock
import ctypes
from collections import deque
import math

# ==================== 제어 파라미터 클래스 ====================
class ControlParams:
    """튜닝 가능한 모든 파라미터를 한 곳에 모음"""

    # PID Gains (RIGHT 모드) - 고속 주행 최적화
    Kp_r = 7.5   # 증가 (빠른 반응)
    Ki_r = 0.06  # 약간 증가
    Kd_r = 2.0   # 증가 (진동 억제 강화)

    # PID Gains (LEFT 모드) - 고속 주행 최적화
    Kp_l = 6.0   # 증가
    Ki_l = 0.04  # 약간 증가
    Kd_l = 1.8   # 증가

    # 기준 거리
    ref_distance_right = 15.0
    ref_distance_left = 15.0

    # 서보 기본 각도
    base_angle = 90.0
    angle_min = 45.0
    angle_max = 135.0

    # 속도 설정 (고속 주행 최적화)
    speed_base = 95.0          # 기본 속도 (증가)
    speed_max = 100.0          # 최대 속도 (증가)
    speed_min = 60.0           # 최소 속도 (증가 - 곡선에서도 빠르게)

    # 속도 조절 계수 (각도 변화에 따른 속도 감소) - 감소하여 더 빠르게
    speed_angle_factor_r = 0.35  # 감소 (곡선에서도 더 빠르게)
    speed_angle_factor_l = 0.30  # 감소

    # 적응형 속도 파라미터
    straight_speed_boost = 5.0    # 직선 구간 속도 부스트
    curve_detection_threshold = 3.0  # 곡선 감지 임계값 (각도 변화)

    # 필터링 파라미터 (고속 주행 최적화)
    smoothing_alpha = 0.80     # 증가 (더 빠른 반응)
    median_filter_size = 3     # 감소 (더 빠른 반응, 5->3)

    # 모드 전환 파라미터
    right_to_left_thresh = 55.0    # RIGHT -> LEFT 전환 임계값
    left_to_right_thresh = 25.0    # LEFT -> RIGHT 전환 임계값
    mode_switch_delay = 0.25       # 모드 전환 지연 시간 (초)

    # 안전 파라미터
    min_safe_distance = 5.0        # 최소 안전 거리 (cm)
    max_safe_distance = 100.0       # 최대 유효 거리 (cm)

    # 적분 제한 (Windup 방지)
    integral_max = 50.0
    integral_min = -50.0

    # 센서 읽기 주기 (최적화)
    sensor_read_period = 0.002      # 약 500 Hz (증가)
    control_period = 0.0005         # 약 2000 Hz (증가 - 더 빠른 제어)

# ==================== GPIO 핀 설정 ====================
"""
하드웨어 연결 가이드:

모터 제어:
- DIR_PIN (GPIO 16): 모터 방향 제어 핀
- PWM_PIN (GPIO 12): 모터 속도 제어 (하드웨어 PWM 채널 0 사용 가능)
  * GPIO 12는 하드웨어 PWM 지원 (더 정확하고 CPU 부하 적음)

서보 제어:
- SERVO_PIN (GPIO 13): 서보 각도 제어 (하드웨어 PWM 채널 1 사용 가능)
  * GPIO 13은 하드웨어 PWM 지원

초음파 센서 (HC-SR04):
- TRIG_LEFT (GPIO 17): 왼쪽 센서 트리거 핀
- ECHO_LEFT (GPIO 27): 왼쪽 센서 에코 핀 (풀다운 저항 권장: 10kΩ)
- TRIG_RIGHT (GPIO 5): 오른쪽 센서 트리거 핀
- ECHO_RIGHT (GPIO 6): 오른쪽 센서 에코 핀 (풀다운 저항 권장: 10kΩ)

하드웨어 권장사항:
1. 전원 공급: 모터와 서보는 별도 전원 권장 (5V 서보, 모터는 배터리)
2. 전류 용량: 모터 드라이버는 최소 2A 이상 권장
3. 노이즈 필터: 모터 근처에 100nF 커패시터 추가 권장
4. 배선: 센서 케이블은 모터 케이블과 분리 (노이즈 감소)
5. 그라운드: 모든 그라운드를 공통으로 연결
"""
DIR_PIN   = 16
PWM_PIN   = 12  # 하드웨어 PWM 채널 0
SERVO_PIN = 13  # 하드웨어 PWM 채널 1

MOTOR_FREQ = 1000  # 모터 PWM 주파수 (1000Hz = 부드러운 제어)
SERVO_FREQ = 50    # 서보 PWM 주파수 (50Hz = 표준 서보 주파수)
SERVO_MAX_DUTY = 12  # 서보 최대 Duty Cycle (%)
SERVO_MIN_DUTY = 3   # 서보 최소 Duty Cycle (%)

TRIG_LEFT  = 17
ECHO_LEFT  = 27  # 풀다운 저항 권장
TRIG_RIGHT = 5
ECHO_RIGHT = 6   # 풀다운 저항 권장

# 전역 PWM 객체
motor_pwm = None
servo_pwm = None

# ==================== 코어 affinity 설정 ====================
def set_affinity(core_ids):
    try:
        os.sched_setaffinity(0, core_ids)
        print(f"Process {os.getpid()} pinned to cores {core_ids}")
    except AttributeError:
        print("sched_setaffinity not supported on this system")

# ==================== 향상된 필터링 클래스 ====================
class DistanceFilter:
    """중앙값 필터 + EMA 필터 조합"""

    def __init__(self, alpha=0.75, median_size=5):
        self.alpha = alpha
        self.median_buffer = deque(maxlen=median_size)
        self.last_value = None
        self.timeout_code = 8787

    def smooth(self, raw_value):
        """이중 필터링: 중앙값 필터 + EMA"""
        # 타임아웃 처리
        if raw_value == self.timeout_code:
            return self.last_value

        if raw_value is None:
            return self.last_value

        # 거리 클리핑
        raw_value = max(ControlParams.min_safe_distance,
                       min(raw_value, ControlParams.max_safe_distance))

        # 중앙값 필터
        self.median_buffer.append(raw_value)
        if len(self.median_buffer) < self.median_buffer.maxlen:
            median_value = raw_value
        else:
            sorted_buffer = sorted(self.median_buffer)
            median_value = sorted_buffer[len(sorted_buffer) // 2]

        # EMA 필터
        if self.last_value is None:
            self.last_value = median_value
        else:
            self.last_value = (self.alpha * median_value +
                             (1.0 - self.alpha) * self.last_value)

        return self.last_value

# ==================== 초음파 센서 함수 ====================
def sample_distance(trig, echo):
    """
    초음파 센서로 거리 측정

    하드웨어 최적화:
    - TRIG 펄스는 최소 10us 필요 (HC-SR04 사양)
    - GPIO 핀은 하드웨어 PWM 사용 가능하면 더 정확함
    - ECHO 핀은 풀다운 저항 권장 (노이즈 감소)
    """
    GPIO.output(trig, True)
    time.sleep(0.00001)  # 10us (HC-SR04 최소 요구사항)
    GPIO.output(trig, False)

    t0 = time.time()
    while GPIO.input(echo) == 0:
        if time.time() - t0 > 0.02:
            return None
    start = time.time()

    while GPIO.input(echo) == 1:
        if time.time() - start > 0.02:
            return 8787  # 타임아웃 코드
    end = time.time()

    dist = (end - start) * 34300.0 / 2.0
    return dist

# ==================== PID 제어기 클래스 ====================
class PIDController:
    """PID 제어기 (모드별로 별도 인스턴스)"""

    def __init__(self, Kp, Ki, Kd, integral_max=50.0, integral_min=-50.0):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.integral_max = integral_max
        self.integral_min = integral_min

        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_time = time.time()

    def reset(self):
        """PID 상태 초기화"""
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_time = time.time()

    def compute(self, error, dt=None):
        """PID 출력 계산 (고정 주기 최적화)"""
        if dt is None:
            now = time.time()
            dt = now - self.prev_time
            if dt <= 0 or dt > 0.01:  # 비정상적인 dt 방지
                dt = ControlParams.control_period
            self.prev_time = now
        else:
            # 고정 주기 사용 (더 효율적)
            pass

        # 비례 항
        P = self.Kp * error

        # 적분 항 (Windup 방지)
        self.integral += error * dt
        self.integral = max(self.integral_min,
                           min(self.integral_max, self.integral))
        I = self.Ki * self.integral

        # 미분 항
        derivative = (error - self.prev_error) / dt
        D = self.Kd * derivative

        # 출력
        output = P + I + D

        # 상태 업데이트
        self.prev_error = error

        return output

# ==================== 모터 / 서보 제어 ====================
def init_motor_servo():
    """
    모터 및 서보 초기화

    하드웨어 최적화:
    - MOTOR_FREQ = 1000Hz: 모터 제어에 적합한 주파수 (너무 높으면 전류 소비 증가)
    - SERVO_FREQ = 50Hz: 표준 서보 주파수 (20ms 주기)
    - GPIO 핀은 하드웨어 PWM 핀 사용 권장 (GPIO 12, 13, 18, 19)
      - GPIO 12: 하드웨어 PWM 채널 0
      - GPIO 13: 하드웨어 PWM 채널 1
    - 소프트웨어 PWM은 CPU 부하가 있지만 유연함
    """
    global motor_pwm, servo_pwm
    GPIO.setmode(GPIO.BCM)

    # GPIO 핀 초기 상태 설정 (노이즈 방지)
    GPIO.setup(DIR_PIN, GPIO.OUT, initial=GPIO.LOW)
    GPIO.setup(PWM_PIN, GPIO.OUT, initial=GPIO.LOW)
    GPIO.setup(SERVO_PIN, GPIO.OUT, initial=GPIO.LOW)

    motor_pwm = GPIO.PWM(PWM_PIN, MOTOR_FREQ)
    servo_pwm = GPIO.PWM(SERVO_PIN, SERVO_FREQ)
    motor_pwm.start(0)
    servo_pwm.start(0)

    # 초기화 후 짧은 대기 (하드웨어 안정화)
    time.sleep(0.01)

def set_servo_angle(degree):
    """
    서보 각도 설정 (45~135도 제한)

    하드웨어 최적화:
    - 서보 모터는 50Hz PWM 사용 (표준)
    - Duty Cycle 범위: 3~12% (약 0.6ms~2.4ms 펄스 폭)
    - 각도 제한으로 서보 모터 보호
    - 고속 주행 시 서보 응답 속도가 중요 (일반 서보: 0.1~0.2초)
    - 금속 기어 서보 사용 권장 (내구성 향상)
    """
    degree = max(ControlParams.angle_min,
                min(ControlParams.angle_max, degree))
    duty = (SERVO_MIN_DUTY +
           (degree * (SERVO_MAX_DUTY - SERVO_MIN_DUTY) / 180.0))
    servo_pwm.ChangeDutyCycle(duty)

def move_forward(speed):
    """
    전진 제어

    하드웨어 최적화:
    - DIR_PIN: 모터 방향 제어 (HIGH=전진, LOW=후진)
    - PWM Duty Cycle: 0~100% (실제 모터 전압 비율)
    - 고속 주행 시 모터 드라이버 과열 주의
    - 전류 소비 모니터링 권장 (퓨즈/차단기 필요)
    - 모터 드라이버는 최소 2A 이상 권장
    """
    GPIO.output(DIR_PIN, GPIO.HIGH)
    speed = max(0.0, min(100.0, speed))
    motor_pwm.ChangeDutyCycle(speed)

def stop_motor():
    """모터 정지"""
    motor_pwm.ChangeDutyCycle(0)

# ==================== 속도 계산 함수 (고속 최적화) ====================
def calculate_speed(angle_output, mode='RIGHT', prev_angle_output=None):
    """각도 출력에 따른 속도 계산 (고속 주행 최적화)"""
    abs_output = abs(angle_output)

    if mode == 'RIGHT':
        factor = ControlParams.speed_angle_factor_r
    else:
        factor = ControlParams.speed_angle_factor_l

    # 직선 구간 감지 (각도 변화가 작으면)
    is_straight = False
    if prev_angle_output is not None:
        angle_change = abs(abs_output - abs(prev_angle_output))
        if angle_change < ControlParams.curve_detection_threshold:
            is_straight = True

    # 속도 감소 계산 (더 공격적인 곡선)
    if abs_output < ControlParams.curve_detection_threshold:
        # 직선 구간: 최대 속도
        speed = ControlParams.speed_max
        if is_straight:
            speed = min(100.0, speed + ControlParams.straight_speed_boost)
    else:
        # 곡선 구간: 각도에 따라 감소 (더 완만한 감소)
        speed_reduction = factor * abs_output
        speed = ControlParams.speed_base - speed_reduction

    # 속도 제한
    speed = max(ControlParams.speed_min,
               min(ControlParams.speed_max, speed))

    return speed

# ==================== 센서 프로세스 ====================
def sensor_process(left_val, right_val, lock):
    """
    센서 읽기 전용 프로세스 (코어 1)

    하드웨어 최적화:
    - 코어 1에 고정하여 센서 읽기 전용 (CPU 캐시 효율)
    - GPIO 핀은 초기 상태 설정으로 노이즈 방지
    - ECHO 핀은 풀다운 저항으로 플로팅 방지
    - 센서 간 간섭 방지를 위해 순차 읽기 (병렬 가능하지만 노이즈 위험)
    """
    set_affinity({1})

    GPIO.setmode(GPIO.BCM)
    # TRIG 핀은 초기 LOW 상태 (노이즈 방지)
    GPIO.setup([TRIG_LEFT, TRIG_RIGHT], GPIO.OUT, initial=GPIO.LOW)
    # ECHO 핀은 입력, 풀다운 저항 권장 (하드웨어 또는 소프트웨어)
    GPIO.setup([ECHO_LEFT, ECHO_RIGHT], GPIO.IN, pull_up_down=GPIO.PUD_DOWN)

    # 필터 초기화
    left_filter = DistanceFilter(
        alpha=ControlParams.smoothing_alpha,
        median_size=ControlParams.median_filter_size
    )
    right_filter = DistanceFilter(
        alpha=ControlParams.smoothing_alpha,
        median_size=ControlParams.median_filter_size
    )

    try:
        print("센서 프로세스 시작")

        while True:
            # 센서 읽기
            raw_left = sample_distance(TRIG_LEFT, ECHO_LEFT)
            raw_right = sample_distance(TRIG_RIGHT, ECHO_RIGHT)

            # 필터링
            left = left_filter.smooth(raw_left)
            right = right_filter.smooth(raw_right)

            # 공유 메모리 갱신
            with lock:
                left_val.value = float(left) if left is not None else 0.0
                right_val.value = float(right) if right is not None else 0.0

            time.sleep(ControlParams.sensor_read_period)

    except KeyboardInterrupt:
        print("센서 프로세스 종료")
    finally:
        GPIO.cleanup()
        print("센서 프로세스 GPIO cleanup 완료")

# ==================== 메인 제어 프로세스 ====================
def main_control(left_val, right_val, lock):
    """메인 제어 프로세스 (코어 0)"""
    set_affinity({0})

    global motor_pwm, servo_pwm
    init_motor_servo()

    # 초기 상태
    set_servo_angle(ControlParams.base_angle)
    stop_motor()

    print("메인 제어 프로세스 시작")

    # PID 제어기 초기화
    pid_right = PIDController(
        ControlParams.Kp_r, ControlParams.Ki_r, ControlParams.Kd_r,
        ControlParams.integral_max, ControlParams.integral_min
    )
    pid_left = PIDController(
        ControlParams.Kp_l, ControlParams.Ki_l, ControlParams.Kd_l,
        ControlParams.integral_max, ControlParams.integral_min
    )

    # 모드 상태
    mode = 'RIGHT'
    mode_switch_time = None
    right_over_start = None

    # 속도 제어를 위한 이전 출력 저장
    prev_output = 0.0

    # 로깅 (빈도 감소로 성능 향상)
    log_counter = 0
    log_interval = 100  # 100번마다 출력 (50->100)

    # 고정 제어 주기 (성능 최적화)
    control_dt = ControlParams.control_period
    next_control_time = time.time()

    try:
        while True:
            # 센서 값 읽기
            with lock:
                left = left_val.value
                right = right_val.value

            # 유효성 검사
            if left <= 0.0 or right <= 0.0:
                time.sleep(control_dt)
                next_control_time += control_dt
                continue

            now = time.time()

            # ========== 모드 전환 로직 (개선) ==========
            if mode == 'RIGHT':
                # RIGHT -> LEFT 전환 조건
                if right > ControlParams.right_to_left_thresh:
                    if right_over_start is None:
                        right_over_start = now
                    elif (now - right_over_start) > ControlParams.mode_switch_delay:
                        mode = 'LEFT'
                        right_over_start = None
                        pid_left.reset()  # PID 상태 초기화
                        print(f"[MODE] RIGHT -> LEFT (L:{left:.1f}, R:{right:.1f})")
                else:
                    right_over_start = None  # 조건 불만족 시 리셋
            else:  # mode == 'LEFT'
                # LEFT -> RIGHT 전환 조건 (히스테리시스)
                if right < ControlParams.left_to_right_thresh:
                    mode = 'RIGHT'
                    pid_right.reset()  # PID 상태 초기화
                    print(f"[MODE] LEFT -> RIGHT (L:{left:.1f}, R:{right:.1f})")

            # ========== 모드별 PID 제어 ==========
            if mode == 'RIGHT':
                error = ControlParams.ref_distance_right - right
                output = pid_right.compute(error, control_dt)  # 고정 주기 사용

                angle_cmd = ControlParams.base_angle - output
                speed_cmd = calculate_speed(output, 'RIGHT', prev_output)
            else:  # mode == 'LEFT'
                error = ControlParams.ref_distance_left - left
                output = pid_left.compute(error, control_dt)  # 고정 주기 사용

                angle_cmd = ControlParams.base_angle - output
                speed_cmd = calculate_speed(output, 'LEFT', prev_output)

            # 이전 출력 저장
            prev_output = output

            # 각도 제한
            angle_cmd = max(ControlParams.angle_min,
                           min(ControlParams.angle_max, angle_cmd))

            # 안전 검사: 너무 가까우면 감속 (더 공격적인 임계값)
            min_dist = min(left, right)
            if min_dist < ControlParams.min_safe_distance:
                speed_cmd *= 0.6  # 속도 40% 감소 (50%->60%로 완화)
            elif min_dist < ControlParams.min_safe_distance + 2.0:
                speed_cmd *= 0.85  # 약간 감속

            # 제어 출력
            set_servo_angle(angle_cmd)
            move_forward(speed_cmd)

            # 로깅
            log_counter += 1
            if log_counter >= log_interval:
                print(f"[{mode}] L:{left:.1f} R:{right:.1f} "
                      f"Err:{error:.2f} Angle:{angle_cmd:.1f} "
                      f"Speed:{speed_cmd:.1f}")
                log_counter = 0

            # 고정 주기 제어 (더 정확한 타이밍)
            next_control_time += control_dt
            sleep_time = next_control_time - time.time()
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                # 제어가 지연되면 다음 주기로
                next_control_time = time.time() + control_dt

    except KeyboardInterrupt:
        print("메인 제어 종료")
    finally:
        # 안전 정지
        try:
            stop_motor()
            set_servo_angle(ControlParams.base_angle)
            if motor_pwm is not None:
                motor_pwm.stop()
            if servo_pwm is not None:
                servo_pwm.stop()
        except Exception as e:
            print(f"PWM stop 중 오류: {e}")
        GPIO.cleanup()
        print("메인 제어 GPIO cleanup 완료")

# ==================== 엔트리 포인트 ====================
if __name__ == "__main__":
    print("=" * 50)
    print("고속 최적화 벽 추종 시스템 시작")
    print("=" * 50)
    print(f"PID RIGHT: Kp={ControlParams.Kp_r}, Ki={ControlParams.Ki_r}, Kd={ControlParams.Kd_r}")
    print(f"PID LEFT:  Kp={ControlParams.Kp_l}, Ki={ControlParams.Ki_l}, Kd={ControlParams.Kd_l}")
    print(f"기본 속도: {ControlParams.speed_base} (최대: {ControlParams.speed_max})")
    print(f"제어 주기: {ControlParams.control_period*1000:.1f}ms ({1/ControlParams.control_period:.0f} Hz)")
    print(f"센서 주기: {ControlParams.sensor_read_period*1000:.1f}ms ({1/ControlParams.sensor_read_period:.0f} Hz)")
    print("=" * 50)

    left_val = Value(ctypes.c_double, 0.0)
    right_val = Value(ctypes.c_double, 0.0)
    lock = Lock()

    # 센서 프로세스 시작
    p_sensor = Process(target=sensor_process, args=(left_val, right_val, lock))
    p_sensor.start()

    try:
        # 메인 제어 실행
        main_control(left_val, right_val, lock)
    except KeyboardInterrupt:
        print("전체 시스템 종료")
    finally:
        # 센서 프로세스 정리
        if p_sensor.is_alive():
            p_sensor.terminate()
            p_sensor.join()
        print("전체 종료 완료")

