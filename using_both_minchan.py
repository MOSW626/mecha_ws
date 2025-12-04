import RPi.GPIO as GPIO
import time
import os
from multiprocessing import Process, Value, Lock
import ctypes

# ==================== 제어 파라미터 ====================
# PD Gains (초음파 모드용)
Kp = 1.2

ref_distance_difference = -12.0
base_angle = 90.0

speed_angle_diff = 0.6

# 속도 설정
SPEED_ULTRASONIC = 100.0

# Distance Clipping values
MIN_CM, MAX_CM = 3.0, 150.0
ALPHA = 0.5 #85

# ==================== GPIO 핀 설정 ====================
# 모터 / 서보
DIR_PIN   = 16
PWM_PIN   = 12
SERVO_PIN = 13

MOTOR_FREQ = 1000
SERVO_FREQ = 50
SERVO_MAX_DUTY = 12
SERVO_MIN_DUTY = 3

# 초음파 센서
TRIG_LEFT  = 17
ECHO_LEFT  = 27
TRIG_RIGHT = 5
ECHO_RIGHT = 6

# 전역 PWM 객체 (메인 프로세스에서만 사용)
motor_pwm = None
servo_pwm = None

# ==================== 코어 affinity 설정 함수 ====================
def set_affinity(core_ids):
    try:
        os.sched_setaffinity(0, core_ids)
        print(f"Process {os.getpid()} pinned to cores {core_ids}")
    except AttributeError:
        print("sched_setaffinity not supported on this system")

# ==================== 초음파 센서 함수 ====================
def sample_distance(trig, echo):
    GPIO.output(trig, True)
    time.sleep(0.00001)  # 10us
    GPIO.output(trig, False)

    t0 = time.time()
    while GPIO.input(echo) == 0:
        if time.time() - t0 > 0.02:   # 20ms 타임아웃
            return None
    start = time.time()

    while GPIO.input(echo) == 1:
        if time.time() - start > 0.02:
            return 8787   # 타임아웃 코드
    end = time.time()

    dist = (end - start) * 34300.0 / 2.0
    dist = max(MIN_CM, min(dist, MAX_CM))
    return dist

def read_stable(trig, echo):
    # 필요하면 평균/다중 샘플링 로직 추가 가능  
    return sample_distance(trig, echo)

def smooth(prev_value, new_value, alpha=ALPHA):
    if new_value == 8787:
        return None
    if new_value is None:
        return prev_value
    if prev_value is None:
        return new_value

    if abs(prev_value - new_value) > 50.0:
        return prev_value
    else:
        return alpha * new_value + (1.0 - alpha) * prev_value

# ==================== 모터 / 서보 제어 함수 ====================
def init_motor_servo():
    global motor_pwm, servo_pwm
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(DIR_PIN, GPIO.OUT)
    GPIO.setup(PWM_PIN, GPIO.OUT)
    GPIO.setup(SERVO_PIN, GPIO.OUT)

    motor_pwm = GPIO.PWM(PWM_PIN, MOTOR_FREQ)
    servo_pwm = GPIO.PWM(SERVO_PIN, SERVO_FREQ)
    motor_pwm.start(0)
    servo_pwm.start(0)

def set_servo_angle(degree):
    degree = max(45.0, min(135.0, degree))
    duty = SERVO_MIN_DUTY + (degree * (SERVO_MAX_DUTY - SERVO_MIN_DUTY) / 180.0)
    servo_pwm.ChangeDutyCycle(duty)

def move_forward(speed):
    GPIO.output(DIR_PIN, GPIO.HIGH)
    motor_pwm.ChangeDutyCycle(max(0.0, min(100.0, speed)))

def move_backward(speed):
    GPIO.output(DIR_PIN, GPIO.LOW)
    motor_pwm.ChangeDutyCycle(max(0.0, min(100.0, speed)))

def stop_motor():
    motor_pwm.ChangeDutyCycle(0)

# ==================== 센서 프로세스 (코어 1에 핀 고정) ====================
def sensor_process(left_val, right_val, lock):
    set_affinity({1})  # 코어 번호는 상황에 맞게 조정

    GPIO.setmode(GPIO.BCM)
    GPIO.setup([TRIG_LEFT, TRIG_RIGHT], GPIO.OUT)
    GPIO.setup([ECHO_LEFT, ECHO_RIGHT], GPIO.IN)

    last_left = None
    last_right = None

    try:
        print("메인 제어 프로세스 시작")

        while True:
            raw_left  = read_stable(TRIG_LEFT, ECHO_LEFT)
            raw_right = read_stable(TRIG_RIGHT, ECHO_RIGHT)

            left  = smooth(last_left, raw_left)
            right = smooth(last_right, raw_right)

            last_left, last_right = raw_left, raw_right

            # 공유 메모리 갱신
            with lock:
                left_val.value  = float(left) if left is not None else 0.0
                right_val.value = float(right) if right is not None else 0.0

            # 초음파 센서 보호 + CPU 사용률 조절
            time.sleep(0.002)  # ≈ 50 Hz 정도

    except KeyboardInterrupt:
        print("센서 프로세스 종료 (KeyboardInterrupt)")
    finally:
        GPIO.cleanup()
        print("센서 프로세스 GPIO cleanup 완료")

# ==================== 메인 제어 프로세스 (코어 0에 핀 고정) ====================
def main_control(left_val, right_val, lock):
    set_affinity({0})

    global motor_pwm, servo_pwm
    init_motor_servo()

    # 초기 상태
    set_servo_angle(base_angle)
    stop_motor()

    print("메인 제어 프로세스 시작")

    log = 0

    try:
        while True:
            with lock:
                left  = left_val.value
                right = right_val.value

            if left <= 0.0 or right <= 0.0:
                time.sleep(0.001)
                continue

            error = ref_distance_difference - (right - left)
            output = Kp * error


            angle_cmd = base_angle - output
            angle_cmd = max(45.0, min(135.0, angle_cmd))

            speed_cmd = SPEED_ULTRASONIC - speed_angle_diff * abs(output)

            if log == 20:
                print('right : ', right, 'left : ', left, 'angle_cmd : ', angle_cmd, 'speed_cmd : ', speed_cmd)
                log = 0
            
            log += 1

            if speed_cmd < 0.0:
                speed_cmd = 0.0

            set_servo_angle(angle_cmd)
            move_forward(speed_cmd)

            time.sleep(0.001)


    except KeyboardInterrupt:
        print("메인 제어 종료 (KeyboardInterrupt)")
    finally:
        # 안전 정지
        try:
            stop_motor()
            set_servo_angle(base_angle)
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
    left_val  = Value(ctypes.c_double, 0.0)
    right_val = Value(ctypes.c_double, 0.0)
    lock = Lock()

    # 센서 프로세스 시작
    p_sensor = Process(target=sensor_process, args=(left_val, right_val, lock))
    p_sensor.start()

    try:
        # 현재 프로세스에서 메인 제어 수행
        main_control(left_val, right_val, lock)
    except KeyboardInterrupt:
        print("전체 시스템 KeyboardInterrupt")
    finally:
        # 센서 프로세스 정리
        if p_sensor.is_alive():
            p_sensor.terminate()
            p_sensor.join()
        print("전체 종료")
