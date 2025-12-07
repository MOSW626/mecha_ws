import RPi.GPIO as GPIO
import time
import os
from multiprocessing import Process, Value, Lock
import ctypes

# lab time 8.6s ~ 8.7s, more then 8 lab

# ==================== 제어 파라미터 ====================
# PD Gains (초음파 모드용)
Kp = 2.55 #1,5
Kd = 0.0 #0.01

base_angle = 100.0

speed_angle_diff = 0.28 #0.45

# 속도 설정
SPEED_ULTRASONIC = 90.0

# Distance Clipping values
MIN_CM, MAX_CM = 3.0, 150.0

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

GPIO.setmode(GPIO.BCM)
GPIO.setup([TRIG_LEFT, TRIG_RIGHT], GPIO.OUT)
GPIO.setup([ECHO_LEFT, ECHO_RIGHT], GPIO.IN)

# ==================== 초음파 센서 함수 ====================
def sample_distance(trig, echo):
    GPIO.output(trig, True)
    time.sleep(0.00001)  # 10us
    GPIO.output(trig, False)

    t0 = time.time()
    while GPIO.input(echo) == 0:
        if time.time() - t0 > 0.02:   # 20ms 타임아웃
            return 7878
    start = time.time()

    while GPIO.input(echo) == 1:
        if time.time() - start > 0.02:
            return 7878
    end = time.time()

    dist = (end - start) * 34300.0 / 2.0
    dist = max(MIN_CM, min(dist, MAX_CM))
    return dist

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

def main_control():
    global motor_pwm, servo_pwm
    init_motor_servo()

    # 초기 상태
    set_servo_angle(base_angle)
    stop_motor()

    print("메인 제어 프로세스 시작")

    corner_time = None
    straight_time = None
    corner_thershold = 30.0
    corner_time_threshold = 0.2
    straight_time_threshold = 0.2

    
    speed_straight_diff = 0.0
    ref_distance_difference = 0.0

    track_change_lock = time.time()
    track_change_lock_time = 0.3

    straight_continous_time = None

    corner = True
    straight = False

    status_before = straight

    left_prev  = sample_distance(TRIG_LEFT, ECHO_LEFT)
    right_prev = sample_distance(TRIG_RIGHT, ECHO_RIGHT)

    log = 0

    try:
        while True:
            left  = sample_distance(TRIG_LEFT, ECHO_LEFT)
            right = sample_distance(TRIG_RIGHT, ECHO_RIGHT)

            if left == 7878 or left >= 120.0:
                left = left_prev
            else:
                left_prev = left
            
            if right == 7878 or right >= 120.0:
                right = right_prev
            else:
                right_prev = right

            difference = right - left

            if time.time() - track_change_lock > track_change_lock_time:
                if abs(difference) > corner_thershold:
                    if corner_time is None:
                        corner_time = time.time()
                    else:
                        if time.time() - corner_time > corner_time_threshold:
                            corner_time = None
                            straight_continous_time = None
                            speed_straight_diff = 0.0
                            status_before = corner
                            print('corner detected')

                            track_change_lock = time.time()
                else:
                    if status_before == corner:
                        if straight_time is None:
                            straight_time = time.time()
                        else:
                            if time.time() - straight_time > straight_time_threshold:
                                straight_time = None
                                straight_continous_time = time.time()
                                status_before = straight

                                print('straight path detected')

                                track_change_lock = time.time()

            if straight_continous_time is not None:
                if time.time() - straight_continous_time > 0.9:
                    speed_straight_diff -= 0.6
                    print('speed down')
                    if speed_straight_diff < -20.0:
                        speed_straight_diff = -20.0

            error = ref_distance_difference - (right - left)
            output = Kp * error

            angle_cmd = base_angle - output
            angle_cmd = max(45.0, min(135.0, angle_cmd))

            speed_cmd = SPEED_ULTRASONIC - speed_angle_diff * abs(output) + speed_straight_diff
            
            print('left:', left, '\t', 'right:', right, '\t',)
            # print('status_before', status_before, '\t', 'speed_straight_diff', speed_straight_diff, '\t', 'speed_cmd:', speed_cmd)

            if speed_cmd < 0.0:
                speed_cmd = 0.0

            set_servo_angle(angle_cmd)
            move_forward(speed_cmd)

            time.sleep(0.002)


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
    try:
        time.sleep(1)
        # 현재 프로세스에서 메인 제어 수행
        main_control()
    except KeyboardInterrupt:
        print("전체 시스템 KeyboardInterrupt")
    finally:
        print("전체 종료")