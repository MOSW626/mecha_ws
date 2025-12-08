import RPi.GPIO as GPIO
import time
import os

# ==================== 제어 파라미터 ====================
Kp = 2.55
Kd = 0.0
base_angle = 100.0
speed_angle_diff = 0.28
SPEED_ULTRASONIC = 90.0
MIN_CM, MAX_CM = 3.0, 150.0

# ==================== GPIO 핀 설정 ====================
DIR_PIN   = 16
PWM_PIN   = 12
SERVO_PIN = 13
MOTOR_FREQ = 1000
SERVO_FREQ = 50
SERVO_MAX_DUTY = 12
SERVO_MIN_DUTY = 3

TRIG_LEFT  = 17
ECHO_LEFT  = 27
TRIG_RIGHT = 5
ECHO_RIGHT = 6

# 전역 변수 (초기값 None)
motor_pwm = None
servo_pwm = None

# ★ 함수 밖에서는 GPIO 설정을 절대 하지 않습니다.

# ==================== 초음파 센서 함수 ====================
def sample_distance(trig, echo):
    GPIO.output(trig, True)
    time.sleep(0.00001)
    GPIO.output(trig, False)

    t0 = time.time()
    while GPIO.input(echo) == 0:
        if time.time() - t0 > 0.02:
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

    # ★ 모든 설정은 이 함수가 호출될 때 수행합니다.
    try:
        GPIO.setmode(GPIO.BCM)
    except Exception:
        pass # 이미 설정되어 있어도 무시

    GPIO.setup(DIR_PIN, GPIO.OUT)
    GPIO.setup(PWM_PIN, GPIO.OUT)
    GPIO.setup(SERVO_PIN, GPIO.OUT)

    # 초음파 핀 설정
    GPIO.setup([TRIG_LEFT, TRIG_RIGHT], GPIO.OUT)
    GPIO.setup([ECHO_LEFT, ECHO_RIGHT], GPIO.IN)

    # PWM 생성
    motor_pwm = GPIO.PWM(PWM_PIN, MOTOR_FREQ)
    servo_pwm = GPIO.PWM(SERVO_PIN, SERVO_FREQ)
    motor_pwm.start(0)
    servo_pwm.start(0)
    print(">>> Low Defense GPIO Initialized.")

def set_servo_angle(degree):
    global servo_pwm
    if servo_pwm is None: return
    degree = max(45.0, min(135.0, degree))
    duty = SERVO_MIN_DUTY + (degree * (SERVO_MAX_DUTY - SERVO_MIN_DUTY) / 180.0)
    servo_pwm.ChangeDutyCycle(duty)

def move_forward(speed):
    global motor_pwm
    if motor_pwm is None: return
    GPIO.output(DIR_PIN, GPIO.HIGH)
    motor_pwm.ChangeDutyCycle(max(0.0, min(100.0, speed)))

def stop_motor():
    global motor_pwm
    if motor_pwm is None: return
    motor_pwm.ChangeDutyCycle(0)

def main_control():
    global motor_pwm, servo_pwm

    # ★ 여기서 초기화를 시작합니다
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

    # 첫 센싱
    left_prev  = sample_distance(TRIG_LEFT, ECHO_LEFT)
    right_prev = sample_distance(TRIG_RIGHT, ECHO_RIGHT)

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
                            # print('corner detected')
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
                                # print('straight path detected')
                                track_change_lock = time.time()

            if straight_continous_time is not None:
                if time.time() - straight_continous_time > 0.9:
                    speed_straight_diff -= 0.6
                    # print('speed down')
                    if speed_straight_diff < -20.0:
                        speed_straight_diff = -20.0

            error = ref_distance_difference - (right - left)
            output = Kp * error

            angle_cmd = base_angle - output
            angle_cmd = max(45.0, min(135.0, angle_cmd))

            speed_cmd = SPEED_ULTRASONIC - speed_angle_diff * abs(output) + speed_straight_diff

            print(f'L: {left:.1f} | R: {right:.1f} | Speed: {speed_cmd:.1f}')

            if speed_cmd < 0.0:
                speed_cmd = 0.0

            set_servo_angle(angle_cmd)
            move_forward(speed_cmd)

            time.sleep(0.002)

    except KeyboardInterrupt:
        print("메인 제어 종료 (KeyboardInterrupt)")
    finally:
        try:
            stop_motor()
            set_servo_angle(base_angle)
            if motor_pwm is not None:
                motor_pwm.stop()
            if servo_pwm is not None:
                servo_pwm.stop()
        except Exception as e:
            print(f"PWM stop 중 오류: {e}")

        # main.py에서 cleanup 할 것이므로 여기서는 생략하거나 안전하게 처리
        pass

if __name__ == "__main__":
    try:
        main_control()
    except KeyboardInterrupt:
        print("시스템 종료")
    finally:
        GPIO.cleanup()
