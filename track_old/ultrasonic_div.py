import RPi.GPIO as GPIO
import time
import os
from multiprocessing import Process, Value, Lock
import ctypes

# GPIO pin locations
TRIG_LEFT  = 9 
ECHO_LEFT  = 11
TRIG_RIGHT = 5
ECHO_RIGHT = 6

MIN_CM, MAX_CM = 3.0, 150.0
ALPHA = 0.5

# ------------ 코어 affinity 설정 함수 ------------
def set_affinity(core_ids):
    """
    core_ids: {0}, {1}, {0,1} 이런 식의 set
    """
    os.sched_setaffinity(0, core_ids)
# -------------------------------------------------

# ------------ 초음파 관련 함수 ------------
def sample_distance(trig, echo):
    GPIO.output(trig, True)
    time.sleep(0.00001)
    GPIO.output(trig, False)

    t0 = time.time()
    while GPIO.input(echo) == 0:
        if time.time() - t0 > 0.02:
            return None
    start = time.time()

    while GPIO.input(echo) == 1:
        if time.time() - start > 0.02:
            return 8787
    end = time.time()

    dist = (end - start) * 34300 / 2.0
    dist = max(MIN_CM, min(dist, MAX_CM))
    return dist

def read_stable(trig, echo):
    return sample_distance(trig, echo)

def smooth(prev_value, new_value, alpha=ALPHA):
    if new_value == 8787:
        return 150
    if new_value is None:
        return prev_value
    if prev_value is None:
        return new_value
    return alpha * new_value + (1 - alpha) * prev_value
# -----------------------------------------------

# ------------ 센서 프로세스 (코어1 전담) ------------
def sensor_process(left_val, right_val, lock):
    # 코어 1번에 pinning (원하는 코어 번호로 바꿔도 됨)
    set_affinity({1})

    GPIO.setmode(GPIO.BCM)
    GPIO.setup([TRIG_LEFT, TRIG_RIGHT], GPIO.OUT)
    GPIO.setup([ECHO_LEFT, ECHO_RIGHT], GPIO.IN)

    last_left = None
    last_right = None

    try:
        loop_count = 0
        t_prev = time.perf_counter()
        sum_dt = 0.0
        min_dt = float('inf')
        max_dt = 0.0
        PRINT_INTERVAL = 50

        while True:
            t_now = time.perf_counter()
            dt = t_now - t_prev
            t_prev = t_now

            if loop_count > 0:
                sum_dt += dt
                if dt < min_dt: min_dt = dt
                if dt > max_dt: max_dt = dt
                if loop_count % PRINT_INTERVAL == 0:
                    avg_dt = sum_dt / loop_count
                    # print(f"[SENSOR loop {loop_count}] "
                    #       f"last={dt*1000:.3f} ms, "
                    #       f"avg={avg_dt*1000:.3f} ms, "
                    #       f"min={min_dt*1000:.3f} ms, "
                    #       f"max={max_dt*1000:.3f} ms")

            loop_count += 1

            raw_left  = read_stable(TRIG_LEFT, ECHO_LEFT)
            raw_right = read_stable(TRIG_RIGHT, ECHO_RIGHT)

            left  = smooth(last_left, raw_left)
            right = smooth(last_right, raw_right)
            last_left, last_right = left, right

            with lock:
                left_val.value  = left if left is not None else 0.0
                right_val.value = right if right is not None else 0.0

            time.sleep(0.002)

    except KeyboardInterrupt:
        print("센서 프로세스 종료")
    finally:
        GPIO.cleanup()
# ---------------------------------------------------

# ------------ 메인 프로세스 (코어0 전담) ------------
def main_control(left_val, right_val, lock):
    # 코어 0번에 pinning
    set_affinity({0})

    print("메인 제어 프로세스 시작")

    loop_count = 0
    t_prev = time.perf_counter()
    sum_dt = 0.0
    min_dt = float('inf')
    max_dt = 0.0
    PRINT_INTERVAL = 50

    try:
        while True:
            t_now = time.perf_counter()
            dt = t_now - t_prev
            t_prev = t_now

            if loop_count > 0:
                sum_dt += dt
                if dt < min_dt: min_dt = dt
                if dt > max_dt: max_dt = dt
                if loop_count % PRINT_INTERVAL == 0:
                    avg_dt = sum_dt / loop_count
                    with lock:
                        l = left_val.value
                        r = right_val.value
                    print(f"[MAIN loop {loop_count}] "
                          f"last={dt*1000:.3f} ms, "
                          f"avg={avg_dt*1000:.3f} ms, "
                          f"min={min_dt*1000:.3f} ms, "
                          f"max={max_dt*1000:.3f} ms | "
                          f"L={l:.1f}cm R={r:.1f}cm")

            loop_count += 1

            # 여기서 l, r을 읽어서 PID, 모터 제어 등 실제 제어 수행
            with lock:
                left  = left_val.value
                right = right_val.value

            # 예시: 그냥 값만 보고 있음
            # 실제로는 set_servo_angle, move_forward 등을 호출

            time.sleep(0.001)  # 제어 루프 주기 ~1ms

    except KeyboardInterrupt:
        print("메인 제어 종료")
# ---------------------------------------------------

if __name__ == "__main__":
    # 공유 메모리 준비
    left_val  = Value(ctypes.c_double, 0.0)
    right_val = Value(ctypes.c_double, 0.0)
    lock = Lock()

    # 센서 프로세스 시작
    p_sensor = Process(target=sensor_process, args=(left_val, right_val, lock))
    p_sensor.start()

    try:
        # 메인 제어 루틴 실행 (현재 프로세스에서)
        main_control(left_val, right_val, lock)
    except KeyboardInterrupt:
        pass
    finally:
        p_sensor.terminate()
        p_sensor.join()
        print("전체 종료")
