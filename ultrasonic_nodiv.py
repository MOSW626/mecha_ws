import RPi.GPIO as GPIO
import time
import sys
import tty
import termios
import select


# GPIO pin locations
TRIG_LEFT = 17
ECHO_LEFT = 27
TRIG_RIGHT = 5
ECHO_RIGHT = 6

# Distance Clipping values
MIN_CM, MAX_CM = 3.0, 150.0
ALPHA = 0.85

# ==================== GPIO 초기화 ====================
GPIO.setmode(GPIO.BCM)
GPIO.setup([TRIG_LEFT, TRIG_RIGHT], GPIO.OUT)
GPIO.setup([ECHO_LEFT, ECHO_RIGHT], GPIO.IN)

# ==================== 초음파 센서 함수 ====================
def sample_distance(trig, echo):
    GPIO.output(trig, True)
    time.sleep(0.00001)  # 10us로 단축 (더 빠른 샘플링)
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
    val = sample_distance(trig, echo)
    # time.sleep(0.0005)  # 샘플링 간격 단축
    return val

def smooth(prev_value, new_value, alpha=ALPHA):
    if new_value == 8787:
        return 150
    if new_value is None:
        return prev_value
    if prev_value is None:
        return new_value
    return alpha * new_value + (1 - alpha) * prev_value

# ==================== 초음파 센서 모드 (루프 주기 측정) ====================
def ultrasonic_nodiv():
    last_left = None
    last_right = None

    # ---- 루프 주기 측정용 변수 ----
    loop_count = 0
    t_prev = time.perf_counter()
    sum_dt = 0.0
    min_dt = float('inf')
    max_dt = 0.0
    PRINT_INTERVAL = 10  # 몇 번마다 한 번씩 출력할지
    # --------------------------------

    try:
        while True:
            # 루프 시작 시각
            t_now = time.perf_counter()
            dt = t_now - t_prev
            t_prev = t_now

            # 첫 루프는 기준만 잡고 통계에서는 제외
            if loop_count > 0:
                sum_dt += dt
                if dt < min_dt:
                    min_dt = dt
                if dt > max_dt:
                    max_dt = dt

                if loop_count % PRINT_INTERVAL == 0:
                    avg_dt = sum_dt / loop_count
                    print(
                        f"[loop {loop_count}] "
                        f"last = {dt*1000:.3f} ms, "
                        f"avg = {avg_dt*1000:.3f} ms, "
                        f"min = {min_dt*1000:.3f} ms, "
                        f"max = {max_dt*1000:.3f} ms"
                    )

            loop_count += 1
            # ================= 루프 본연의 작업 =================
            # 초음파 센서 읽기
            raw_left = read_stable(TRIG_LEFT, ECHO_LEFT)
            raw_right = read_stable(TRIG_RIGHT, ECHO_RIGHT)

            # 스무딩
            left = smooth(last_left, raw_left)
            right = smooth(last_right, raw_right)
            last_left, last_right = left, right

            if left is None or right is None:
                continue
            # ====================================================

    except KeyboardInterrupt:
        print("초음파 모드 중단")
    except Exception as e:
        print(f"초음파 모드 오류: {e}")

# ==================== 메인 함수 ====================
def main():
    print("초음파 센서 분리 전 (루프 주기 측정 모드)")

    try:
        ultrasonic_nodiv()
    except KeyboardInterrupt:
        print("\n사용자에 의해 중단됨")
    except Exception as e:
        print(f"오류 발생: {e}")
        import traceback
        traceback.print_exc()
    finally:
        GPIO.cleanup()
        print("시스템 종료")

if __name__ == "__main__":
    main()
