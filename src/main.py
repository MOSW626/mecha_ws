#!/usr/bin/env python3
# main.py

import time
import sys
import RPi.GPIO as GPIO
import gc  # ★ [추가] 가비지 컬렉터 모듈

# 모듈 import
try:
    import linetracing
    import low_defense
except ImportError as e:
    print(f"❌ Module import error: {e}")
    sys.exit(1)

def main():
    print("="*60)
    print("🚀 Auto Driving System Started")
    print("="*60)

    # PART 1: Line Tracing
    print("\n>>> STARTING PART 1: Line Tracing")
    try:
        success = linetracing.run_linetracing_sequence()
        if not success:
            print("❌ Part 1 Failed. Stopping.")
            return
    except Exception as e:
        print(f"❌ Error during Part 1: {e}")
        return

    print("\n✅ PART 1 Complete. Cleaning up memory...")

    # ★ [핵심 수정] 메모리에 남은 PWM 객체 강제 삭제 (Zombie Process 제거)
    try:
        GPIO.cleanup() # 1차 하드웨어 정리

        # linetracing 모듈 내부의 참조를 끊어줍니다 (선택사항이나 안전을 위해)
        if 'linetracing_drive' in sys.modules:
            sys.modules['linetracing_drive'].motor_pwm = None
            sys.modules['linetracing_drive'].servo_pwm = None

        gc.collect()   # 2차 메모리 정리 (여기서 __del__ 에러가 해소됨)
        print("✓ Memory Cleaned (GC Collected).")
    except Exception as e:
        print(f"Warning during cleanup: {e}")

    time.sleep(0) # 안정화 대기

    # PART 2: Ultrasonic Driving
    print("\n>>> STARTING PART 2: Low Defense Driving")
    try:
        # 안전하게 실행
        low_defense.main_control()
    except KeyboardInterrupt:
        print("\n🛑 System stopped by user.")
    except Exception as e:
        print(f"❌ Error during Part 2: {e}")
    finally:
        try:
            GPIO.cleanup()
        except:
            pass
        print("System Shutdown.")

if __name__ == "__main__":
    main()
