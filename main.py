#!/usr/bin/env python3
# main.py
# Orchestrator: Linetracing (Start -> Traffic Light) -> Low Defense (Ultrasonic Drive)

import time
import sys
import RPi.GPIO as GPIO

# 모듈 import
try:
    import linetracing
    import low_defense
except ImportError as e:
    print(f"❌ Module import error: {e}")
    print("Make sure linetracing.py and low_defense.py are in the same directory.")
    sys.exit(1)

def main():
    print("="*60)
    print("🚀 Auto Driving System Started")
    print("Sequence: [1] Line Tracing & Traffic Light -> [2] Ultrasonic Driving")
    print("="*60)

    # ----------------------------------------
    # PART 1: Line Tracing + Traffic Light
    # ----------------------------------------
    print("\n>>> STARTING PART 1: Line Tracing")

    try:
        # linetracing 모듈의 함수 실행
        # 이 함수는 Green 신호 후 1초 직진하고 종료됨
        success = linetracing.run_linetracing_sequence()

        if not success:
            print("❌ Part 1 interrupted or failed. Stopping.")
            return

    except Exception as e:
        print(f"❌ Error during Part 1: {e}")
        return

    print("\n✅ PART 1 Complete. Switching modes...")
    time.sleep(0.1) # 잠시 대기 (전류 안정화 및 기계적 관성 제거)

    # ----------------------------------------
    # PART 2: Ultrasonic Driving (Low Defense)
    # ----------------------------------------
    print("\n>>> STARTING PART 2: Low Defense Driving")

    try:
        # low_defense 모듈의 메인 함수 실행
        # GPIO cleanup이 Part 1에서 되었으므로 여기서 다시 init 함
        low_defense.main_control()

    except KeyboardInterrupt:
        print("\n🛑 System stopped by user.")
    except Exception as e:
        print(f"❌ Error during Part 2: {e}")
    finally:
        GPIO.cleanup()
        print("System Shutdown.")

if __name__ == "__main__":
    main()
