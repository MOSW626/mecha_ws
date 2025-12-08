#!/usr/bin/env python3
# main.py

import time
import sys
import RPi.GPIO as GPIO

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

    print("\n✅ PART 1 Complete. Switching to Ultrasonic Mode...")
    time.sleep(0.5)

    # PART 2: Ultrasonic Driving
    print("\n>>> STARTING PART 2: Low Defense Driving")
    try:
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
