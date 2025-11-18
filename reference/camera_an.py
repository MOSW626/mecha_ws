from picamera2 import Picamera2
import cv2

picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration())
picam2.start()
try:
    while True:
        frame = cv2.cvtColor(picam2.capture_array(), cv2.COLOR_RGB2BGR)
        cv2.imshow("Pi Camera (OpenCV)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    picam2.stop()
    cv2.destroyAllWindows()
