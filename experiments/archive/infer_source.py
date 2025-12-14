#!/home/mecha/venvs/tflite/bin/python
# infer_source.py

import numpy as np, cv2, time
import tflite_runtime.interpreter as tflite
from picamera2 import Picamera2

IMG = 240
model = "./model.tflite"
labels = ["middle", "green", "left", "right", "red", "noline"]

interpreter = tflite.Interpreter(model_path=model)
interpreter.allocate_tensors()
inp = interpreter.get_input_details()[0]
out = interpreter.get_output_details()[0]

def preprocess_frame_for_model(frame):
    """
    frame: numpy array in RGB (H,W,3) or other channel counts from Picamera2.capture_array()
    Returns: array shaped like interpreter input (including batch dim) with the correct dtype.
    """
    # expected shape and dtype from interpreter
    f = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # expected shape and dtype from interpreter
    expected_shape = inp["shape"]
    expected_c = int(expected_shape[-1])
    expected_dtype = np.dtype(inp["dtype"])

    # normalize channels
    if f.ndim == 2:
        if expected_c == 1:
            f = f[..., None]
        else:
            f = cv2.cvtColor(f, cv2.COLOR_GRAY2RGB)
    elif f.ndim == 3:
        c = f.shape[2]
        if c == 4 and expected_c == 3:
            f = cv2.cvtColor(f, cv2.COLOR_RGBA2RGB)
        elif c == 3 and expected_c == 1:
            f = cv2.cvtColor(f, cv2.COLOR_RGB2GRAY)[..., None]
        elif c != expected_c:
            if c > expected_c:
                f = f[..., :expected_c]
            else:
                f = np.repeat(f[..., :1], expected_c, axis=2)

    # resize to model spatial size
    f = cv2.resize(f, (expected_shape[2], expected_shape[1]), interpolation=cv2.INTER_AREA)

    # convert to correct dtype and scaling
    if expected_dtype == np.uint8:
        out = (f).astype(np.uint8)
    else:
        out = (f.astype(np.float32) / 255.0).astype(np.float32)

    # add batch dim
    return out[None, ...]

def map_to_order(pred_label):
    return pred_label

def main():
    picam2 = Picamera2()
    # request an easy preview size; libcamera may pick a closest supported format
    # picam2.configure(picam2.create_preview_configuration({"main":{"size":(640,480)}}))
    config = picam2.create_preview_configuration(main={"size": (640, 480), "format": "RGB888"})
    picam2.configure(config)
    picam2.start()

    try:
        while True:
            # capture_array returns RGB image
            frame_rgb = picam2.capture_array()

            # optional debug: uncomment to log shape once
            # print("capture shape:", frame_rgb.shape)

            # prepare input for model
            x = preprocess_frame_for_model(frame_rgb)

            # set tensor and run
            interpreter.set_tensor(inp["index"], x)
            t0 = time.time()
            interpreter.invoke()
            dt = (time.time() - t0) * 1e3

            probs = interpreter.get_tensor(out["index"])[0]
            pred_id = int(np.argmax(probs))
            pred_label = labels[pred_id]
            order = map_to_order(pred_label)

            print(f"{order}  (model label={pred_label})  probs={probs}  ({dt:.1f} ms)")

            # show live camera (convert RGB -> BGR for OpenCV)
            cam_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            # overlay the order text on the camera view
            txt = f"Order: {order}"
            cv2.putText(cam_bgr, txt, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2, cv2.LINE_AA)

            cv2.imshow("camera", cam_bgr)

            # also show the actual model input we sent (for debugging)
            # convert x back to uint8 BGR for display
            disp = x[0]
            if disp.dtype != np.uint8:
                disp = (disp * 255).astype(np.uint8)
            # disp is in RGB order if channels==3
            if disp.shape[2] == 3:
                disp_bgr = cv2.cvtColor(disp, cv2.COLOR_RGB2BGR)
            elif disp.shape[2] == 1:
                disp_bgr = cv2.cvtColor(disp[...,0], cv2.COLOR_GRAY2BGR)
            else:
                disp_bgr = disp[..., :3]
            cv2.imshow("model_input", disp_bgr)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            time.sleep(0.03)

    finally:
        picam2.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
