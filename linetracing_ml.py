#!/usr/bin/env python3
# Performs line tracing judgment using cnn.tflite file.
# Processes camera images to find lines and returns judgment results.
# Returns one of: "forward", "green", "left", "non", "red", "right".

import cv2
import numpy as np
import os
from collections import Counter

# Keras 모델 지원
try:
    import tensorflow as tf
    from tensorflow import keras
    USE_KERAS = True
except ImportError:
    USE_KERAS = False
    print("⚠ TensorFlow/Keras not available.")

# TFLite 모델 지원 (Keras가 없을 때 사용)
try:
    import tflite_runtime.interpreter as tflite
    USE_TFLITE = True
except ImportError:
    USE_TFLITE = False
    print("⚠ TFLite not available.")

# ==================== ML Model Settings ====================
# Use model from cnn folder (Keras first, TFLite if not available)
MODEL_DIR = "./cnn"
KERAS_MODEL_PATH = os.path.join(MODEL_DIR, "cnn_model2.keras")
H5_MODEL_PATH = os.path.join(MODEL_DIR, "cnn_model2.h5")
TFLITE_MODEL_PATH = os.path.join(MODEL_DIR, "cnn_model2.tflite")

# Class order of trained model (same as train_model.py)
LABELS = ["forward", "green", "left", "non", "red", "right"]
IMG_SIZE = 256  # Image size used during training

# ML related variables
model = None  # Keras model
interpreter = None  # TFLite interpreter
inp = None
out = None
use_keras = False  # True for Keras, False for TFLite

# Buffer for prediction stabilization
prediction_buffer = []
buffer_size = 3

# ==================== ML Image Processing Functions ====================
def preprocess_frame(frame_rgb):
    """Preprocess image for ML"""
    img = cv2.resize(frame_rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    return img[None, ...]

def predict_ml(frame_rgb):
    """Predict using ML model"""
    global model, interpreter, inp, out, use_keras

    if model is None and interpreter is None:
        return None, 0.0

    try:
        x = preprocess_frame(frame_rgb)

        if use_keras and model is not None:
            # Use Keras model
            probs = model.predict(x, verbose=0)[0]
        elif interpreter is not None:
            # Use TFLite model
            interpreter.set_tensor(inp["index"], x)
            interpreter.invoke()
            probs = interpreter.get_tensor(out["index"])[0]
        else:
            return None, 0.0

        pred_id = int(np.argmax(probs))
        pred_label = LABELS[pred_id]
        confidence = float(probs[pred_id])
        return pred_label, confidence
    except Exception as e:
        print(f"⚠ ML prediction error: {e}")
        return None, 0.0

# ==================== Judgment Function ====================
def judge_ml(frame_rgb):
    """
    Analyzes frame using ML model and returns judgment result.

    Args:
        frame_rgb: Image frame in RGB format (numpy array)

    Returns:
        str: One of "forward", "green", "left", "non", "red", "right"
    """
    global prediction_buffer

    if model is None and interpreter is None:
        return None

    # ML prediction
    pred_label, confidence = predict_ml(frame_rgb)

    if pred_label is None:
        return None

    # Add to prediction buffer
    prediction_buffer.append(pred_label)
    if len(prediction_buffer) > buffer_size:
        prediction_buffer.pop(0)

    # Select most common prediction from buffer
    if len(prediction_buffer) > 0:
        most_common = Counter(prediction_buffer).most_common(1)[0][0]
        return most_common

    return pred_label

def init_ml():
    """Initialize and load ML model"""
    global model, interpreter, inp, out, use_keras, prediction_buffer

    prediction_buffer = []

    # Load ML model (Keras first, TFLite if not available)
    model_loaded = False

    # Check absolute paths as well
    abs_keras_path = os.path.abspath(KERAS_MODEL_PATH)
    abs_h5_path = os.path.abspath(H5_MODEL_PATH)
    abs_tflite_path = os.path.abspath(TFLITE_MODEL_PATH)

    print(f"Looking for ML models...")
    print(f"  Current directory: {os.getcwd()}")
    print(f"  Model directory: {os.path.abspath(MODEL_DIR)}")

    # 1. Try Keras model (.keras first, .h5 if not available)
    if USE_KERAS:
        if os.path.exists(KERAS_MODEL_PATH):
            print(f"Attempting to load Keras model: {KERAS_MODEL_PATH}")
            try:
                model = keras.models.load_model(KERAS_MODEL_PATH)
                use_keras = True
                model_loaded = True
                print("✓ Keras model loaded (.keras)")
            except Exception as e:
                print(f"⚠ Failed to load .keras model: {e}")
        elif os.path.exists(abs_keras_path):
            print(f"Attempting to load Keras model (abs): {abs_keras_path}")
            try:
                model = keras.models.load_model(abs_keras_path)
                use_keras = True
                model_loaded = True
                print("✓ Keras model loaded (.keras)")
            except Exception as e:
                print(f"⚠ Failed to load .keras model: {e}")

        if not model_loaded and os.path.exists(H5_MODEL_PATH):
            print(f"Attempting to load Keras model: {H5_MODEL_PATH}")
            try:
                model = keras.models.load_model(H5_MODEL_PATH)
                use_keras = True
                model_loaded = True
                print("✓ Keras model loaded (.h5)")
            except Exception as e:
                print(f"⚠ Failed to load .h5 model: {e}")
        elif not model_loaded and os.path.exists(abs_h5_path):
            print(f"Attempting to load Keras model (abs): {abs_h5_path}")
            try:
                model = keras.models.load_model(abs_h5_path)
                use_keras = True
                model_loaded = True
                print("✓ Keras model loaded (.h5)")
            except Exception as e:
                print(f"⚠ Failed to load .h5 model: {e}")

    # 2. Try TFLite model (when Keras is not available)
    if not model_loaded and USE_TFLITE:
        if os.path.exists(TFLITE_MODEL_PATH):
            print(f"Attempting to load TFLite model: {TFLITE_MODEL_PATH}")
            try:
                interpreter = tflite.Interpreter(model_path=TFLITE_MODEL_PATH)
                interpreter.allocate_tensors()
                inp = interpreter.get_input_details()[0]
                out = interpreter.get_output_details()[0]
                use_keras = False
                model_loaded = True
                print("✓ TFLite model loaded")
            except Exception as e:
                print(f"⚠ Failed to load TFLite model: {e}")
        elif os.path.exists(abs_tflite_path):
            print(f"Attempting to load TFLite model (abs): {abs_tflite_path}")
            try:
                interpreter = tflite.Interpreter(model_path=abs_tflite_path)
                interpreter.allocate_tensors()
                inp = interpreter.get_input_details()[0]
                out = interpreter.get_output_details()[0]
                use_keras = False
                model_loaded = True
                print("✓ TFLite model loaded")
            except Exception as e:
                print(f"⚠ Failed to load TFLite model: {e}")

    if not model_loaded:
        print("✗ No available model file found.")
        print(f"  Attempted paths:")
        if USE_KERAS:
            print(f"    - {KERAS_MODEL_PATH} (exists: {os.path.exists(KERAS_MODEL_PATH)})")
            print(f"    - {abs_keras_path} (exists: {os.path.exists(abs_keras_path)})")
            print(f"    - {H5_MODEL_PATH} (exists: {os.path.exists(H5_MODEL_PATH)})")
            print(f"    - {abs_h5_path} (exists: {os.path.exists(abs_h5_path)})")
        if USE_TFLITE:
            print(f"    - {TFLITE_MODEL_PATH} (exists: {os.path.exists(TFLITE_MODEL_PATH)})")
            print(f"    - {abs_tflite_path} (exists: {os.path.exists(abs_tflite_path)})")
        return False

    print(f"Model used: {'Keras' if use_keras else 'TFLite'}")
    print(f"Image size: {IMG_SIZE}x{IMG_SIZE}")
    print(f"Classes: {LABELS}")
    return True
