#!/usr/bin/env python3
# Performs line tracing judgment using OpenCV.
# Processes camera images to find lines and returns judgment results.
# Returns one of: "forward", "green", "left", "non", "red", "right".
# Judges based on the angle and center position of white lines.

import cv2
import numpy as np

# ==================== Image Processing Settings ====================
IMG_WIDTH = 320
IMG_HEIGHT = 240
ROI_TOP = 0.4
ROI_BOTTOM = 1.0

# Line detection settings
WHITE_THRESHOLD = 200
MIN_LINE_WIDTH = 2
MAX_LINE_WIDTH = 20

# Control variables (for state maintenance)
prev_center_error = 0
prev_line_angle = 0.0

# ==================== Image Processing Functions ====================
def preprocess_image(frame):
    """Preprocess image (including contrast enhancement)"""
    img = cv2.resize(frame, (IMG_WIDTH, IMG_HEIGHT))

    # Contrast enhancement (using CLAHE - Adaptive Histogram Equalization)
    # Convert to LAB color space and apply only to L channel (preserve color)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l_channel, a, b = cv2.split(lab)

    # Apply CLAHE (contrast enhancement)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)

    # Convert back to RGB
    img = cv2.merge([l_channel, a, b])
    img = cv2.cvtColor(img, cv2.COLOR_LAB2RGB)

    # Extract ROI
    h, w = img.shape[:2]
    roi_top = int(h * ROI_TOP)
    roi_bottom = int(h * ROI_BOTTOM)
    roi = img[roi_top:roi_bottom, :]
    return roi, roi_top

def detect_line_with_angle(roi):
    """Detect line and calculate angle"""
    global prev_line_angle

    gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, WHITE_THRESHOLD, 255, cv2.THRESH_BINARY)

    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    h, w = binary.shape

    # Find line center at bottom and top
    bottom_center = find_line_center(binary, int(h * 0.8))
    top_center = find_line_center(binary, int(h * 0.2))

    # Calculate line angle
    line_angle = 0.0
    if bottom_center is not None and top_center is not None:
        # Calculate angle between two points (radians)
        dy = h * 0.6  # 상단과 하단 사이의 거리
        dx = bottom_center - top_center
        line_angle = np.arctan2(dy, dx) * 180.0 / np.pi  # Convert to degrees
        # Normalize to -90 ~ 90 degree range
        if line_angle > 90:
            line_angle = line_angle - 180
        elif line_angle < -90:
            line_angle = line_angle + 180
    elif bottom_center is not None:
        # If only bottom exists, maintain previous angle or set to 0
        line_angle = prev_line_angle

    return binary, top_center, bottom_center, line_angle

def find_line_center(binary, y_pos):
    """Find center x coordinate of line at specific y position"""
    row = binary[y_pos, :]
    white_pixels = np.where(row > 128)[0]

    if len(white_pixels) == 0:
        return None

    center = int(np.mean(white_pixels))
    line_width = white_pixels[-1] - white_pixels[0]

    if line_width < MIN_LINE_WIDTH or line_width > MAX_LINE_WIDTH:
        return None

    return center

def update_state(bottom_center, line_angle, img_center):
    """Update state variables"""
    global prev_center_error, prev_line_angle

    if bottom_center is not None:
        center_error = bottom_center - img_center
        prev_center_error = center_error
        prev_line_angle = line_angle

def detect_traffic_light(frame):
    """Detect traffic light"""
    h, w = frame.shape[:2]
    roi = frame[0:int(h*0.3), :]

    hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)

    # Red color range
    red_lower1 = np.array([0, 50, 50])
    red_upper1 = np.array([10, 255, 255])
    red_lower2 = np.array([170, 50, 50])
    red_upper2 = np.array([180, 255, 255])
    red_mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
    red_mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)

    # Green color range
    green_lower = np.array([40, 50, 50])
    green_upper = np.array([80, 255, 255])
    green_mask = cv2.inRange(hsv, green_lower, green_upper)

    red_pixels = cv2.countNonZero(red_mask)
    green_pixels = cv2.countNonZero(green_mask)
    threshold = 100

    if red_pixels > threshold:
        return 'red'
    elif green_pixels > threshold:
        return 'green'
    else:
        return None

def judge_direction(bottom_center, line_angle, img_center):
    """Judge left, right, forward, non"""
    if bottom_center is None:
        return 'non'

    center_error = abs(bottom_center - img_center)
    threshold = IMG_WIDTH * 0.15  # 15% threshold

    if center_error < threshold and abs(line_angle) < 10:
        return 'forward'
    elif bottom_center < img_center - threshold:
        return 'left'
    elif bottom_center > img_center + threshold:
        return 'right'
    else:
        return 'forward'

# ==================== Judgment Function ====================
def judge_cv(frame_rgb):
    """
    Analyzes frame using OpenCV and returns judgment result.

    Args:
        frame_rgb: Image frame in RGB format (numpy array)

    Returns:
        str: One of "forward", "green", "left", "non", "red", "right"
    """
    global prev_center_error, prev_line_angle

    img_center = IMG_WIDTH / 2

    # Detect traffic light (priority)
    traffic_light = detect_traffic_light(frame_rgb)
    if traffic_light == 'red':
        return 'red'
    elif traffic_light == 'green':
        return 'green'

    # Preprocess image
    roi, roi_top = preprocess_image(frame_rgb)

    # Detect line (including angle)
    binary, top_center, bottom_center, line_angle = detect_line_with_angle(roi)

    # Judge direction
    direction = judge_direction(bottom_center, line_angle, img_center)

    # Update state
    update_state(bottom_center, line_angle, img_center)

    return direction

def init_cv():
    """Initialize CV module"""
    global prev_center_error, prev_line_angle
    prev_center_error = 0
    prev_line_angle = 0.0
