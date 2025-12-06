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
ROI_TOP = 0.5  # Wider ROI to reduce non detection (was 0.4)
ROI_BOTTOM = 1.0

# Line detection settings
WHITE_THRESHOLD = 160  # Lower threshold to detect more lines (was 200)
MIN_LINE_WIDTH = 5  # Lower minimum to catch thinner lines (was 2)
MAX_LINE_WIDTH = 200  # Higher maximum to catch wider lines (was 20)

# Control variables (for state maintenance)
prev_center_error = 0
prev_line_angle = 0.0

# PID control settings
Kp = 0.8  # Proportional gain
Ki = 0.0  # Integral gain
Kd = 0.1  # Derivative gain

# PID state variables
prev_error = 0
integral = 0

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

    # Use larger kernel for better line connection
    kernel = np.ones((5, 5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    # Remove small noise
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    h, w = binary.shape

    # Find line center at bottom and top (try multiple positions for robustness)
    bottom_center = find_line_center(binary, int(h * 0.8))
    if bottom_center is None:
        # Try slightly different positions
        bottom_center = find_line_center(binary, int(h * 0.7))
    if bottom_center is None:
        bottom_center = find_line_center(binary, int(h * 0.9))

    top_center = find_line_center(binary, int(h * 0.2))
    if top_center is None:
        # Try slightly different positions
        top_center = find_line_center(binary, int(h * 0.3))
    if top_center is None:
        top_center = find_line_center(binary, int(h * 0.1))

    # Calculate line angle
    line_angle = 0.0
    if bottom_center is not None and top_center is not None:
        # Calculate angle between two points (radians)
        dy = h * 0.6  # Distance between top and bottom
        dx = bottom_center - top_center
        line_angle = np.arctan2(dx, dy) * 180.0 / np.pi  # Convert to degrees
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
    # Use multiple rows around y_pos for more robust detection
    h, w = binary.shape
    y_start = max(0, y_pos - 2)
    y_end = min(h, y_pos + 3)

    # Combine multiple rows
    region = binary[y_start:y_end, :]
    white_pixels = np.where(region > 128)[1]  # Get column indices

    if len(white_pixels) == 0:
        return None

    # Sort pixels to find continuous regions
    white_pixels = np.sort(white_pixels)

    # Find the largest continuous white region (line)
    # Group consecutive pixels
    if len(white_pixels) == 1:
        return int(white_pixels[0])

    # Find gaps larger than 5 pixels (likely separate regions)
    gaps = np.diff(white_pixels)
    gap_indices = np.where(gaps > 5)[0]

    if len(gap_indices) == 0:
        # Single continuous region
        line_start = white_pixels[0]
        line_end = white_pixels[-1]
        line_width = line_end - line_start
        center = int((line_start + line_end) / 2)
    else:
        # Multiple regions - find the largest one
        regions = []
        start_idx = 0
        for gap_idx in gap_indices:
            regions.append((start_idx, gap_idx))
            start_idx = gap_idx + 1
        regions.append((start_idx, len(white_pixels) - 1))

        # Find largest region
        largest_region = None
        largest_width = 0
        for start_idx, end_idx in regions:
            region_start = white_pixels[start_idx]
            region_end = white_pixels[end_idx]
            region_width = region_end - region_start
            if region_width > largest_width:
                largest_width = region_width
                largest_region = (region_start, region_end)

        if largest_region is None:
            return None

        line_start, line_end = largest_region
        line_width = line_end - line_start
        center = int((line_start + line_end) / 2)

    # --- Debugging print (Optional: Remove after fixing) ---
    # print(f"Detected Width: {line_width}, Center: {center}")
    # -----------------------------------------------------

    # Check if line width is within acceptable range
    if line_width < MIN_LINE_WIDTH or line_width > MAX_LINE_WIDTH:
        return None

    return center

def calculate_control_output(bottom_center, line_angle, img_center):
    """
    Calculate servo angle and center error using PID control.

    Args:
        bottom_center: Line center at bottom of ROI (pixels)
        line_angle: Line angle in degrees
        img_center: Image center x coordinate (pixels)

    Returns:
        tuple: (servo_angle, center_error) or (None, None) if line not found
    """
    global prev_error, integral, prev_center_error, prev_line_angle

    if bottom_center is None:
        return None, None

    # Calculate center error
    center_error = bottom_center - img_center

    # Consider line angle for better control
    # Combine center error with angle-based correction
    angle_factor = line_angle / 90.0  # Normalize angle to -1 to 1
    combined_error = center_error + (angle_factor * IMG_WIDTH * 0.1)

    # PID control calculation
    integral += combined_error
    integral = max(-100, min(100, integral))  # Limit integral
    derivative = combined_error - prev_error

    # PID output
    output = Kp * combined_error + Ki * integral + Kd * derivative

    # Convert error to servo angle
    # Error range: -img_center ~ +img_center
    # Angle range: -45 degrees ~ +45 degrees
    max_error = IMG_WIDTH / 2
    angle_offset = (output / max_error) * 45  # Max 45 degree rotation

    # Calculate servo angle (90 is center)
    SERVO_ANGLE_CENTER = 90
    SERVO_ANGLE_MIN = 45
    SERVO_ANGLE_MAX = 135

    angle = SERVO_ANGLE_CENTER - angle_offset
    angle = max(SERVO_ANGLE_MIN, min(SERVO_ANGLE_MAX, angle))

    # Update state variables
    prev_error = combined_error
    prev_center_error = center_error
    prev_line_angle = line_angle

    return angle, center_error

def update_state(bottom_center, line_angle, img_center):
    """Update state variables"""
    global prev_center_error, prev_line_angle

    if bottom_center is not None:
        center_error = bottom_center - img_center
        prev_center_error = center_error
        prev_line_angle = line_angle

def detect_traffic_light(frame):
    """Detect traffic light with improved logic for real-world LED lights"""
    h, w = frame.shape[:2]
    # Use smaller ROI for traffic light (top 25% only) to avoid interfering with line detection
    roi = frame[0:int(h*0.25), :]

    hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)

    # For LED traffic lights, use more restrictive HSV ranges to avoid false positives
    # Red LED range (more restrictive to avoid false detection)
    red_lower1 = np.array([0, 50, 100])  # Higher saturation to avoid false positives
    red_upper1 = np.array([10, 255, 255])
    red_lower2 = np.array([170, 50, 100])
    red_upper2 = np.array([180, 255, 255])
    red_mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
    red_mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)

    # Green LED range (more restrictive)
    green_lower = np.array([40, 40, 100])  # Higher saturation, narrower range
    green_upper = np.array([90, 255, 255])  # Narrower range

    green_mask = cv2.inRange(hsv, green_lower, green_upper)

    # Apply morphological operations to reduce noise
    kernel = np.ones((5, 5), np.uint8)  # Larger kernel for better noise reduction
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)
    # Remove small noise
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    red_pixels = cv2.countNonZero(red_mask)
    green_pixels = cv2.countNonZero(green_mask)

    # Higher threshold to avoid false positives from line or other objects
    threshold = 300  # Increased from 150 to reduce false positives

    # Calculate ratio to determine which is stronger
    total_pixels = red_pixels + green_pixels
    if total_pixels < threshold:
        return None

    # Red has priority - if red is detected and is significant, return red
    if red_pixels > threshold:
        # If red is significantly stronger than green, return red
        if red_pixels > green_pixels * 1.5:  # Higher ratio for more certainty
            return 'red'
        # If red is detected but green is also strong, prefer red (safety first)
        elif red_pixels >= green_pixels:
            return 'red'
        # If green is much stronger than red, it might be green
        elif green_pixels > red_pixels * 3.0:  # Higher ratio for more certainty
            return 'green'
        else:
            # Ambiguous case - prefer red for safety
            return 'red'
    elif green_pixels > threshold:
        # Only return green if red is not detected or very weak
        if red_pixels < threshold * 0.3:  # Lower threshold for red interference
            return 'green'
        else:
            # Red is also present, prefer red
            return 'red'
    else:
        return None

def judge_direction(bottom_center, line_angle, img_center):
    """Judge left, right, forward, non"""
    if bottom_center is None:
        return 'non'

    center_error = abs(bottom_center - img_center)
    threshold = IMG_WIDTH * 0.15  # 15% threshold

    # line_angle이 None이거나 계산되지 않은 경우 처리
    if line_angle is None:
        line_angle = 0.0

    # 중심 오차와 각도를 고려한 판단
    # 중심이 가까우면서 각도가 작으면 forward
    if center_error < threshold and abs(line_angle) < 10:  # 각도 임계값을 10도에서 15도로 완화
        return 'forward'
    # 중심이 왼쪽에 있으면 left
    elif bottom_center < img_center - threshold:
        return 'left'
    # 중심이 오른쪽에 있으면 right
    elif bottom_center > img_center + threshold:
        return 'right'
    # 그 외의 경우 (중심 오차가 threshold 내이지만 각도가 큰 경우)
    # 각도 방향에 따라 판단
    elif center_error < threshold:
        # 각도가 크면 각도 방향으로 판단
        if line_angle < -5:
            return 'left'
        elif line_angle > 5:
            return 'right'
        else:
            return 'forward'
    # 중심 오차가 threshold보다 크지만 left/right 조건을 만족하지 않는 경우
    # 각도 정보를 활용
    else:
        if line_angle < -10:
            return 'left'
        elif line_angle > 10:
            return 'right'
        else:
            # 각도 정보가 없거나 작으면 중심 위치로 판단
            if bottom_center < img_center:
                return 'left'
            else:
                return 'right'

# ==================== Judgment Function ====================
def judge_cv(frame_rgb, return_debug=False):
    """
    Analyzes frame using OpenCV and returns judgment result.

    Args:
        frame_rgb: Image frame in RGB format (numpy array)
        return_debug: If True, returns tuple (direction, debug_info)

    Returns:
        str or tuple: One of "forward", "green", "left", "non", "red", "right"
                      If return_debug=True, returns (direction, debug_info dict)
    """
    global prev_center_error, prev_line_angle

    img_center = IMG_WIDTH / 2
    debug_info = {}

    # Detect traffic light (priority)
    traffic_light = detect_traffic_light(frame_rgb)
    if traffic_light == 'red':
        if return_debug:
            debug_info['traffic_light'] = 'red'
            debug_info['binary'] = None
            debug_info['roi'] = None
            debug_info['bottom_center'] = None
            debug_info['top_center'] = None
            return 'red', debug_info
        return 'red'
    elif traffic_light == 'green':
        if return_debug:
            debug_info['traffic_light'] = 'green'
            debug_info['binary'] = None
            debug_info['roi'] = None
            debug_info['bottom_center'] = None
            debug_info['top_center'] = None
            return 'green', debug_info
        return 'green'

    # Preprocess image
    roi, roi_top = preprocess_image(frame_rgb)

    # Detect line (including angle)
    binary, top_center, bottom_center, line_angle = detect_line_with_angle(roi)

    # Judge direction
    direction = judge_direction(bottom_center, line_angle, img_center)

    # Update state
    update_state(bottom_center, line_angle, img_center)

    if return_debug:
        debug_info['traffic_light'] = None
        debug_info['binary'] = binary
        debug_info['roi'] = roi
        debug_info['bottom_center'] = bottom_center
        debug_info['top_center'] = top_center
        debug_info['line_angle'] = line_angle
        debug_info['roi_top'] = roi_top
        return direction, debug_info

    return direction

def init_cv():
    """Initialize CV module"""
    global prev_center_error, prev_line_angle, prev_error, integral
    prev_center_error = 0
    prev_line_angle = 0.0
    prev_error = 0
    integral = 0
