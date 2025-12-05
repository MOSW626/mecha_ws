#!/usr/bin/env python3
# Combines judgments from linetracing_ml.py and linetracing_cv.py to make final judgment.
# Combines CV and ML judgment results using weights.

# ==================== Judgment Criteria Settings ====================
CV_WEIGHT = 0.8  # CV 80%
ML_WEIGHT = 0.2  # ML 20%

# Possible judgment results
VALID_LABELS = ["forward", "green", "left", "non", "red", "right"]

def combine_judgments(cv_result, ml_result, cv_weight=None, ml_weight=None):
    """
    Combines CV and ML judgment results to return final judgment.

    Args:
        cv_result: CV judgment result (one of "forward", "green", "left", "non", "red", "right")
        ml_result: ML judgment result (one of "forward", "green", "left", "non", "red", "right")
        cv_weight: CV weight (default: CV_WEIGHT)
        ml_weight: ML weight (default: ML_WEIGHT)

    Returns:
        str: Final judgment result (one of "forward", "green", "left", "non", "red", "right")
    """
    if cv_weight is None:
        cv_weight = CV_WEIGHT
    if ml_weight is None:
        ml_weight = ML_WEIGHT

    # Traffic lights have high priority (red > green > others)
    if cv_result == "red" or ml_result == "red":
        return "red"
    if cv_result == "green" or ml_result == "green":
        return "green"

    # If both are None, return non
    if cv_result is None and ml_result is None:
        return "non"

    # If only one is None, return the other
    if cv_result is None:
        return ml_result
    if ml_result is None:
        return cv_result

    # If both have same result, return as is
    if cv_result == ml_result:
        return cv_result

    # Weight-based selection
    # When CV and ML differ, select the one with higher weight
    if cv_weight >= ml_weight:
        return cv_result
    else:
        return ml_result

