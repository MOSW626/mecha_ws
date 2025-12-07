#!/usr/bin/env python3
# linetracing_Judgment.py

def combine_judgments(cv_result, ml_result):
    """
    ML이 Red/Green을 감지하면 그것을 따르고,
    그렇지 않으면(주행 관련) 무조건 CV 결과를 따릅니다.
    """

    # 1. ML이 명확하게 'red'나 'green'이라고 하면 최우선 반영
    if ml_result == "red":
        return "red"
    if ml_result == "green":
        return "green"

    # 2. 그 외의 경우 (ML이 left, right, middle, noline 이라 하더라도)
    # 정밀한 주행은 CV가 더 잘하므로 CV 결과를 신뢰함
    if cv_result is not None:
        return cv_result

    # 3. CV도 놓쳤는데 ML이 방향을 가리키는 경우 (예비책)
    # 필요 없다면 삭제하고 'non' 리턴해도 됨
    if ml_result in ["left", "right"]:
        return ml_result

    if ml_result == "middle":
        return "forward"

    return "non"
