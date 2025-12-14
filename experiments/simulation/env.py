#!/usr/bin/env python3
"""
Custom Gymnasium Environment for Autonomous Racing Car Training
랜덤 트랙 생성, 운동학적 자전거 모델, 초음파 센서 시뮬레이션을 포함한 강화학습 환경
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Polygon
from scipy.interpolate import splprep, splev
import math


class RandomTrackEnv(gym.Env):
    """
    랜덤 트랙 환경: 매 에피소드마다 새로운 폐루프 트랙을 생성
    Sim-to-Real을 위한 도메인 랜덤화 포함
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        num_ultrasonic_sensors=2,
        sensor_angles=[-15, 75],  # 센서 장착 각도 (도)
        sensor_max_range=150.0,  # 센서 최대 감지 거리 (cm)
        sensor_noise_std_dev=2.0,  # 센서 노이즈 표준편차 (Sim-to-Real)
        track_width_min=40.0,  # 최소 트랙 폭 (cm)
        track_width_max=47.0,  # 최대 트랙 폭 (cm)
        track_length_min=1000.0,  # 최소 트랙 길이 (cm)
        track_length_max=4000.0,  # 최대 트랙 길이 (cm)
        car_length=25.0,  # 차량 길이 (cm)
        car_width=17.0,  # 차량 폭 (cm)
        max_steering_angle=20.0,  # 최대 조향각 (도)
        max_speed=100.0,  # 최대 속도 (cm/s)
        dt=0.1,  # 시뮬레이션 시간 간격 (초)
        friction_variation=0.1,  # 마찰 계수 변동 (Sim-to-Real)
        render_mode=None,
    ):
        super().__init__()

        # 파라미터 저장
        self.num_ultrasonic_sensors = num_ultrasonic_sensors
        self.sensor_angles = np.array(sensor_angles[:num_ultrasonic_sensors]) * np.pi / 180.0  # 라디안 변환
        self.sensor_max_range = sensor_max_range
        self.sensor_noise_std_dev = sensor_noise_std_dev
        self.track_width_min = track_width_min
        self.track_width_max = track_width_max
        self.track_length_min = track_length_min
        self.track_length_max = track_length_max
        self.car_length = car_length
        self.car_width = car_width
        self.max_steering_angle = max_steering_angle * np.pi / 180.0  # 라디안 변환
        self.max_speed = max_speed
        self.dt = dt
        self.friction_variation = friction_variation
        self.render_mode = render_mode

        # 상태 공간: [초음파 거리들, 카메라 라인 에러] (정규화됨)
        # 각 센서 거리는 [0, 1]로 정규화 (sensor_max_range로 나눔)
        # 카메라 라인 에러는 [-1, 1]로 정규화 (트랙 폭의 절반으로 나눔)
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(num_ultrasonic_sensors + 1,),
            dtype=np.float32
        )

        # 액션 공간: [조향각, 스로틀] (연속)
        # 조향각: [-1, 1] -> [-max_steering_angle, max_steering_angle]
        # 스로틀: [0, 1] -> [0, max_speed]
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )

        # 트랙 및 차량 상태 초기화
        self.track_center = None
        self.track_width = None
        self.track_left = None
        self.track_right = None
        self.car_state = None  # [x, y, yaw, velocity]
        self.track_length = None
        self.center_line_error = None  # 카메라 라인 에러 (크로스 트랙 에러)

        # 결승점 관련
        self.finish_line = None  # 결승선 위치 및 방향
        self.start_position = None  # 시작 위치
        self.start_yaw = None  # 시작 방향
        self.lap_count = 0  # 완주 횟수
        self.last_crossed_idx = 0  # 마지막으로 통과한 트랙 인덱스
        self.episode_start_time = 0.0  # 에피소드 시작 시간
        self.best_lap_time = None  # 최고 랩 타임

        # 렌더링용
        self.fig = None
        self.ax = None

    def _generate_random_track(self):
        """
        F1 스타일의 다이나믹한 랜덤 폐루프 트랙 생성
        직선, 급커브, S자 커브, 다양한 반지름의 커브를 조합
        """
        # 트랙 길이 랜덤 선택
        self.track_length = np.random.uniform(
            self.track_length_min, self.track_length_max
        )

        # 트랙 폭 랜덤 선택
        self.track_width = np.random.uniform(
            self.track_width_min, self.track_width_max
        )

        # F1 스타일 트랙 생성: 세그먼트 기반 접근
        max_attempts = 5
        for attempt in range(max_attempts):
            try:
                segments = self._generate_f1_style_segments()

                # 세그먼트를 연결하여 제어점 생성
                control_points = self._connect_segments(segments)

                # 제어점 검증
                if len(control_points) < 4:
                    raise ValueError("Not enough control points")

                # 제어점이 충분히 분산되어 있는지 확인
                point_distances = np.linalg.norm(
                    np.diff(control_points, axis=0), axis=1
                )
                if np.max(point_distances) < 1.0 or np.min(point_distances) < 0.1:
                    raise ValueError("Control points too close together")

                # 스플라인 보간으로 부드러운 트랙 생성
                # k=3 (cubic)을 사용하려면 최소 4개 이상의 점이 필요
                k = min(3, len(control_points) - 1) if len(control_points) > 3 else 1

                tck, u = splprep([control_points[:, 0], control_points[:, 1]],
                                s=0, k=k, per=True)

                # 더 많은 점으로 평가하여 부드러운 트랙 생성
                u_new = np.linspace(0, 1, int(self.track_length / 2))
                center_smooth = splev(u_new, tck)
                self.track_center = np.array(center_smooth).T

                # 트랙 경계 생성 (중심선으로부터 수직 거리)
                self._generate_track_boundaries()

                # 성공적으로 생성됨
                break

            except (ValueError, Exception) as e:
                if attempt == max_attempts - 1:
                    # 마지막 시도 실패 시 간단한 원형 트랙으로 폴백
                    print(f"[경고] F1 스타일 트랙 생성 실패, 원형 트랙으로 폴백: {e}")
                    self._generate_fallback_track()
                    break
                # 재시도
                continue

        # 트랙 경계가 생성되지 않았으면 생성
        if self.track_left is None or self.track_right is None:
            self._generate_track_boundaries()

    def _generate_track_boundaries(self):
        """
        트랙 경계 생성 (중심선으로부터 수직 거리)
        """
        if self.track_center is None or len(self.track_center) == 0:
            return

        self.track_left = []
        self.track_right = []

        for i in range(len(self.track_center)):
            # 현재 점과 다음 점 사이의 방향 벡터
            if i < len(self.track_center) - 1:
                dx = self.track_center[i + 1, 0] - self.track_center[i, 0]
                dy = self.track_center[i + 1, 1] - self.track_center[i, 1]
            else:
                dx = self.track_center[0, 0] - self.track_center[i, 0]
                dy = self.track_center[0, 1] - self.track_center[i, 1]

            # 수직 벡터 (왼쪽/오른쪽)
            length = np.sqrt(dx**2 + dy**2)
            if length > 0:
                perp_x = -dy / length
                perp_y = dx / length
            else:
                perp_x = 1.0
                perp_y = 0.0

            # 트랙 경계점
            half_width = self.track_width / 2.0
            self.track_left.append(
                self.track_center[i] + half_width * np.array([perp_x, perp_y])
            )
            self.track_right.append(
                self.track_center[i] - half_width * np.array([perp_x, perp_y])
            )

        self.track_left = np.array(self.track_left)
        self.track_right = np.array(self.track_right)

    def _generate_f1_style_segments(self):
        """
        F1 스타일 세그먼트 생성: 직선, 급커브, S자 커브 등
        """
        segments = []
        current_pos = np.array([0.0, 0.0])
        current_angle = 0.0
        total_length = 0.0
        target_length = self.track_length

        # 세그먼트 타입: 'straight', 'gentle_curve', 'sharp_curve', 'hairpin', 'chicane', 's_curve'
        segment_types = ['straight', 'gentle_curve', 'sharp_curve', 'hairpin', 'chicane', 's_curve']

        while total_length < target_length * 0.9:  # 90% 채우면 종료
            # 세그먼트 타입 랜덤 선택 (가중치 적용)
            weights = [0.2, 0.25, 0.2, 0.1, 0.1, 0.15]  # 직선과 완만한 커브가 더 많음
            seg_type = np.random.choice(segment_types, p=weights)

            if seg_type == 'straight':
                # 직선 구간
                length = np.random.uniform(50, 150)
                angle_change = np.random.uniform(-15, 15) * np.pi / 180  # 약간의 방향 변화
                segments.append({
                    'type': 'straight',
                    'length': length,
                    'angle': current_angle + angle_change,
                    'start_pos': current_pos.copy()
                })
                current_angle += angle_change
                current_pos += length * np.array([np.cos(current_angle), np.sin(current_angle)])
                total_length += length

            elif seg_type == 'gentle_curve':
                # 완만한 커브 (반지름 큰 커브)
                radius = np.random.uniform(80, 200)
                angle = np.random.uniform(30, 90) * np.pi / 180  # 커브 각도
                direction = np.random.choice([-1, 1])  # 좌회전 또는 우회전
                segments.append({
                    'type': 'gentle_curve',
                    'radius': radius,
                    'angle': angle,
                    'direction': direction,
                    'start_pos': current_pos.copy(),
                    'start_angle': current_angle
                })
                # 원호의 중심 계산
                center = current_pos + radius * direction * np.array([
                    -np.sin(current_angle), np.cos(current_angle)
                ])
                # 원호의 끝점 계산
                end_angle = current_angle + direction * angle
                current_pos = center + radius * np.array([
                    np.cos(end_angle), np.sin(end_angle)
                ])
                current_angle = end_angle
                total_length += radius * angle

            elif seg_type == 'sharp_curve':
                # 급커브 (반지름 작은 커브)
                radius = np.random.uniform(30, 60)
                angle = np.random.uniform(60, 120) * np.pi / 180
                direction = np.random.choice([-1, 1])
                segments.append({
                    'type': 'sharp_curve',
                    'radius': radius,
                    'angle': angle,
                    'direction': direction,
                    'start_pos': current_pos.copy(),
                    'start_angle': current_angle
                })
                center = current_pos + radius * direction * np.array([
                    -np.sin(current_angle), np.cos(current_angle)
                ])
                end_angle = current_angle + direction * angle
                current_pos = center + radius * np.array([
                    np.cos(end_angle), np.sin(end_angle)
                ])
                current_angle = end_angle
                total_length += radius * angle

            elif seg_type == 'hairpin':
                # 헤어핀 턴 (180도 급커브)
                radius = np.random.uniform(25, 45)
                direction = np.random.choice([-1, 1])
                segments.append({
                    'type': 'hairpin',
                    'radius': radius,
                    'angle': np.pi,  # 180도
                    'direction': direction,
                    'start_pos': current_pos.copy(),
                    'start_angle': current_angle
                })
                center = current_pos + radius * direction * np.array([
                    -np.sin(current_angle), np.cos(current_angle)
                ])
                end_angle = current_angle + direction * np.pi
                current_pos = center + radius * np.array([
                    np.cos(end_angle), np.sin(end_angle)
                ])
                current_angle = end_angle
                total_length += radius * np.pi

            elif seg_type == 'chicane':
                # 시케인 (좌우 급커브 연속)
                length1 = np.random.uniform(20, 40)
                angle1 = np.random.uniform(30, 50) * np.pi / 180
                direction1 = np.random.choice([-1, 1])
                length2 = np.random.uniform(20, 40)
                angle2 = -direction1 * np.random.uniform(30, 50) * np.pi / 180  # 반대 방향

                segments.append({
                    'type': 'chicane',
                    'length1': length1,
                    'angle1': angle1,
                    'direction1': direction1,
                    'length2': length2,
                    'angle2': angle2,
                    'start_pos': current_pos.copy(),
                    'start_angle': current_angle
                })
                # 첫 번째 커브
                current_angle += direction1 * angle1
                current_pos += length1 * np.array([np.cos(current_angle), np.sin(current_angle)])
                # 두 번째 커브
                current_angle += angle2
                current_pos += length2 * np.array([np.cos(current_angle), np.sin(current_angle)])
                total_length += length1 + length2

            elif seg_type == 's_curve':
                # S자 커브 (좌우 연속 커브)
                radius = np.random.uniform(40, 80)
                angle = np.random.uniform(60, 90) * np.pi / 180
                direction = np.random.choice([-1, 1])

                segments.append({
                    'type': 's_curve',
                    'radius': radius,
                    'angle': angle,
                    'direction': direction,
                    'start_pos': current_pos.copy(),
                    'start_angle': current_angle
                })
                # 첫 번째 커브
                center1 = current_pos + radius * direction * np.array([
                    -np.sin(current_angle), np.cos(current_angle)
                ])
                mid_angle = current_angle + direction * angle
                mid_pos = center1 + radius * np.array([
                    np.cos(mid_angle), np.sin(mid_angle)
                ])
                # 두 번째 커브 (반대 방향)
                center2 = mid_pos + radius * (-direction) * np.array([
                    -np.sin(mid_angle), np.cos(mid_angle)
                ])
                end_angle = mid_angle + (-direction) * angle
                current_pos = center2 + radius * np.array([
                    np.cos(end_angle), np.sin(end_angle)
                ])
                current_angle = end_angle
                total_length += 2 * radius * angle

        return segments

    def _connect_segments(self, segments):
        """
        세그먼트를 연결하여 제어점 배열 생성
        """
        control_points = []

        for seg in segments:
            if seg['type'] == 'straight':
                # 직선: 시작점과 끝점
                start = seg['start_pos']
                end = start + seg['length'] * np.array([
                    np.cos(seg['angle']), np.sin(seg['angle'])
                ])
                control_points.append(start)
                control_points.append(end)

            elif seg['type'] in ['gentle_curve', 'sharp_curve', 'hairpin']:
                # 원호: 여러 점으로 샘플링
                radius = seg['radius']
                angle = seg['angle']
                direction = seg['direction']
                start_angle = seg['start_angle']
                center = seg['start_pos'] + radius * direction * np.array([
                    -np.sin(start_angle), np.cos(start_angle)
                ])

                num_points = max(5, int(angle * radius / 10))  # 곡률에 따라 점 개수 조정
                angles = np.linspace(start_angle, start_angle + direction * angle, num_points)
                for a in angles:
                    point = center + radius * np.array([np.cos(a), np.sin(a)])
                    control_points.append(point)

            elif seg['type'] == 'chicane':
                # 시케인: 두 개의 짧은 직선
                start = seg['start_pos']
                angle1 = seg['start_angle'] + seg['direction1'] * seg['angle1']
                mid = start + seg['length1'] * np.array([np.cos(angle1), np.sin(angle1)])
                angle2 = angle1 + seg['angle2']
                end = mid + seg['length2'] * np.array([np.cos(angle2), np.sin(angle2)])
                control_points.append(start)
                control_points.append(mid)
                control_points.append(end)

            elif seg['type'] == 's_curve':
                # S자 커브: 두 개의 원호
                radius = seg['radius']
                angle = seg['angle']
                direction = seg['direction']
                start_angle = seg['start_angle']

                # 첫 번째 커브
                center1 = seg['start_pos'] + radius * direction * np.array([
                    -np.sin(start_angle), np.cos(start_angle)
                ])
                mid_angle = start_angle + direction * angle
                num_points1 = max(3, int(angle * radius / 10))
                angles1 = np.linspace(start_angle, mid_angle, num_points1)
                for a in angles1:
                    point = center1 + radius * np.array([np.cos(a), np.sin(a)])
                    control_points.append(point)

                # 두 번째 커브
                mid_pos = center1 + radius * np.array([np.cos(mid_angle), np.sin(mid_angle)])
                center2 = mid_pos + radius * (-direction) * np.array([
                    -np.sin(mid_angle), np.cos(mid_angle)
                ])
                end_angle = mid_angle + (-direction) * angle
                num_points2 = max(3, int(angle * radius / 10))
                angles2 = np.linspace(mid_angle, end_angle, num_points2)
                for a in angles2:
                    point = center2 + radius * np.array([np.cos(a), np.sin(a)])
                    control_points.append(point)

        # 제어점을 numpy 배열로 변환
        if len(control_points) == 0:
            # 제어점이 없으면 기본 원형 트랙 생성
            return self._generate_simple_circle_points()

        control_points = np.array(control_points)

        # 중복 점 제거 (너무 가까운 점들)
        if len(control_points) > 1:
            distances = np.linalg.norm(np.diff(control_points, axis=0), axis=1)
            min_dist = 5.0  # 최소 거리
            keep_mask = np.ones(len(control_points), dtype=bool)
            for i in range(1, len(control_points)):
                if i-1 < len(distances) and distances[i-1] < min_dist:
                    keep_mask[i] = False
            control_points = control_points[keep_mask]

        # 최소 4개 이상의 점이 필요
        if len(control_points) < 4:
            return self._generate_simple_circle_points()

        # 폐루프를 위해 시작점과 끝점이 가까워지도록 조정
        if len(control_points) > 0:
            # 시작점과 끝점 사이의 거리 계산
            start_end_dist = np.linalg.norm(control_points[0] - control_points[-1])
            if start_end_dist > 50:  # 너무 멀면 연결
                # 끝점을 시작점으로 이동하는 세그먼트 추가
                num_connect_points = max(3, int(start_end_dist / 20))
                connect_points = np.linspace(
                    control_points[-1], control_points[0], num_connect_points
                )
                control_points = np.vstack([control_points, connect_points[1:]])
            else:
                # 가까우면 첫 점을 마지막에 추가
                control_points = np.vstack([control_points, control_points[0]])

        return control_points

    def _generate_simple_circle_points(self):
        """
        간단한 원형 제어점 생성 (폴백용)
        """
        base_radius = self.track_length / (2 * np.pi)
        num_points = 8
        angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)

        control_points = np.zeros((num_points, 2))
        for i in range(num_points):
            radius = base_radius + np.random.uniform(-base_radius * 0.1, base_radius * 0.1)
            control_points[i, 0] = radius * np.cos(angles[i])
            control_points[i, 1] = radius * np.sin(angles[i])

        # 폐루프
        control_points = np.vstack([control_points, control_points[0]])

        return control_points

    def _generate_fallback_track(self):
        """
        폴백: 간단한 원형 트랙 생성 (F1 스타일 생성 실패 시)
        """
        # 트랙 길이 랜덤 선택
        self.track_length = np.random.uniform(
            self.track_length_min, self.track_length_max
        )

        # 트랙 폭 랜덤 선택
        self.track_width = np.random.uniform(
            self.track_width_min, self.track_width_max
        )

        # 간단한 원형 트랙 생성
        base_radius = self.track_length / (2 * np.pi)
        num_points = max(20, int(self.track_length / 10))
        angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)

        # 약간의 변형을 주어 완전한 원이 아니게
        radii = base_radius + np.random.uniform(
            -base_radius * 0.2, base_radius * 0.2, num_points
        )

        # 중심선 생성
        self.track_center = np.zeros((num_points, 2))
        for i in range(num_points):
            self.track_center[i, 0] = radii[i] * np.cos(angles[i])
            self.track_center[i, 1] = radii[i] * np.sin(angles[i])

        # 트랙 경계 생성
        self._generate_track_boundaries()

    def _get_nearest_center_point(self, x, y):
        """
        차량 위치에서 가장 가까운 트랙 중심선 점 찾기
        """
        distances = np.sqrt(
            (self.track_center[:, 0] - x)**2 + (self.track_center[:, 1] - y)**2
        )
        idx = np.argmin(distances)
        return idx, self.track_center[idx]

    def _calculate_cross_track_error(self, x, y, yaw):
        """
        크로스 트랙 에러 계산 (카메라 라인 에러 시뮬레이션)
        """
        nearest_idx, nearest_center = self._get_nearest_center_point(x, y)

        # 차량에서 중심선까지의 벡터
        dx = nearest_center[0] - x
        dy = nearest_center[1] - y

        # 차량의 전방 방향 벡터
        forward_x = np.cos(yaw)
        forward_y = np.sin(yaw)

        # 크로스 트랙 에러 (차량의 좌우 방향으로의 거리)
        # 오른쪽이 양수, 왼쪽이 음수
        cross_error = dx * (-forward_y) + dy * forward_x

        # 정규화 (트랙 폭의 절반으로 나눔)
        normalized_error = cross_error / (self.track_width / 2.0)
        normalized_error = np.clip(normalized_error, -1.0, 1.0)

        return normalized_error

    def _raycast_ultrasonic(self, x, y, yaw, sensor_angle):
        """
        레이캐스팅을 사용하여 초음파 센서 거리 계산
        """
        # 트랙 경계가 아직 생성되지 않았으면 최대 거리 반환
        if self.track_left is None or self.track_right is None:
            return self.sensor_max_range

        # 트랙 경계가 비어있으면 최대 거리 반환
        if len(self.track_left) == 0 or len(self.track_right) == 0:
            return self.sensor_max_range

        # 센서의 절대 각도
        absolute_angle = yaw + sensor_angle

        # 레이 방향 벡터
        ray_dir = np.array([np.cos(absolute_angle), np.sin(absolute_angle)])

        # 레이 시작점
        ray_start = np.array([x, y])

        min_distance = self.sensor_max_range

        # 트랙 경계와의 교차점 찾기
        for boundary in [self.track_left, self.track_right]:
            if boundary is None or len(boundary) < 2:
                continue
            for i in range(len(boundary) - 1):
                # 선분의 두 점
                p1 = boundary[i]
                p2 = boundary[i + 1]

                # 선분과 레이의 교차점 계산
                # 레이: ray_start + t * ray_dir
                # 선분: p1 + s * (p2 - p1)
                seg_dir = p2 - p1

                # 교차점 계산 (2D 선분-레이 교차)
                denom = ray_dir[0] * seg_dir[1] - ray_dir[1] * seg_dir[0]

                if abs(denom) > 1e-6:
                    t = ((p1[0] - ray_start[0]) * seg_dir[1] -
                         (p1[1] - ray_start[1]) * seg_dir[0]) / denom
                    s = ((p1[0] - ray_start[0]) * ray_dir[1] -
                         (p1[1] - ray_start[1]) * ray_dir[0]) / denom

                    if t > 0 and 0 <= s <= 1:
                        intersection = ray_start + t * ray_dir
                        distance = np.linalg.norm(intersection - ray_start)
                        min_distance = min(min_distance, distance)

        return min_distance

    def _kinematic_bicycle_model(self, x, y, yaw, velocity, steering_angle, throttle):
        """
        운동학적 자전거 모델 (Kinematic Bicycle Model)
        """
        # 마찰 변동 (Sim-to-Real)
        friction_coeff = 1.0 - np.random.uniform(
            -self.friction_variation, self.friction_variation
        )

        # 속도 업데이트 (스로틀과 마찰)
        acceleration = throttle * self.max_speed * friction_coeff
        new_velocity = velocity + acceleration * self.dt
        new_velocity = np.clip(new_velocity, 0, self.max_speed)

        # 조향각 제한
        steering_angle = np.clip(steering_angle, -self.max_steering_angle, self.max_steering_angle)

        # 자전거 모델 파라미터 (차량 길이)
        L = self.car_length

        # 각속도 계산
        if abs(new_velocity) > 0.1:
            angular_velocity = (new_velocity / L) * np.tan(steering_angle)
        else:
            angular_velocity = 0.0

        # 상태 업데이트
        new_yaw = yaw + angular_velocity * self.dt
        new_x = x + new_velocity * np.cos(new_yaw) * self.dt
        new_y = y + new_velocity * np.sin(new_yaw) * self.dt

        return new_x, new_y, new_yaw, new_velocity

    def _check_collision(self, x, y):
        """
        차량이 트랙 경계와 충돌했는지 확인
        """
        # 차량의 네 모서리 점
        corners = [
            [x + self.car_length/2, y + self.car_width/2],
            [x + self.car_length/2, y - self.car_width/2],
            [x - self.car_length/2, y + self.car_width/2],
            [x - self.car_length/2, y - self.car_width/2],
        ]

        for corner in corners:
            # 트랙 내부에 있는지 확인 (점이 폴리곤 내부에 있는지)
            # 간단한 방법: 가장 가까운 중심선 점까지의 거리가 트랙 폭/2보다 큰지 확인
            _, nearest_center = self._get_nearest_center_point(corner[0], corner[1])
            dist_to_center = np.linalg.norm(
                np.array(corner) - nearest_center
            )

            if dist_to_center > self.track_width / 2.0 + 2.0:  # 여유 공간
                return True

        return False

    def reset(self, seed=None, options=None):
        """
        환경 리셋: 새로운 랜덤 트랙 생성 및 차량 초기화
        """
        super().reset(seed=seed)

        # 랜덤 트랙 생성
        self._generate_random_track()

        # 차량을 트랙 시작점에 배치
        start_point = self.track_center[0]
        start_yaw = np.arctan2(
            self.track_center[1, 1] - self.track_center[0, 1],
            self.track_center[1, 0] - self.track_center[0, 0]
        )

        # 초기 속도를 약간 주어서 학습 시작을 쉽게
        initial_velocity = self.max_speed * 0.2  # 최대 속도의 20%

        self.car_state = np.array([
            start_point[0],
            start_point[1],
            start_yaw,
            initial_velocity  # 초기 속도 (0이 아닌 작은 값)
        ])

        # 스텝 카운터 리셋
        self.step_count = 0

        # 결승선 설정 (시작점)
        self.start_position = start_point.copy()
        self.start_yaw = start_yaw
        # 결승선 방향 (트랙 진행 방향)
        if len(self.track_center) > 1:
            finish_dir = self.track_center[1] - self.track_center[0]
            finish_dir = finish_dir / np.linalg.norm(finish_dir)
        else:
            finish_dir = np.array([np.cos(start_yaw), np.sin(start_yaw)])
        self.finish_line = {
            'position': start_point,
            'direction': finish_dir,  # 정규화된 방향 벡터
            'normal': np.array([-finish_dir[1], finish_dir[0]])  # 수직 벡터
        }

        # 완주 관련 변수 초기화
        self.lap_count = 0
        self.last_crossed_idx = 0
        self.episode_start_time = 0.0
        self.best_lap_time = None
        self.last_side = None  # 결승선 통과 추적용

        # 초기 관측값 계산
        observation = self._get_observation()

        info = {}

        if self.render_mode == "human":
            self.render()

        return observation, info

    def _get_observation(self):
        """
        현재 상태 관측값 계산 (정규화됨)
        """
        x, y, yaw, velocity = self.car_state

        # 초음파 센서 거리 계산
        ultrasonic_distances = []
        for sensor_angle in self.sensor_angles:
            distance = self._raycast_ultrasonic(x, y, yaw, sensor_angle)

            # 가우시안 노이즈 추가 (Sim-to-Real)
            noise = np.random.normal(0, self.sensor_noise_std_dev)
            distance = distance + noise
            distance = np.clip(distance, 0, self.sensor_max_range)

            # 정규화 [0, 1]
            normalized_distance = distance / self.sensor_max_range
            ultrasonic_distances.append(normalized_distance)

        # 카메라 라인 에러 계산
        self.center_line_error = self._calculate_cross_track_error(x, y, yaw)

        # 관측값 결합
        observation = np.array(ultrasonic_distances + [self.center_line_error], dtype=np.float32)

        return observation

    def step(self, action):
        """
        환경 스텝 실행
        """
        # 액션 해석
        steering_normalized = action[0]  # [-1, 1]
        throttle_normalized = action[1]  # [0, 1]

        steering_angle = steering_normalized * self.max_steering_angle
        throttle = throttle_normalized

        # 차량 상태 업데이트 (운동학적 자전거 모델)
        x, y, yaw, velocity = self.car_state
        new_x, new_y, new_yaw, new_velocity = self._kinematic_bicycle_model(
            x, y, yaw, velocity, steering_angle, throttle
        )

        self.car_state = np.array([new_x, new_y, new_yaw, new_velocity])

        # 충돌 확인
        collision = self._check_collision(new_x, new_y)

        # 결승선 통과 확인
        lap_completed, lap_time = self._check_finish_line_crossing(new_x, new_y, new_yaw)

        # 완주 메시지 출력
        if lap_completed:
            print(f"\n🏁 완주! 랩 타임: {lap_time:.2f}초")
            if self.best_lap_time == lap_time:
                print(f"⭐ 최고 기록 갱신!")

        # 리워드 계산
        reward = self._calculate_reward(
            collision, new_velocity, self.center_line_error,
            lap_completed, lap_time
        )

        # 종료 조건
        terminated = collision

        # 완주 시 종료 (선택사항: 여러 랩을 돌 수도 있음)
        # 일단 첫 완주 시 종료로 설정
        if lap_completed and self.lap_count >= 1:
            terminated = True

        # 시간 제한 (너무 오래 주행하면 종료)
        # 트랙 길이에 비례하여 시간 제한 설정
        max_steps = int(self.track_length / 5.0)  # 트랙 길이의 1/5만큼 스텝
        if not hasattr(self, 'step_count'):
            self.step_count = 0
        self.step_count += 1
        truncated = self.step_count >= max_steps

        # 다음 관측값
        observation = self._get_observation()

        info = {
            "collision": collision,
            "velocity": new_velocity,
            "cross_track_error": self.center_line_error,
            "lap_count": self.lap_count,
            "lap_time": lap_time if lap_completed else None,
            "best_lap_time": self.best_lap_time,
        }

        if self.render_mode == "human":
            self.render()

        return observation, reward, terminated, truncated, info

    def _check_finish_line_crossing(self, x, y, yaw):
        """
        결승선 통과 확인
        시작점을 다시 통과했는지 확인 (정방향으로만)

        Returns:
            (lap_completed, lap_time): 완주 여부와 랩 타임
        """
        if self.finish_line is None:
            return False, None

        # 시작점과의 거리
        dist_to_start = np.linalg.norm(
            np.array([x, y]) - self.finish_line['position']
        )

        # 결승선 근처에 있는지 확인 (트랙 폭의 1.5배 이내)
        finish_threshold = self.track_width * 1.5

        if dist_to_start < finish_threshold:
            # 진행 방향 확인 (전방으로 통과해야 함)
            to_finish = self.finish_line['position'] - np.array([x, y])
            to_finish = to_finish / (np.linalg.norm(to_finish) + 1e-6)

            # 차량의 전방 방향
            forward_dir = np.array([np.cos(yaw), np.sin(yaw)])

            # 내적이 양수면 전방으로 진행 중
            dot_product = np.dot(to_finish, forward_dir)

            # 결승선의 수직 벡터와의 관계 확인 (결승선을 통과했는지)
            finish_normal = self.finish_line['normal']
            side = np.dot(np.array([x, y]) - self.finish_line['position'], finish_normal)

            # 이전에 통과한 적이 있고, 반대편으로 넘어갔는지 확인
            if self.last_side is not None:
                if (self.last_side * side < 0) and dot_product > 0.3:  # 반대편으로 통과
                    # 완주! (단, 이미 한 바퀴 이상 돌았어야 함)
                    if self.lap_count == 0 or self.step_count > 50:  # 최소 50 스텝은 지나야 완주로 인정
                        self.lap_count += 1
                        lap_time = self.step_count * self.dt

                        # 최고 랩 타임 업데이트
                        if self.best_lap_time is None or lap_time < self.best_lap_time:
                            self.best_lap_time = lap_time

                        self.last_side = side
                        return True, lap_time

            if self.last_side is None:
                self.last_side = side
            elif abs(side) < abs(self.last_side):  # 결승선에 더 가까워지면 업데이트
                self.last_side = side

        return False, None

    def _calculate_reward(self, collision, velocity, cross_track_error,
                         lap_completed=False, lap_time=None):
        """
        개선된 리워드 함수 (완주 시간 보상 추가)
        - 생존 보상 (매 스텝마다)
        - 속도 보상
        - 중심선 유지 보상
        - 진행 보상 (트랙을 따라가는 것)
        - 완주 시간 보상 (빠를수록 큰 보상)
        - 충돌 페널티
        """
        if collision:
            return -50.0  # 충돌 페널티 (조정됨)

        # 완주 보상 (매우 큰 보상)
        if lap_completed and lap_time is not None:
            # 기준 시간 설정 (트랙 길이에 비례)
            # 트랙 길이를 평균 속도로 나눈 시간을 기준으로
            base_time = self.track_length / (self.max_speed * 0.5)  # 평균 속도의 50%로 주행 시 예상 시간

            # 시간이 빠를수록 큰 보상
            # base_time보다 빠르면 보너스, 느리면 페널티
            time_ratio = base_time / (lap_time + 1e-6)  # 빠를수록 큰 값

            # 완주 보상: 기본 보상 + 시간 보너스
            completion_base = 100.0  # 완주 기본 보상
            time_bonus = 50.0 * max(0, time_ratio - 0.5)  # 기준보다 빠르면 보너스

            return completion_base + time_bonus

        # 1. 생존 보상 (매 스텝마다 작은 보상)
        survival_reward = 0.1

        # 2. 속도 보상 (더 큰 가중치)
        speed_reward = (velocity / self.max_speed) * 2.0

        # 3. 중심선 유지 보상 (에러가 작을수록 좋음)
        # 중심선에 가까울수록 큰 보상
        center_reward = (1.0 - abs(cross_track_error)) * 1.5

        # 4. 진행 보상 (속도가 있을 때만)
        # 차량이 움직이고 있으면 추가 보상
        progress_reward = 0.0
        if velocity > 10.0:  # 최소 속도 이상일 때
            progress_reward = 0.5 * (velocity / self.max_speed)

        total_reward = survival_reward + speed_reward + center_reward + progress_reward

        return total_reward

    def render(self):
        """
        환경 시각화
        """
        if self.render_mode is None:
            return

        if self.fig is None:
            plt.ion()
            self.fig, self.ax = plt.subplots(figsize=(12, 12))
            self.ax.set_aspect('equal')
            self.ax.set_xlabel('X (cm)', fontsize=12)
            self.ax.set_ylabel('Y (cm)', fontsize=12)
            self.ax.set_title('Random Track Environment - Racing Car',
                            fontsize=14, fontweight='bold')
            self.ax.set_facecolor('#1a1a1a')  # 어두운 배경

        self.ax.clear()

        # 트랙 그리기
        if self.track_center is not None:
            # 트랙 내부 영역 채우기 (회색 배경)
            track_polygon = np.vstack([
                self.track_left,
                self.track_right[::-1]  # 역순으로 닫힌 폴리곤 만들기
            ])
            track_fill = Polygon(track_polygon, facecolor='#2d2d2d',
                               edgecolor='none', alpha=0.3, zorder=1)
            self.ax.add_patch(track_fill)

            # 트랙 경계선 (두껍게, 검은색)
            self.ax.plot(self.track_left[:, 0], self.track_left[:, 1],
                        'k-', linewidth=4, label='Track Boundary', zorder=2)
            self.ax.plot(self.track_right[:, 0], self.track_right[:, 1],
                        'k-', linewidth=4, zorder=2)

            # 중심선 (초록색 점선)
            self.ax.plot(self.track_center[:, 0], self.track_center[:, 1],
                        'g--', linewidth=2, alpha=0.7, label='Center Line', zorder=3)

            # 결승선 표시 (초록색 선과 화살표)
            if self.finish_line is not None:
                finish_pos = self.finish_line['position']
                finish_normal = self.finish_line['normal']
                finish_dir = self.finish_line['direction']

                # 결승선 그리기 (트랙 폭만큼)
                line_length = self.track_width * 1.5
                line_start = finish_pos - finish_normal * line_length / 2
                line_end = finish_pos + finish_normal * line_length / 2
                self.ax.plot([line_start[0], line_end[0]], [line_start[1], line_end[1]],
                            'g-', linewidth=3, alpha=0.8, zorder=4, label='Finish Line')

                # 시작 방향 화살표
                arrow_length = 30
                self.ax.arrow(finish_pos[0], finish_pos[1],
                            finish_dir[0] * arrow_length, finish_dir[1] * arrow_length,
                            head_width=10, head_length=8,
                            fc='green', ec='green', zorder=4)

                # 완주 정보 표시
                if self.lap_count > 0:
                    info_text = f"Laps: {self.lap_count}"
                    if self.best_lap_time is not None:
                        info_text += f"\nBest: {self.best_lap_time:.2f}s"
                    self.ax.text(0.02, 0.98, info_text,
                               transform=self.ax.transAxes,
                               fontsize=10, verticalalignment='top',
                               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
                               zorder=10)

            # 트랙 표면 느낌 (중심선 주변 점선)
            for offset in [-self.track_width/4, self.track_width/4]:
                track_inner_left = []
                track_inner_right = []
                for i in range(len(self.track_center)):
                    if i < len(self.track_center) - 1:
                        dx = self.track_center[i + 1, 0] - self.track_center[i, 0]
                        dy = self.track_center[i + 1, 1] - self.track_center[i, 1]
                    else:
                        dx = self.track_center[0, 0] - self.track_center[i, 0]
                        dy = self.track_center[0, 1] - self.track_center[i, 1]
                    length = np.sqrt(dx**2 + dy**2)
                    if length > 0:
                        perp_x = -dy / length
                        perp_y = dx / length
                        inner_point = self.track_center[i] + offset * np.array([perp_x, perp_y])
                        if offset < 0:
                            track_inner_left.append(inner_point)
                        else:
                            track_inner_right.append(inner_point)

                if track_inner_left:
                    inner_left = np.array(track_inner_left)
                    self.ax.plot(inner_left[:, 0], inner_left[:, 1],
                               'w:', linewidth=1, alpha=0.3, zorder=2)
                if track_inner_right:
                    inner_right = np.array(track_inner_right)
                    self.ax.plot(inner_right[:, 0], inner_right[:, 1],
                               'w:', linewidth=1, alpha=0.3, zorder=2)

        # 차량 그리기
        if self.car_state is not None:
            x, y, yaw, velocity = self.car_state

            # 차량 사각형의 네 모서리 (회전 전)
            corners = np.array([
                [-self.car_length/2, -self.car_width/2],
                [self.car_length/2, -self.car_width/2],
                [self.car_length/2, self.car_width/2],
                [-self.car_length/2, self.car_width/2]
            ])

            # 회전 변환 적용
            cos_yaw = np.cos(yaw)
            sin_yaw = np.sin(yaw)
            rotation_matrix = np.array([
                [cos_yaw, -sin_yaw],
                [sin_yaw, cos_yaw]
            ])

            rotated_corners = corners @ rotation_matrix.T
            rotated_corners[:, 0] += x
            rotated_corners[:, 1] += y

            # Polygon으로 그리기 (차량)
            car_polygon = Polygon(rotated_corners,
                                 facecolor='#1f77b4',
                                 edgecolor='#0a4d7a', linewidth=2,
                                 alpha=0.8, zorder=5)
            self.ax.add_patch(car_polygon)

            # 차량 방향 표시 (전방 화살표)
            arrow_length = self.car_length * 0.6
            arrow_x = x + arrow_length * np.cos(yaw)
            arrow_y = y + arrow_length * np.sin(yaw)
            self.ax.arrow(x, y, arrow_length * np.cos(yaw), arrow_length * np.sin(yaw),
                         head_width=3, head_length=2, fc='yellow', ec='yellow', zorder=6)

            # 차량 중심점
            self.ax.plot(x, y, 'ro', markersize=4, zorder=6)

            # 초음파 센서 레이 표시 (더 눈에 띄게)
            colors = ['#ff4444', '#44ff44', '#4444ff']
            for i, sensor_angle in enumerate(self.sensor_angles):
                absolute_angle = yaw + sensor_angle
                distance = self._raycast_ultrasonic(x, y, yaw, sensor_angle)
                end_x = x + distance * np.cos(absolute_angle)
                end_y = y + distance * np.sin(absolute_angle)
                color = colors[i % len(colors)]
                self.ax.plot([x, end_x], [y, end_y],
                           color=color, linewidth=2, alpha=0.7, zorder=4)
                # 센서 끝점 표시
                self.ax.plot(end_x, end_y, 'o', color=color,
                           markersize=4, alpha=0.7, zorder=4)

        self.ax.legend()
        self.ax.grid(True, alpha=0.3)
        plt.draw()
        plt.pause(0.01)

    def close(self):
        """
        환경 종료
        """
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None

