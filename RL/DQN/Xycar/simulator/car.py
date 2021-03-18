import numpy as np

from simulator.utils import *


class Car(object):
    def __init__(self, position=(0, 0), yaw=0, steering_deg=0):
        self.position = position            # 현재 차량 위치(center_x, center_y)
        self.yaw = yaw                      # 현재 차량 방향
        self.steering_deg = steering_deg    # 차량 좌/우 앞바퀴의 각도

        self.velocity = 0.0         # 현재 차량 속도
        self.acceleration = 0.0     # 현채 차량 가속도

        self.max_velocity = 100.0       # 차량 속도 한계값
        self.max_acceleration = 50.0    # 차량 가속도 한계값
        self.max_steering_deg = 30.0    # 차량 앞바퀴 각도 한계값

        self.delta_acceleration = 50    # 시간 변화량동안 증가하는 가속도 변화량

        # Xycar B2 모델 차량 규격 (폭 x 길이)
        # - 실물 규격: 0.3m x 0.6m
        # - 시뮬레이터 규격: 63px x 126px
        # - 1m = 210px
        #   +---+-------------------+---+
        #   | ==+==---------------==+== |
        #   |   |         x          |  |
        #   | ==+==---------------==+== |
        #   +---+-------------------+---+
        self.meter_per_pixel = 1 / 210
        self.border_width = 63      # 차량 폭
        self.border_length = 126    # 차량 길이

        self.border_left = self.border_width / 2        # 중점 기준, 차량의 좌측 폭
        self.border_right = self.border_width / 2       # 중점 기준, 차량의 우측 폭
        self.border_front = self.border_length / 2      # 중점 기준, 차량의 전방 길이
        self.border_back = self.border_length / 2       # 중점 기준, 차량의 후방 길이

        self.wheel_width = 6        # 바퀴 폭
        self.wheel_length = 14      # 바퀴 길이

        self.wheel_width_gap = 56   # 차량 좌/우 바퀴 사이 폭
        self.wheel_length_gap = 74  # 차량 앞/뒤 바퀴 사이 길이
        
        self.wheel_left = self.wheel_width_gap / 2      # 중점 기준, 차량 좌측 바퀴 폭
        self.wheel_right = self.wheel_width_gap / 2     # 중점 기준, 차량 우측 바퀴 폭
        self.wheel_front = self.wheel_length_gap / 2    # 중점 기준, 차량 전방 바퀴 길이
        self.wheel_back = self.wheel_length_gap / 2     # 중점 기준, 차량 전방 바퀴 길이

        self.DRIVE = 1      # 차량 주행 기어
        self.REVERSE = 2    # 차량 후진 기어
        self.BREAK = 3      # 차량 정지 기어

        self.max_ultrasonic_seek_meter = 2
        self.max_ultrasonic_seek_pixel = self.max_ultrasonic_seek_meter / self.meter_per_pixel


    def reset(self):
        self.velocity = 0.0         # 현재 차량 속도
        self.acceleration = 0.0     # 현채 차량 가속도


    def get_ultrasonic_pos_and_yaw(self):
        """현재 차량에 부착되어 있는 초음파센서의 위치정보(position)와 방향정보(yaw)를 반환

        차량의 형태가 아래와 같을 때, **U**는 초음파의 위치를 나타낸다.
        +---+---------U---------+---+
        U ==+==---------------==+== U
        U   |         x          |  U
        U ==+==---------------==+== U
        +---+---------U---------+---+

        Returns:
            list: 차량에 부탁되어 있는 초음파센서의 위치 정보((x, y))
            list: 차량에 부탁되어 있는 초음파센서의 방향 정보(yaw)
        """
        # yaw 방향의 회전변환행렬
        mtx = get_rotation_matrix(self.yaw)

        ultrasonic_pos = [
            [0, -self.border_left],                     # 1번 초음파센서 위치(좌측)
            [self.border_front, -self.border_left/2],   # 2번 초음파센서 위치(좌상단)
            [self.border_front, 0],                     # 3번 초음파센서 위치(상단)
            [self.border_front, self.border_right/2],   # 4번 초음파센서 위치(우상단)
            [0, self.border_right],                     # 5번 초음파센서 위치(우측)
            [-self.border_back, self.border_right/2],   # 6번 초음파센서 위치(우하단)
            [-self.border_back, 0],                     # 7번 초음파센서 위치(하단)
            [-self.border_back, -self.border_left/2]    # 8번 초음파센서 위치(좌하단)
        ]

        ultrasonic_yaw = np.array([
            np.radians(-90),        # 1번 초음파센서 방향(-90)
            np.radians(-30),        # 2번 초음파센서 방향(-30)
            np.radians(0),          # 3번 초음파센서 방향(0)
            np.radians(30),         # 4번 초음파센서 방향(30)
            np.radians(90),         # 5번 초음파센서 방향(90)
            np.radians(150),        # 6번 초음파센서 방향(180-30)
            np.radians(180),        # 7번 초음파센서 방향(180)
            np.radians(-150),       # 8번 초음파센서 방향(-180+30)
        ])

        # yaw 방향 회전이 적용된 초음파센서 위치
        rotated_ultrasonic_pos = np.dot(ultrasonic_pos, mtx)

        # 차량의 위치 적용
        rotated_ultrasonic_pos += self.position

        # 초음파센서 방향 계산
        rotated_ultrasonic_yaw = ultrasonic_yaw + self.yaw
        rotated_ultrasonic_yaw = [ normalize_radian(rotated_yaw) for rotated_yaw in rotated_ultrasonic_yaw ]
        
        return rotated_ultrasonic_pos, rotated_ultrasonic_yaw


    def update(self, delta_time, gear, steering_deg=None):
        # 요청한 steering_deg 각도를 최소/최대값 범위 내로 조정
        if steering_deg:
            self.steering_deg = max(-self.max_steering_deg, min(steering_deg, self.max_steering_deg))

        # 정지 기어일 경우, 가속도, 속도는 0
        if gear == self.BREAK:
            self.acceleration = 0
            self.velocity = 0
        else:
            # 기어에 따른 가속도 계산
            if gear == self.DRIVE:
                self.acceleration += self.delta_acceleration * delta_time
            elif gear == self.REVERSE:
                self.acceleration -= self.delta_acceleration * delta_time
            self.acceleration = max(-self.max_acceleration, min(self.acceleration, self.max_acceleration))

            # 속도 계산
            self.velocity += self.acceleration * delta_time
            self.velocity = max(-self.max_velocity, min(self.velocity, self.max_velocity))

        # 현재 위치 계산
        x, y = self.position
        x += self.velocity * delta_time * np.cos(self.yaw)
        y += self.velocity * delta_time * np.sin(self.yaw)
        self.position = (x, y)

        # 각속도 계산
        # - steering_rad를 theta라고 할 때, tan(theta) = wheel_length_gap / radius
        # - 각속도 w = velocity / radius
        #          = velicity * tan(theta) / wheel_length_gap
        theta = np.radians(self.steering_deg)
        angular_velocity = self.velocity * np.tan(theta) / self.wheel_length_gap

        # yaw 계산
        self.yaw += angular_velocity * delta_time
        self.yaw = normalize_radian(self.yaw)
    

    def get_border_points(self):
        # 차량의 모서리 좌표
        points = [
            [-self.border_back, -self.border_left],     # 좌측 후방
            [self.border_front, -self.border_left],     # 좌측 전방
            [self.border_front, self.border_right],     # 우측 전방
            [-self.border_back, self.border_right]      # 우측 후방
        ]

        # 현재 차량 방향의 회전변환행렬
        mtx = get_rotation_matrix(self.yaw)

        # 회전이 적용된 차량의 모서리 좌표
        rotated_points = np.dot(points, mtx)
        
        # 차량의 위치 적용
        rotated_points += self.position
        return rotated_points


    def get_front_wheel_border_points(self):
        # 바퀴 1개의 모서리 좌표
        border_points = [
            [-self.wheel_length/2, -self.wheel_width/2],
            [self.wheel_length/2, -self.wheel_width/2],
            [self.wheel_length/2, self.wheel_width/2],
            [-self.wheel_length/2, self.wheel_width/2]
        ]

        # 차량 기준, 바퀴 위치 좌표
        front_points = [
            [self.wheel_front, -self.wheel_left],       # 전방 좌측
            [self.wheel_front, self.wheel_right]        # 전방 우측
        ]

        # 현재 차량 방향의 회전변환행렬
        mtx1 = get_rotation_matrix(self.yaw)

        # 현재 바퀴 방향의 회전변환행렬
        steering_rad = np.radians(self.steering_deg)
        mtx2 = get_rotation_matrix(self.yaw + steering_rad)
        
        # 회전이 적용된 바퀴의 모서리 좌표
        rotated_border_points1 = np.dot(border_points, mtx1)  # 차량의 방향만 적용
        rotated_border_points2 = np.dot(border_points, mtx2)  # 차량의 방향 + steering_rad가 적용

        # 회전이 적용된 바퀴 위치 좌표
        rotated_front_points = np.dot(front_points, mtx1)

        # 모든 회전이 적용된 바퀴의 모서리 좌표
        rotated_front_points = np.array([ rotated_border_points2 + pt for pt in rotated_front_points ])

        # 차량의 위치 적용
        rotated_front_points += self.position
        return rotated_front_points
        
    
    def get_back_wheel_border_points(self):
        # 바퀴 1개의 모서리 좌표
        border_points = [
            [-self.wheel_length/2, -self.wheel_width/2],
            [self.wheel_length/2, -self.wheel_width/2],
            [self.wheel_length/2, self.wheel_width/2],
            [-self.wheel_length/2, self.wheel_width/2]
        ]

        # 차량 기준, 바퀴 위치 좌표
        back_points = [
            [-self.wheel_back, -self.wheel_left],       # 후방 좌측
            [-self.wheel_back, self.wheel_right]        # 후방 우측
        ]

        # 현재 차량 방향의 회전변환행렬
        mtx1 = get_rotation_matrix(self.yaw)

        # 회전이 적용된 바퀴의 모서리 좌표
        rotated_border_points1 = np.dot(border_points, mtx1)  # 차량의 방향만 적용

        # 회전이 적용된 바퀴 위치 좌표
        rotated_back_points = np.dot(back_points, mtx1)

        # 모든 회전이 적용된 바퀴의 모서리 좌표
        rotated_back_points = np.array([ rotated_border_points1 + pt for pt in rotated_back_points ])

        # 차량의 위치 적용
        rotated_back_points += self.position
        return rotated_back_points
