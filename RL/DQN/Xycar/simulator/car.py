import numpy as np


def get_roate_pos(pos, yaw):
    x, y = pos
    return (x * np.cos(yaw), y * np.sin(yaw))


def get_rotation_matrix(yaw):
    return np.array([
        [np.cos(yaw), np.sin(yaw)],
        [-np.sin(yaw), np.cos(yaw)]])


def normalize_radian(rad):
    while rad < -np.pi:
        rad += 2 * np.pi
    while np.pi < rad:
        rad -= 2 * np.pi
    return rad


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


    def get_ultrasonic_pos_with_yaw(self):
        """현재 차량에 부착되어 있는 초음파센서의 위치정보(position)와 방향정보(yaw)를 반환

        차량의 형태가 아래와 같을 때, **U**는 초음파의 위치를 나타낸다.
        +---+---------U---------+---+
        | ==+==---------------==+== U
        |   |         x          |  U
        | ==+==---------------==+== U
        +---+---------U---------+---+

        Returns:
            list: 차량에 부탁되어 있는 초음파센서의 위치 정보((x, y))
            list: 차량에 부탁되어 있는 초음파센서의 방향 정보(yaw)
        """
        mtx = get_rotation_matrix(self.yaw)  # yaw 방향의 회전변환행렬

        ultrasonic_pos = [
            [0, -self.border_left],                     # 1번 초음파센서 위치(좌측)
            [self.border_front, -self.border_left/2],   # 2번 초음파센서 위치(좌상단)
            [self.border_front, 0],                     # 3번 초음파센서 위치(상단)
            [self.border_front, self.border_right/2],   # 4번 초음파센서 위치(우상단)
            [0, self.border_right]                      # 5번 초음파센서 위치(우측)
        ]

        ultrasonic_yaw = [
            np.radians(-90),        # 1번 초음파센서 방향(-90)
            np.radians(-30),        # 2번 초음파센서 방향(-30)
            np.radians(0),          # 3번 초음파센서 방향(0)
            np.radians(30),         # 4번 초음파센서 방향(30)
            np.radians(90),         # 5번 초음파센서 방향(90)
        ]

        rotated_ultrasonic_pos = np.dot(ultrasonic_pos, mtx)    # yaw 방향 회전이 적용된 초음파센서 위치
        rotated_ultrasonic_yaw = ultrasonic_yaw + self.yaw

        return rotated_ultrasonic_pos, rotated_ultrasonic_yaw


    def update(self, delta_time, gear, steering_deg=None):
        # 요청한 steering_deg 각도를 최소/최대값 범위 내로 조정
        if steering_deg:
            self.steering_deg = max(-self.max_steering_deg, min(steering_deg, self.max_steering_deg))

        # 기어에 따른 가속도 계산
        if gear == self.DRIVE:
            self.acceleration += self.delta_acceleration * delta_time
        elif gear == self.REVERSE:
            self.acceleration -= self.delta_acceleration * delta_time
        self.acceleration = max(-self.max_acceleration, min(self.acceleration, self.max_acceleration))

        # 속도 계산
        self.velocity += self.acceleration * delta_time
        self.velocity = max(-self.max_velocity, min(self.velocity, self.max_velocity))

        # 정지 기어일 경우, 가속도, 속도는 0
        if gear == self.BREAK:
            self.acceleration = 0
            self.velocity = 0

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
    

    def get_border_pos(self):
        # x 좌표 취득
        width = self.border_front + self.border_back
        xs = np.linspace(-self.border_back, self.border_front, width, endpoint=True).reshape((-1, 1))

        # y 좌표 취득
        height = self.border_left + self.border_right
        ys = np.linspace(-self.border_left, self.border_right, height, endpoint=True).reshape((-1, 1))

        # 외곽 좌표 취득
        up_pos = np.insert(xs, 1, ys[0,0], axis=1)
        down_pos = np.insert(xs, 1, ys[-1,-1], axis=1)
        left_pos = np.insert(ys, 0, xs[0,0], axis=1)
        right_pos = np.insert(ys, 0, xs[-1,-1], axis=1)

        # Affine 회전
        affine_mtx = get_rotate_mtx(-self.yaw)
        rotated_border = np.dot(np.vstack((up_pos, down_pos, left_pos, right_pos)), affine_mtx)

        return rotated_border + self.position


    def get_front_wheel_border_pos(self):
        # x 좌표 취득
        xs = np.linspace(-self.wheel_width//2, self.wheel_width//2, self.wheel_width, endpoint=True).reshape((-1, 1))

        # y 좌표 취득
        ys = np.linspace(-self.wheel_height//2, self.wheel_height//2, self.wheel_height, endpoint=True).reshape((-1, 1))

        # 외곽 좌표 취득
        up_pos = np.insert(xs, 1, ys[0,0], axis=1)
        down_pos = np.insert(xs, 1, ys[-1,-1], axis=1)
        left_pos = np.insert(ys, 0, xs[0,0], axis=1)
        right_pos = np.insert(ys, 0, xs[-1,-1], axis=1)

        # 외곽 좌료 회전
        steering_rad = np.radians(self.steering_deg)
        affine_mtx = get_rotate_mtx(-self.yaw-steering_rad)
        rotated_border = np.dot(np.vstack((up_pos, down_pos, left_pos, right_pos)), affine_mtx)

        # 바퀴 좌표 취득
        left_wheel = self.wheel_height_gap//2, -self.wheel_width_gap//2
        right_wheel = self.wheel_height_gap//2, self.wheel_width_gap//2

        # 바퀴 좌표 회전
        affine_mtx = get_rotate_mtx(-self.yaw)
        rotated_left_wheel = np.dot(left_wheel, affine_mtx)
        rotated_right_wheel = np.dot(right_wheel, affine_mtx)

        return (
            rotated_border + rotated_left_wheel + self.position,
            rotated_border + rotated_right_wheel + self.position)


    def get_back_wheel_border_pos(self):
        # x 좌표 취득
        xs = np.linspace(-self.wheel_width//2, self.wheel_width//2, self.wheel_width, endpoint=True).reshape((-1, 1))

        # y 좌표 취득
        ys = np.linspace(-self.wheel_height//2, self.wheel_height//2, self.wheel_height, endpoint=True).reshape((-1, 1))

        # 외곽 좌표 취득
        up_pos = np.insert(xs, 1, ys[0,0], axis=1)
        down_pos = np.insert(xs, 1, ys[-1,-1], axis=1)
        left_pos = np.insert(ys, 0, xs[0,0], axis=1)
        right_pos = np.insert(ys, 0, xs[-1,-1], axis=1)

        # 외곽 좌료 회전
        affine_mtx = get_rotate_mtx(-self.yaw)
        rotated_border = np.dot(np.vstack((up_pos, down_pos, left_pos, right_pos)), affine_mtx)

        # 바퀴 좌표 취득
        left_wheel = -self.wheel_height_gap//2, -self.wheel_width_gap//2
        right_wheel = -self.wheel_height_gap//2, self.wheel_width_gap//2

        # 바퀴 좌표 회전
        affine_mtx = get_rotate_mtx(-self.yaw)
        rotated_left_wheel = np.dot(left_wheel, affine_mtx)
        rotated_right_wheel = np.dot(right_wheel, affine_mtx)

        return (
            rotated_border + rotated_left_wheel + self.position,
            rotated_border + rotated_right_wheel + self.position)
    

    def get_ultrasonic_distance(self, env, ratio=None):
        x, y = self.position

        distances = []
        start_end_pos = []

        for (ultra_x, ultra_y), ultra_yaw in self.get_ultrasonic_pos_with_yaw():
            here_x, here_y = ultra_x, ultra_y
            gradient = np.tan(ultra_yaw)
            
            if -np.pi/2 < ultra_yaw < np.pi/2:
                sign = 0.7
            else:
                sign = -0.7

            if np.abs(gradient) >= 100:
                gradient = np.sign(gradient) * 100
                sign = np.sign(sign) * 0.005

            here_x, here_y = ultra_x, ultra_y
            while True:
                round_x = np.rint(here_x).astype(np.int16)
                round_y = np.rint(here_y).astype(np.int16)
                
                if not env.in_range((round_x, round_y)):
                    break

                next_x = here_x + sign
                next_y = gradient * (next_x - ultra_x) + ultra_y

                next_round_x = np.rint(next_x).astype(np.int16)
                next_round_y = np.rint(next_y).astype(np.int16)

                if env.in_range((next_round_x, next_round_y))  \
                and env.MAP[next_round_y, next_round_x] == env.WALL:
                    break

                here_x, here_y = next_x, next_y
            
            dist = np.sqrt((here_x - ultra_x)**2 + (here_y - ultra_y)**2)

            distances.append(dist)
            start_end_pos.append(((ultra_x, ultra_y), (here_x, here_y)))
        
        distances = np.rint(np.array(distances) / 2)

        if ratio:
            distances = np.rint(distances * ratio)
            
        distances = np.clip(distances, 0, 200)

        # 초음파 무작위 랜덤 값 적용
        if np.random.sample() < 0.3:
            rand_idx = np.random.choice(len(distances), size=np.rint(len(distances)/3.0).astype(np.int8), replace=False)
            distances[rand_idx] += np.rint(np.random.normal(loc=0, scale=20, size=rand_idx.size))
            distances = np.clip(distances, 0, 200)
        
        return distances, start_end_pos

