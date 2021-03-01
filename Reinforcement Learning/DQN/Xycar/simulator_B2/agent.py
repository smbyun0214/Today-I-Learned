import numpy as np


def get_roate_pos(pos, yaw):
    x, y = pos
    return (x * np.cos(yaw), y * np.sin(yaw))


def get_rotate_mtx(yaw):
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
    def __init__(self):
        # 현재 위치/방향/steering 정보
        self.position = (0, 0)  # center_x, center_y
        self.yaw = 0
        self.steering_deg = 0   # 좌/우 앞바퀴의 중점 기준

        # 현재 속도/가속도 정보
        self.velocity = 0.0
        self.acceleration = 0.0

        # 제한 속도/가속도/steering 정보
        self.max_velocity = 100.0
        self.max_acceleration = 50.0
        self.max_steering_deg = 30.0

        # 기어 정보
        self.DRIVE = 1
        self.REVERSE = 2
        self.NEUTRAL = 3

        # 기어를 선택할 때, 가속도 증가량
        self.delta_acc = 50

        # steering을 조절할 때, angle 증가량
        self.delta_steering_deg = 200

        # 바퀴 사이의 너비/높이
        #               left
        #     ==+==---------------==+==
        # back  |         x          |   front
        #     ==+==---------------==+==
        #               right
        self.wheel_width_gap = 56
        self.wheel_height_gap = 74

        # 바퀴 너비/높이
        self.wheel_width = 14
        self.wheel_height = 6

        # 차체의 테두리 간격
        self.border_back = 62
        self.border_front = 62
        self.border_left = 31
        self.border_right = 31


    # 차체 초음파 센서 정보
    # - return: (x, y, yaw)의 리스트
    def get_ultrasonic_pos_with_yaw(self):
        #   +---+---------U---------+---+
        #   | ==+==---------------==+== U
        #   |   |         x          |  U
        #   | ==+==---------------==+== U
        #   +---+---------U---------+---+
        front_gap = 10
        mtx = get_rotate_mtx(-self.yaw)

        ultrasonic_pos1 = np.dot([0, -self.border_left], mtx)
        ultrasonic_pos2 = np.dot([self.border_front, -self.border_left+front_gap], mtx)
        ultrasonic_pos3 = np.dot([self.border_front, 0], mtx)
        ultrasonic_pos4 = np.dot([self.border_front, self.border_right-front_gap], mtx)
        ultrasonic_pos5 = np.dot([0, self.border_right], mtx)

        return (
            (ultrasonic_pos1 + self.position, normalize_radian(np.radians(-90) - self.yaw)),
            (ultrasonic_pos2 + self.position, normalize_radian(np.radians(-30) - self.yaw)),
            (ultrasonic_pos3 + self.position, normalize_radian(np.radians(0) - self.yaw)),
            (ultrasonic_pos4 + self.position, normalize_radian(np.radians(30) - self.yaw)),
            (ultrasonic_pos5 + self.position, normalize_radian(np.radians(90) - self.yaw))
        )


    def get_front_wheel_pos(self):
        x, y = self.position
        left_pos = (x + self.wheel_height_gap//2, y - self.wheel_width_gap//2)
        right_pos = (x + self.wheel_height_gap//2, y - self.wheel_width_gap//2)

        return (get_roate_pos(left_pos, self.yaw), get_roate_pos(right_pos, self.yaw))
    

    def get_back_wheel_pos(self):
        x, y = self.position
        left_pos = (x - self.wheel_height_gap//2, y - self.wheel_width_gap//2)
        right_pos = (x - self.wheel_height_gap//2, y - self.wheel_width_gap//2)

        return (get_roate_pos(left_pos, self.yaw), get_roate_pos(right_pos, self.yaw))


    def update(self, gear, steering_deg, dt):
        # steering 계산
        # self.steering_deg = self.steering_deg + np.sign(steering_deg) * self.delta_steering_deg * dt
        self.steering_deg = max(-self.max_steering_deg, min(steering_deg, self.max_steering_deg))

        # 가속도 계산
        if gear == self.DRIVE:
            self.acceleration += self.delta_acc * dt
        elif gear == self.REVERSE:
            self.acceleration -= self.delta_acc * dt

        self.acceleration = max(-self.max_acceleration, min(self.acceleration, self.max_acceleration))

        # 속도 갱신
        self.velocity += self.acceleration * dt
        self.velocity = max(-self.max_velocity, min(self.velocity, self.max_velocity))

        # 각속도 계산
        # theta: steering_rad
        # sin(theta) = wheel_width_gap / radius
        # 각속도 w = velocity / radius
        #         = velicity * sin(theta) / wheel_width_gap
        theta = np.radians(self.steering_deg)
        angular_velocity = self.velocity * np.sin(theta) / self.wheel_width_gap

        # yaw 갱신
        self.yaw += angular_velocity * dt
        self.yaw = normalize_radian(self.yaw)

        # 현재 위치 갱신
        x, y = self.position
        x += self.velocity * dt * np.cos(-self.yaw)
        y += self.velocity * dt * np.sin(-self.yaw)

        self.position = (x, y)
    

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
    

    def get_ultrasonic_distance(self, env):
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
        
        return np.array(distances), start_end_pos
