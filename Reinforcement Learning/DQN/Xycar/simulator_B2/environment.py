import os
import pygame
import numpy as np


def get_rotate_mtx(yaw):
    return np.array([
        [np.cos(yaw), np.sin(yaw)],
        [-np.sin(yaw), np.cos(yaw)]])


class Environment(object):
    def __init__(self, map, pixel=50, fps=10):
        # 맵 정보 불러오기
        script_dir = os.path.dirname(__file__) 
        map_path = os.path.join(script_dir, "map", map)
        self.RAW_MAP = []
        with open(map_path) as f:
            for row in f:
                self.RAW_MAP.append(row.strip())
    
        # 맵의 크기
        self.WIDTH = len(self.RAW_MAP[0]) * pixel
        self.HEIGHT = len(self.RAW_MAP) * pixel

        # 한 블록당 설정된 픽셀
        self.PIXEL = pixel

        # 맵 정보 가공
        self.WALL = True
        self.ROAD = False
        self.MAP = np.zeros((self.HEIGHT, self.WIDTH), dtype=bool)
        for y, row in enumerate(self.RAW_MAP):
            for x, val in enumerate(row):
                if val == "#":
                    self.MAP[y*self.PIXEL:(y+1)*self.PIXEL, x*self.PIXEL:(x+1)*self.PIXEL] = self.WALL
                else:
                    self.MAP[y*self.PIXEL:(y+1)*self.PIXEL, x*self.PIXEL:(x+1)*self.PIXEL] = self.ROAD


        # Xycar 이미지 불러오기
        xycar_img_path = os.path.join(script_dir, "resources", "xycar.png")
        self.xycar_surface = pygame.image.load(xycar_img_path)

        # 색상 설정
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.RED = (255, 0, 0)
        self.BLUE = (0, 255, 0)
        self.GREEN = (0, 0, 255)

        # Pygame 초기화 및 화면 설정
        num_pass, num_fail = pygame.init()
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))

        # 에피소드가 진행 여부
        self.is_done = False

        # Pygame FPS 설정
        self.fps = fps
        self.clock = pygame.time.Clock()
        self.tick_ms = self.clock.tick(self.fps)


    def draw_map(self):
        for y, row in enumerate(self.RAW_MAP):
            for x, val in enumerate(row):
                if val == "#":
                    pygame.draw.rect(self.screen, self.BLACK, (x*self.PIXEL, y*self.PIXEL, self.PIXEL, self.PIXEL))
                else:
                    pygame.draw.rect(self.screen, self.WHITE, (x*self.PIXEL, y*self.PIXEL, self.PIXEL, self.PIXEL))


    def draw_background(self):
        self.screen.fill(self.BLACK)


    def get_dt(self):
        return 1 / self.fps


    def draw_car(self, car):
        rotated_surf = pygame.transform.rotate(self.xycar_surface, np.degrees(car.yaw))

        delta_x = (self.xycar_surface.get_width() - rotated_surf.get_width()) // 2
        delta_y = (self.xycar_surface.get_height() - rotated_surf.get_height()) // 2

        center_x, center_y = car.position
        x = center_x - self.xycar_surface.get_width()//2
        y = center_y - self.xycar_surface.get_height()//2

        rotated_pos = (x + delta_x, y + delta_y)

        self.screen.blit(rotated_surf, rotated_pos)

    
    def draw_car_wheel(self, car):
        surface = pygame.Surface((car.wheel_width, car.wheel_height), pygame.SRCALPHA)
        surface.fill(self.BLACK)

        car_center_x, car_center_y = car.position

        """
        뒷 바퀴 그리기
        """
        mtx = get_rotate_mtx(-car.yaw)

        rotated_surf = pygame.transform.rotate(surface, np.degrees(car.yaw))

        delta_x = (surface.get_width() - rotated_surf.get_width()) // 2
        delta_y = (surface.get_height() - rotated_surf.get_height()) // 2

        # 왼쪽 바퀴 좌표 계산
        left_center_x = -car.wheel_height_gap // 2
        left_center_y = -car.wheel_width_gap // 2

        # 왼쪽 바퀴 회전
        rotated_left_x, rotated_left_y = np.dot([left_center_x, left_center_y], mtx)

        left_x = rotated_left_x - surface.get_width()//2 + car_center_x
        left_y = rotated_left_y - surface.get_height()//2 + car_center_y

        # 왼쪽 바퀴 그리기
        rotated_left_pos = (left_x + delta_x, left_y + delta_y)
        self.screen.blit(rotated_surf, rotated_left_pos)

        # 오른쪽 바퀴 좌표 계산
        right_center_x = -car.wheel_height_gap // 2
        right_center_y = +car.wheel_width_gap // 2

        # 오른쪽 바퀴 회전
        rotated_right_x, rotated_right_y = np.dot([right_center_x, right_center_y], mtx)

        right_x = rotated_right_x - surface.get_width()//2 + car_center_x
        right_y = rotated_right_y - surface.get_height()//2 + car_center_y

        # 오른쪽 바퀴 그리기
        rotated_right_pos = (right_x + delta_x, right_y + delta_y)
        self.screen.blit(rotated_surf, rotated_right_pos)


        """
        앞 바퀴 그리기
        """
        mtx = get_rotate_mtx(-car.yaw)

        steering_rad = np.radians(car.steering_deg)
        rotated_surf = pygame.transform.rotate(surface, np.degrees(car.yaw+steering_rad))

        delta_x = (surface.get_width() - rotated_surf.get_width()) // 2
        delta_y = (surface.get_height() - rotated_surf.get_height()) // 2

        # 왼쪽 바퀴 좌표 계산
        left_center_x = +car.wheel_height_gap // 2
        left_center_y = -car.wheel_width_gap // 2

        # 왼쪽 바퀴 회전
        rotated_left_x, rotated_left_y = np.dot([left_center_x, left_center_y], mtx)

        left_x = rotated_left_x - surface.get_width()//2 + car_center_x
        left_y = rotated_left_y - surface.get_height()//2 + car_center_y

        # 왼쪽 바퀴 그리기
        rotated_left_pos = (left_x + delta_x, left_y + delta_y)
        self.screen.blit(rotated_surf, rotated_left_pos)

        # 오른쪽 바퀴 좌표 계산
        right_center_x = +car.wheel_height_gap // 2
        right_center_y = +car.wheel_width_gap // 2

        # 오른쪽 바퀴 회전
        rotated_right_x, rotated_right_y = np.dot([right_center_x, right_center_y], mtx)

        right_x = rotated_right_x - surface.get_width()//2 + car_center_x
        right_y = rotated_right_y - surface.get_height()//2 + car_center_y

        # 오른쪽 바퀴 그리기
        rotated_right_pos = (right_x + delta_x, right_y + delta_y)
        self.screen.blit(rotated_surf, rotated_right_pos)


    def draw_car_border(self, car):
        for pos in car.get_border_pos():
            pygame.draw.circle(self.screen, self.GREEN, pos, 1)

    
    def draw_car_wheel_border(self, car):
        # 앞 바퀴 그리기
        left_pos, right_pos = car.get_front_wheel_border_pos()
        for pos in left_pos:
            pygame.draw.circle(self.screen, self.BLUE, pos, 1)
        for pos in right_pos:
            pygame.draw.circle(self.screen, self.BLUE, pos, 1)

        # 뒷 바퀴 그리기
        left_pos, right_pos = car.get_back_wheel_border_pos()
        for pos in left_pos:
            pygame.draw.circle(self.screen, self.BLUE, pos, 1)
        for pos in right_pos:
            pygame.draw.circle(self.screen, self.BLUE, pos, 1)


    def is_collision(self, car):
        border_pos = car.get_border_pos()
        border_pos = np.rint(border_pos).astype(np.int16)
        for pos in border_pos:
            x, y = pos
            if not self.in_range(pos)    \
            or self.MAP[y, x] == self.WALL:
                return True
        return False


    def in_range(self, pos):
        x, y = pos
        return 0 <= x < self.WIDTH and 0 <= y < self.HEIGHT


    def get_start_pos(self, car):
        # scan_dy, scan_dx 설정
        scan_dist = np.sqrt(car.border_front**2 + car.border_left**2)
        scan_dist = int(scan_dist)
        scan_dxs = [
            car.border_front, car.border_front, -car.border_back, -car.border_back,
            -car.border_left, car.border_right, -car.border_left, car.border_right]
        scan_dys = [
            -car.border_left, car.border_right, -car.border_left, car.border_right,
            car.border_front, car.border_front, -car.border_back, -car.border_back]

        # 탐색을 시작하기 가능한 좌표인지 확인
        def is_valid_pos(pos):
            x, y = pos
            if not self.in_range(pos) or self.MAP[y, x] == self.WALL:
                return False

            for scan_dy, scan_dx in zip(scan_dys, scan_dxs):
                next_scan_y, next_scan_x = y + scan_dy, x + scan_dx
                next_scan_pos = (next_scan_x, next_scan_y)

                if not self.in_range(next_scan_pos)  \
                or self.MAP[next_scan_y, next_scan_x] == self.WALL:
                    return False
            return True

        # 방문 좌표, 후보 좌표 리스트
        cand_pos = []

        for (y, x), _ in np.ndenumerate(self.MAP):
            pos = (x, y)
            if is_valid_pos(pos):
                cand_pos.append(pos)
                        
        return cand_pos


    def render(self, car):
        self.tick_ms = self.clock.tick(self.fps)
        self.draw_background()
        self.draw_map()
        self.draw_car(car)
        self.draw_car_wheel(car)
        # self.draw_car_border(car)
        self.draw_car_wheel_border(car)
        pygame.draw.circle(self.screen, self.BLACK, car.position, 5)
        pygame.display.flip()


    # def play(self):
    #     isPlay = True
    #     while isPlay:
    #         self.draw_background()
    #         self.draw_map()
    #         self.draw_car(self.cars)

    #         pygame.display.flip()

    #         for event in pygame.event.get():
    #             # X를 눌렀으면, 게임을 종료
    #             if event.type == pygame.QUIT:
    #                 isPlay = False

    #     # 게임종료한다.
        # pygame.quit()
    

if __name__ == "__main__":
    from agent import Car

    env = Environment("snake", fps=30)
    car = Car()
    car.position = (100, 120)

    i = 0
    delta = -1

    # start_pos = env.get_start_pos(car)

    while not env.is_done:
        is_collision = False

        # 무작위로 선택된 위치의 차량 설정
        while True:
            random_x = np.random.uniform(0, env.WIDTH)
            random_y = np.random.uniform(0, env.HEIGHT)
            random_yaw = np.random.uniform(0, np.pi)

            car.position = (random_x, random_y)
            car.yaw = random_yaw
            car.velocity = 0
            car.acceleration = 0

            if not env.is_collision(car):
                break


        while not is_collision and not env.is_done:
            env.render(car)

            is_collision = env.is_collision(car)

            for event in pygame.event.get():
                # X를 눌렀으면, 게임을 종료
                if event.type == pygame.QUIT:
                    env.is_done = True
                if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                    is_collision = True
            

            car.update(car.DRIVE, i, env.get_dt())
            # car.update(car.NEUTRAL, i, env.get_dt())
            print(car.velocity, car.acceleration, env.is_done, env.get_dt())
            i += delta
            if i <= -15 or 15 <= i:
                delta *= -1
