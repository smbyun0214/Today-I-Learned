import pygame
import numpy as np


def get_rotate_mtx(yaw):
    return np.array([
        [np.cos(yaw), np.sin(yaw)],
        [-np.sin(yaw), np.cos(yaw)]])


class Environment(object):
    def __init__(self, map, pixel=50, fps=10):
        # 맵 정보 불러오기
        map_path = "./map/" + map
        self.RAW_MAP = []
        with open(map_path) as f:
            for row in f:
                self.RAW_MAP.append(row.strip())
    
        # 맵의 크기
        self.WIDTH = len(self.RAW_MAP[0])
        self.HEIGHT = len(self.RAW_MAP)

        # 한 블록당 설정된 픽셀
        self.PIXEL = pixel

        # 맵 정보 가공
        self.WALL = True
        self.ROAD = False
        self.MAP = np.zeros((self.PIXEL*self.HEIGHT, self.PIXEL*self.WIDTH), dtype=bool)
        for y, row in enumerate(self.RAW_MAP):
            for x, val in enumerate(row):
                if val == "#":
                    self.MAP[y*self.PIXEL:(y+1)*self.PIXEL, x*self.PIXEL:(x+1)*self.PIXEL] = self.WALL
                else:
                    self.MAP[y*self.PIXEL:(y+1)*self.PIXEL, x*self.PIXEL:(x+1)*self.PIXEL] = self.ROAD


        # Xycar 이미지 불러오기
        self.xycar_surface = pygame.image.load("./resources/xycar.png")

        # 색상 설정
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.RED = (255, 0, 0)
        self.BLUE = (0, 255, 0)
        self.GREEN = (0, 0, 255)

        # Pygame 초기화 및 화면 설정
        num_pass, num_fail = pygame.init()
        self.screen = pygame.display.set_mode((self.WIDTH*self.PIXEL, self.HEIGHT*self.PIXEL))

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
        self.screen.fill(self.WHITE)


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

        left_x = left_center_x - surface.get_width()//2
        left_y = left_center_y - surface.get_height()//2

        # 왼쪽 바퀴 회전
        rotated_left_x, rotated_left_y = np.dot([left_x, left_y], mtx)

        # 왼쪽 바퀴 그리기
        rotated_left_pos = (rotated_left_x + delta_x + car_center_x, rotated_left_y + delta_y + car_center_y)
        self.screen.blit(rotated_surf, rotated_left_pos)

        # 오른쪽 바퀴 좌표 계산
        right_center_x = -car.wheel_height_gap // 2
        right_center_y = +car.wheel_width_gap // 2

        right_x = right_center_x - surface.get_width()//2
        right_y = right_center_y - surface.get_height()//2

        # 오른쪽 바퀴 회전
        rotated_right_x, rotated_right_y = np.dot([right_x, right_y], mtx)

        # 오른쪽 바퀴 그리기
        rotated_right_pos = (rotated_right_x + delta_x + car_center_x, rotated_right_y + delta_y + car_center_y)
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

        left_x = left_center_x - surface.get_width()//2
        left_y = left_center_y - surface.get_height()//2

        # 왼쪽 바퀴 회전
        rotated_left_x, rotated_left_y = np.dot([left_x, left_y], mtx)

        # 왼쪽 바퀴 그리기
        rotated_left_pos = (rotated_left_x + delta_x + car_center_x, rotated_left_y + delta_y + car_center_y)
        self.screen.blit(rotated_surf, rotated_left_pos)

        # 오른쪽 바퀴 좌표 계산
        right_center_x = +car.wheel_height_gap // 2
        right_center_y = +car.wheel_width_gap // 2

        right_x = right_center_x - surface.get_width()//2
        right_y = right_center_y - surface.get_height()//2

        # 오른쪽 바퀴 회전
        rotated_right_x, rotated_right_y = np.dot([right_x, right_y], mtx)

        # 오른쪽 바퀴 그리기
        rotated_right_pos = (rotated_right_x + delta_x + car_center_x, rotated_right_y + delta_y + car_center_y)
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
        for x, y in border_pos:
            if self.MAP[y, x] == self.WALL:
                return True
        return False


    def get_dt(self):
        return self.tick_ms / 1000.0


    def render(self, car):
        self.tick_ms = self.clock.tick(self.fps)
        self.draw_background()
        self.draw_map()
        self.draw_car_wheel(car)
        self.draw_car(car)
        self.draw_car_border(car)
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
    while not env.is_done:
        env.render(car)

        env.is_collision(car)

        for event in pygame.event.get():
            # X를 눌렀으면, 게임을 종료
            if event.type == pygame.QUIT:
                env.is_done = True
        

        print(i)
        car.update(car.DRIVE, i, env.get_dt())
        # car.update(car.NEUTRAL, i, env.get_dt())
        i += delta
        if i <= -15 or 15 <= i:
            delta *= -1



