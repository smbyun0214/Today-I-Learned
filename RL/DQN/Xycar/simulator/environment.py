import os
import pygame
import numpy as np
import cv2 as cv


def get_rotate_mtx(yaw):
    return np.array([
        [np.cos(yaw), np.sin(yaw)],
        [-np.sin(yaw), np.cos(yaw)]])


class Environment(object):
    """
    map_type: "label", "image"
    """
    def __init__(self, map, map_type="label", pixel=1, fps=10):
        self.script_dir = os.path.dirname(__file__) 

        # 맵 불러오기
        if map_type == "label":
            self._load_map_label(map)
        else:
            self._load_map_image(map)
    
        # 맵의 크기
        self.WIDTH = len(self.RAW_MAP[0]) * pixel
        self.HEIGHT = len(self.RAW_MAP) * pixel

        # 한 블록당 설정된 픽셀
        self.PIXEL = pixel

        # 맵 정보 가공
        self.WALL = 0
        self.ROAD = 1
        self.GOAL = 2

        self.goal_pos = []
        self.MAP = np.zeros((self.HEIGHT, self.WIDTH), dtype=np.int8)
        for y, row in enumerate(self.RAW_MAP):
            for x, val in enumerate(row):
                if val == "#":
                    self.MAP[y*self.PIXEL:(y+1)*self.PIXEL, x*self.PIXEL:(x+1)*self.PIXEL] = self.WALL
                elif val == "G":
                    self.MAP[y*self.PIXEL:(y+1)*self.PIXEL, x*self.PIXEL:(x+1)*self.PIXEL] = self.GOAL
                else:
                    self.MAP[y*self.PIXEL:(y+1)*self.PIXEL, x*self.PIXEL:(x+1)*self.PIXEL] = self.ROAD

        # Xycar 이미지 불러오기
        xycar_img_path = os.path.join(self.script_dir, "resources", "xycar.png")
        self.xycar_surface = pygame.image.load(xycar_img_path)

        # 색상 설정
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.RED = (255, 0, 0)
        self.GREEN = (0, 255, 0)
        self.BLUE = (0, 0, 255)

        # Pygame 초기화 및 화면 설정
        self.screen = None

        # 에피소드가 진행 여부
        self.is_done = False

        # Pygame FPS 설정
        self.fps = fps
        self.clock = pygame.time.Clock()
        self.tick_ms = self.clock.tick(self.fps)
        self.dt = 1 / self.fps

        # 이미지 저장을 위한 맵 정보
        self.MAP_3C = np.zeros((self.HEIGHT, self.WIDTH, 3), dtype=np.uint8)
        for y, row in enumerate(self.RAW_MAP):
            for x, val in enumerate(row):
                if val == "#":
                    self.MAP_3C[y*self.PIXEL:(y+1)*self.PIXEL, x*self.PIXEL:(x+1)*self.PIXEL] = [0, 0, 0]
                else:
                    self.MAP_3C[y*self.PIXEL:(y+1)*self.PIXEL, x*self.PIXEL:(x+1)*self.PIXEL] = [255, 255, 255]

        # 이미지를 저장하기 위한 차량 위치 정보
        self.save_data = []


    def _load_map_label(self, map):
        # 맵 정보 불러오기
        map_path = os.path.join(self.script_dir, "map", map)
        self.RAW_MAP = []
        with open(map_path, "rb") as f:
            for row in f:
                self.RAW_MAP.append(row.strip())


    def _load_map_image(self, map):
        # 맵 정보 불러오기
        map_path = os.path.join(self.script_dir, "map", map)
        image = cv.imread(map_path, cv.IMREAD_GRAYSCALE)
        _, binary = cv.threshold(image, 200, 255, cv.THRESH_BINARY)
        # binary = cv.morphologyEx(binary, cv.MORPH_ERODE, cv.getStructuringElement(cv.MORPH_RECT, (3, 3)), iterations=10)

        # height, width = binary.shape[:2]
        # binary = cv.resize(binary, (width*2//3, height*2//3))

        self.RAW_MAP = []
        for row in binary:
            raw_map_row = [ "#" if val == 0 else " " for val in row ]
            self.RAW_MAP.append(raw_map_row)
    
    def status(self, car):
        border_pos = car.get_border_pos()
        border_pos = np.rint(border_pos).astype(np.int16)

        is_goal = False

        for pos in border_pos:
            x, y = pos

            if not self.in_range(pos)    \
            or self.MAP[y, x] == self.WALL:
                return self.WALL
            elif self.MAP[y, x] == self.GOAL:
                is_goal = True
        
        if is_goal:
            return self.GOAL

        return self.ROAD


    def in_range(self, pos):
        x, y = pos
        return 0 <= x < self.WIDTH and 0 <= y < self.HEIGHT


    def get_random_pos_and_yaw(self, car):
        car.velocity = 0
        car.acceleration = 0

        # 무작위로 선택된 위치의 차량 설정
        while True:
            random_x = np.random.uniform(0, self.WIDTH)
            random_y = np.random.uniform(0, self.HEIGHT)
            random_yaw = np.random.uniform(0, 2*np.pi)

            car.position = (random_x, random_y)
            car.yaw = random_yaw

            if self.status(car) == self.ROAD:
                break
        
        return (random_x, random_y), random_yaw


    def save_pos_and_image(self, car, distances, init=False):
        if init:
            self.save_data = []
            
        rotated_surf = pygame.transform.rotate(self.xycar_surface, np.degrees(car.yaw))

        delta_x = (self.xycar_surface.get_width() - rotated_surf.get_width()) // 2
        delta_y = (self.xycar_surface.get_height() - rotated_surf.get_height()) // 2

        center_x, center_y = car.position
        x = center_x - self.xycar_surface.get_width()//2
        y = center_y - self.xycar_surface.get_height()//2

        rotated_x = np.rint(x + delta_x).astype(np.int16)
        rotated_y = np.rint(y + delta_y).astype(np.int16)
        rotated_pos = (rotated_x, rotated_y)
        self.save_data.append((rotated_pos, distances, pygame.surfarray.array3d(rotated_surf).swapaxes(0, 1)[:,:,::-1]))


    def save_video(self, episode, dir="video"):
        video_dir = os.path.join(self.script_dir, dir)
        if not os.path.isdir(video_dir):
            os.mkdir(video_dir)
        video_path = os.path.join(video_dir, "{:05}.avi".format(episode))
        

        print("[VIDEO SAVE] {:05}.avi...".format(episode))

        fourcc = cv.VideoWriter_fourcc(*"mp4v")
        writer = cv.VideoWriter(video_path, fourcc, self.fps, (self.WIDTH, self.HEIGHT))
        
        for (x, y), distances, car_img in self.save_data:
            img_height, img_width = car_img.shape[:2]

            frame = np.copy(self.MAP_3C)

            cand_roi = frame[y:y+img_height, x:x+img_width]
            cand_roi_height, cand_roi_width = cand_roi.shape[:2]

            img_height, img_width = min(cand_roi_height, img_height), min(cand_roi_width, img_width)
            roi = frame[y:y+img_height, x:x+img_width]
            car_img = car_img[:img_height, :img_width]

            if car_img.size == 0:
                continue

            gray = cv.cvtColor(car_img, cv.COLOR_BGR2GRAY)
            _, mask = cv.threshold(gray, 200, 255, cv.THRESH_BINARY_INV)
            mask_inv = cv.bitwise_not(mask)

            roi_bg = cv.bitwise_and(roi, roi, mask=mask_inv)
            img_fg = cv.bitwise_and(car_img, car_img, mask=mask)

            dst = cv.add(roi_bg, img_fg)
            frame[y:y+img_height, x:x+img_width] = dst
            cv.putText(frame, "Episode: {:05}".format(episode), (10, 100), cv.FONT_HERSHEY_DUPLEX, 1, (255, 255, 0), 2)
            
            round_dists = np.round(distances, 1)
            cv.putText(frame, "Distances: {}".format(round_dists), (10, 200), cv.FONT_HERSHEY_DUPLEX, 1, (255, 0, 255), 2)
            
            cv.putText(frame, "Pose: {}, {}".format(x, y), (10, 300), cv.FONT_HERSHEY_DUPLEX, 1, (255, 0, 255), 2)
            
            writer.write(frame)

        writer.release()
    
        print("[VIDEO SAVE DONE]".format(episode))
        self.save_data = []


    def draw_map(self):
        for y, row in enumerate(self.RAW_MAP):
            for x, val in enumerate(row):
                if val == "#":
                    pygame.draw.rect(self.screen, self.BLACK, (x*self.PIXEL, y*self.PIXEL, self.PIXEL, self.PIXEL))
                else:
                    pygame.draw.rect(self.screen, self.WHITE, (x*self.PIXEL, y*self.PIXEL, self.PIXEL, self.PIXEL))


    def draw_background(self):
        self.screen.fill(self.BLACK)


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


    def render(self, car, draw_all=False):
        if self.screen is None:
            num_pass, num_fail = pygame.init()
            self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))

        self.tick_ms = self.clock.tick(self.fps)
        self.draw_background()
        self.draw_map()
        self.draw_car_wheel(car)
        self.draw_car(car)

        if draw_all:
            self.draw_car_border(car)
            self.draw_car_wheel_border(car)
            self.draw_ultrasonic(car)
            pygame.draw.circle(self.screen, self.GREEN, car.position, 5)
        pygame.display.flip()



if __name__ == "__main__":
    from agent import Car

    env = Environment("rally_map.png", map_type="image", fps=30)
    car = Car()
    car.position = (100, 120)

    i = 0
    delta = -1

    # start_pos = env.get_start_pos(car)

    episode = 0
    while not env.is_done:
        is_collision = False

        # random_pos, random_yaw = env.get_random_pos_and_yaw(car)
        # car.position = random_pos
        # car.yaw = random_yaw
        car.position = (130, 428)
        car.yaw = np.radians(-135)

        episode += 1
        
        obs, _ = car.get_ultrasonic_distance(env)
        env.save_pos_and_image(car, obs)

        while not is_collision and not env.is_done:
            # env.render(car)

            is_collision = env.status(car) == env.WALL
            # print(car.get_ultrasonic_distance(env)[0])

            # for event in pygame.event.get():
            #     # X를 눌렀으면, 게임을 종료
            #     if event.type == pygame.QUIT:
            #         env.is_done = True
            #     if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
            #         is_collision = True

            # pressed = pygame.key.get_pressed()
            # if pressed[pygame.K_LEFT]:
            #     angle += 5
            # elif pressed[pygame.K_RIGHT]:
            #     angle -= 5
            # if pressed[pygame.K_UP]:
            #     drive = car.DRIVE
            # elif pressed[pygame.K_DOWN]:
            #     drive = car.REVERSE
            # elif pressed[pygame.K_s]:
            #     drive = car.NEUTRAL
            #     car.velocity = 0
            
            car.update(car.DRIVE, i, env.dt)
            obs, _ = car.get_ultrasonic_distance(env)
            env.save_pos_and_image(car, obs)
            
            # car.update(car.NEUTRAL, i, env.dt)
            # print(car.velocity, car.acceleration, env.is_done, env.dt)
            i += delta
            if i <= -15 or 15 <= i:
                delta *= -1
        
        env.save_video(episode)
