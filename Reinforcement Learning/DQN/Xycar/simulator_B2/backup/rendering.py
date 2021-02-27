#!/usr/bin/env python

import pygame
import numpy as np
from PIL import Image
from calc import *

class render:
    def __init__(self):
        self.rendering = False
        self.screen = None
        self.clock = None
        self.keyboard = None
        self.event = None
        self.map = None
        self.map_info = None
        self.exit = False

    def set_carImg(self, image_path, image_pathv):
        self.colorImage = Image.open(image_path)
        self.colorvImage = Image.open(image_pathv)
        w, h = self.colorImage.size
        half_w, half_h = float(w)/2.0, float(h)/2.0
        self.car_x_ori = [-half_w, -half_w,  half_w, half_w]
        self.car_y_ori = [-half_h,  half_h, -half_h, half_h]
        return self.colorImage, self.car_x_ori, self.car_y_ori

    def pygame_init(self):
        pygame.init()
        self.rendering = True

    def pygame_display_set_caption(self, title):
        pygame.display.set_caption(title)

    def pygame_screen(self, width, height):
        self.screen = pygame.display.set_mode([width, height])
    
    def pygame_clock(self):
        self.clock = pygame.time.Clock()
    
    def pygame_display_flip(self, fps):
        pygame.display.flip()
        self.clock.tick(fps)

    def pygame_set_event(self):
        self.event = pygame.event.get()

    def pygame_set_key_event(self):
        return pygame.key.get_pressed()

    def get_pygame_up_arrow_key(self):
        return pygame.K_UP

    def get_pygame_down_arrow_key(self):
        return pygame.K_DOWN

    def get_pygame_left_arrow_key(self):
        return pygame.K_LEFT

    def get_pygame_right_arrow_key(self):
        return pygame.K_RIGHT

    def get_pygame_spacebar_key(self):
        return pygame.K_SPACE

    def calculate_dt(self):
        if self.rendering:
            return float(self.clock.get_time()) / 1000.0
        else:
            return float(0.06)

    def pygame_exit_check(self):
        if self.event == None:
            return
        for e in self.event:
            if e.type == pygame.QUIT:
                self.exit = True

    def get_game_exit(self):
        return self.exit

    def PIL2PygameIMG(self, image, angle, center_x, center_y):
        IMG = image.rotate(angle, expand=True, center=(center_x, center_y), fillcolor=(0, 0, 0, 0))
        return pygame.image.fromstring(IMG.tobytes("raw", 'RGBA'), IMG.size, 'RGBA'), IMG
        
    def draw_background(self):
        self.screen.fill((255, 255, 255))

    def draw_map(self, map):
        self.screen.blit(map, [0, 0])

    def draw_wheel(self, x, y, w, h, angle):
        half_w = float(w) / 2.0
        half_h = float(h) / 2.0

        wheel_img_x, wheel_img_y = [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]
        wheel_x_ori = [-half_w, -half_w, half_w, half_w]
        wheel_y_ori = [-half_h,  half_h, -half_h, half_h]

        theta = np.radians(-angle)

        for cnt in range(4):
            wheel_img_x[cnt], wheel_img_y[cnt] = rotation_matrix(wheel_x_ori[cnt], wheel_y_ori[cnt], x, y, theta, float)

        X = int(round(min(wheel_img_x)))
        Y = int(round(min(wheel_img_y)))
        
        wheel_img = self.make_wheel(w, h, angle)

        self.screen.blit(wheel_img, [X, Y])

    def make_wheel(self, w, h, angle):
        image = Image.new("RGBA", (int(round(w)), int(round(h))), (0, 0, 0, 255))
        IMG = image.rotate(angle, expand=True, center=(int(round(w/2)), int(round(h/2))), fillcolor=(0, 0, 0, 0))
        return pygame.image.fromstring(IMG.tobytes("raw", 'RGBA'), IMG.size, 'RGBA')

    def draw_car(self, x, y, yaw):
        w, h = 162, 93

        half_w = float(w) / 2.0
        half_h = float(h) / 2.0

        theta = np.radians(-yaw)
        xycar_img, _ = self.PIL2PygameIMG(self.colorImage, yaw, int(round(half_w)), int(round(half_h)))

        self.screen.blit(xycar_img, [x, y])
        
        # car_img_x, car_img_y = [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]

        # half_w = float(w) / 2.0
        # half_h = float(h) / 2.0

        # theta = np.radians(-yaw)
        # car_x_ori = [-half_w, -half_w, half_w, half_w]
        # car_y_ori = [-half_h, half_h, -half_h, half_h]
        
        # for cnt in range(4):
        #     car_img_x[cnt], car_img_y[cnt] = rotation_matrix(car_x_ori[cnt], car_y_ori[cnt], float(x), float(y), theta, float)

        # X = int(round(min(car_img_x)))
        # Y = int(round(min(car_img_y)))

        # xycar_img, _ = self.PIL2PygameIMG(self.colorImage, yaw, int(round(half_w)), int(round(half_h)))
        # self.screen.blit(xycar_img, [X, Y])

    def rtn_car(self, x, y, yaw):
        w, h = 162, 93
        
        car_img_x, car_img_y = [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]

        half_w = float(w) / 2.0
        half_h = float(h) / 2.0

        theta = np.radians(-yaw)
        car_x_ori = [-half_w, -half_w, half_w, half_w]
        car_y_ori = [-half_h, half_h, -half_h, half_h]
        
        for cnt in range(4):
            car_img_x[cnt], car_img_y[cnt] = rotation_matrix(car_x_ori[cnt], car_y_ori[cnt], float(x), float(y), theta, float)

        X = int(round(min(car_img_x)))
        Y = int(round(min(car_img_y)))

        img = self.colorvImage.rotate(yaw, expand=True, center=(int(round(half_w)), int(round(half_h))), fillcolor=(255, 255, 255))
        #return pygame.image.fromstring(IMG.tobytes("raw", 'RGBA'), IMG.size, 'RGBA'), IMG

        #_, img = self.PIL2PygameIMG(self.colorvImage, yaw, int(round(half_w)), int(round(half_h)))
        
        return X, Y, img



if __name__ == "__main__":
    R = render()
    R.set_carImg("image/car.png", "image/car.jpg")

    from map import map
    M = map()

    png = "maps/main_map.png"
    yml = "maps/main_map.yaml"
    w, h, conf = M.map_read(png, yml)
    xymap = M.get_map()

    from car import car
    xycar = car()
    random_init = r = False
    xymap_info = M.get_map_info()
    xycar.set_random(r)
    xycar.set_init_location([100], [100], random_init)
    xycar.set_init_yaw([0], random_init)
    xycar.set_init_detect_device_us()

    xycar.set_screen_size(w, h)
    M.set_map_obstacle()


    xycar.set_velocity(1000)

    R.pygame_init()
    R.pygame_display_set_caption("simulator")

    R.pygame_screen(w, h)
    screen = R.screen

    xycar.set_screen(R.screen)
    
    R.pygame_clock()
    clock = R.clock
    
    i = 0

    xycar.set_restart()
    i = 0
    delta = 1
    while not R.exit:
        R.draw_background()
        R.draw_map(xymap)

        x, y = xycar.get_location()

        dt = clock.get_time() / 1000.0
        xycar.set_gear("D", 100)
        xycar.set_accel_padal(True)
        xycar.set_steering_angle(i + delta)
        xycar.update(R.calculate_dt())

        R.pygame_exit_check()
        R.draw_car(int(x), int(y), xycar.get_yaw())

        mx, my, mw, mh, mangle = xycar.get_left_front_wheel()
        R.draw_wheel(mx, my, mw, mh, mangle)
        mx, my, mw, mh, mangle = xycar.get_left_back_wheel()
        R.draw_wheel(mx, my, mw, mh, mangle)
        mx, my, mw, mh, mangle = xycar.get_right_front_wheel()
        R.draw_wheel(mx, my, mw, mh, mangle)
        mx, my, mw, mh, mangle = xycar.get_right_back_wheel()
        R.draw_wheel(mx, my, mw, mh, mangle)

        R.pygame_display_flip(clock.get_time())

        x, y = xycar.get_location()
        i += delta
        if i < -15 or i > 15:
            delta *= -1
        
