# -*- coding: utf-8 -*-

import os
import numpy as np
import cv2 as cv

from simulator.car import Car
from simulator.utils import *

class Simulator(object):
    """
    map_type: "label", "image"
    """
    def __init__(self, map="rally_map2.png", fps=10):
        self.script_dir = os.path.dirname(__file__) 

        # 맵 정보 불러오기
        map_path = os.path.join(self.script_dir, "map", map)
        self.map = cv.imread(map_path)
        
        self.car = Car()
        self.fps = fps

        self.DRIVE = self.car.DRIVE
        self.REVERSE = self.car.REVERSE
        self.BREAK = self.car.BREAK

        self.max_steering_deg = self.car.max_steering_deg

        self.is_done = False


    def render(self):
        background = self.map.copy()
        draw_car(background, self.car)
        draw_ultrasonic(background, self.car, self.map)
        cv.imshow("Simulator", background)
        cv.waitKey(int(1000/self.fps))


    def step(self, gear, steering_deg):
        self.car.update(1/self.fps, gear, steering_deg)

        distances_meter = self._get_ultrasonics_distances()
        if is_collision(self.map, self.car) or np.min(distances_meter) <= 10:
            self.is_done = True
        else:
            self.is_done = False

        return distances_meter, self.is_done


    def reset(self):
        self.is_done = False
        self.car.reset()
        self._set_random_start_pos()
        distances_meter = self._get_ultrasonics_distances()
        return distances_meter, self.is_done


    def _set_random_start_pos(self):
        map_height, map_width = self.map.shape[:2]

        while True:
            self.car.position = (np.random.randint(map_height), np.random.randint(map_width))
            self.car.yaw = np.random.uniform(-np.pi, np.pi)
            
            distances_meter = self._get_ultrasonics_distances()

            if not (is_collision(self.map, self.car) or np.min(distances_meter) <= 15):
                break


    def _get_ultrasonics_distances(self):
        ultrasonic_start_points, ultrasonic_end_points, _ = get_ultrasonic_distance(self.map, self.car)
        distances = np.sqrt(np.sum((ultrasonic_start_points - ultrasonic_end_points)**2, axis=1))
        distances_meter = rint(distances * self.car.meter_per_pixel * 100)
        return distances_meter

