#!/usr/bin/env python
#-*- coding: utf-8 -*-

import os, random, sys
import torch, dill
import pygame

from env.game import *
from env.model import *

class learning_xycar:
    env = None
    framepersec = 30
    learning_rate = 0
    racing_map = None
    racing_name = ""

    def __init__(self, md=True):
        self.set_frame(self.framepersec)
        self.curr_path = os.getcwd()
        self.pygame_exit = False
        if md:
            self.make_dir()
            
    def set_map(self, name):
        map_name = 'map.' + name
        self.racing_name = name
        self.racing_map = __import__('%s' %(map_name), globals(), locals())
        if name == "snake":
            self.racing_map = self.racing_map.snake
        elif name == "square":
            self.racing_map = self.racing_map.square

    def make_dir(self):
        if not os.path.isdir(self.curr_path+"/video/"):
            os.mkdir(self.curr_path+"/video/")
        if not os.path.isdir(self.curr_path+"/save/"):
            os.mkdir(self.curr_path+"/save/")

    def set_init_gameover(self):
        self.space_bar = False

    def key_event(self):
        pygame.event.get()
        pressed = pygame.key.get_pressed()
        if pressed[pygame.K_SPACE]:
            self.space_bar = True

    def get_max_score(self):
        return self.env.max_score

    def max_score_update(self, score):
        self.env.max_score = score

    def set_frame(self, fps):
        self.dt = 1.0 / float(fps)

    def set_hyperparam(self, param_dict):
        self.sensor_num = param_dict["sensor_num"]
        self.learning_rate = param_dict["learning_rate"]
        self.discount_factor = param_dict["discount_factor"]
        self.batch_size = param_dict["batch_size"]
        self.min_history = param_dict["min_history"]
        self.buffer_limit = param_dict["buffer_limit"]
        self.hidden_size = param_dict["hidden_size"]

        if self.min_history < self.batch_size:
            self.min_history = self.batch_size
            
    def set_your_number(self, number):
        self.stu_number = number

    def state_setup(self, setup_dict):
        self.env = Game(False, self.sensor_num+1, map_data=self.racing_map, fps=int(1.0/self.dt))
        self.car_sensor = setup_dict["car sensor"]
        self.car_yaw = setup_dict["car yaw"]
        self.car_position = setup_dict["car position"]
        self.car_steer = setup_dict["car steer"]

        input_node_cnt = 0

        if self.car_sensor:
            input_node_cnt += self.sensor_num
        if self.car_yaw:
            input_node_cnt += 1
        if self.car_position:
            input_node_cnt += 2
        if self.car_steer:
            input_node_cnt += 1

        self.input_size = input_node_cnt

    def ML_init(self, model_name):
        study_init(self.input_size, self.hidden_size, self.learning_rate, model_name)
        self.output_size = 3
    
    def Experience_replay_init(self):
        self.memory = ReplayBuffer(self.buffer_limit)

    def Experience_replay_memory_input(self, state, action, next_state, reward):
        done_mask = 0.0 if self.done else 1.0
        self.memory.put((state, action, reward, next_state, done_mask))

    def Experience_replay_close(self):
        self.memory.free()

    def episode_init(self):
        self.env.reset()
        self.done = False
        self.step_count = 1
        self.start_time = time.time()

        cnt = 0
        if self.car_sensor:
            cnt += self.sensor_num
        if self.car_yaw:
            cnt += 1
        if self.car_position:
            cnt += 2
        if self.car_steer:
            cnt += 1

        state = []
        for _ in range(cnt):
            state.append(0.0)

        return np.array(state)

    def pygame_exit_chk(self):
        for event in pygame.event.get():#
            if event.type == pygame.QUIT:
                self.pygame_exit = True

    def set_E_greedy_func(self, Egreedy):
        self.E_greedyFunc = Egreedy

    def get_action(self, state):
        return study_get_action(state, self.E_greedyFunc)

    def get_action_viewer(self, state):
        return study_get_action(state)

    def step_not_move(self):
        info, self.round_count, self.suc_code = self.env.step_not_move(self.dt)
        
        self.done = False
        if self.suc_code > 0:
            self.done = True
        
        return np.array(info["car_sensor"])

    def step(self, action):
        info, self.round_count, self.suc_code = self.env.step(action, self.dt)

        self.done = False
        if self.suc_code > 0:
            self.done = True

        self.distance_device = info["car_sensor"]
        self.xycar_position = info["car_position"]
        self.xycar_yaw = info["car_yaw"]
        self.xycar_steering = info["car_steer"]

        state = []
        if self.car_sensor:
            state += self.distance_device
        if self.car_yaw:
            state += [self.xycar_yaw]
        if self.car_position:
            state += self.xycar_position
        if self.car_steer:
            state += [self.xycar_steering]

        self.step_count += 1

        return np.array(state)

    def train(self, count):
        return study_train(count, self.batch_size, self.discount_factor, self.memory)

    def load_model(self, episode):
        study_model_load(episode)

    def lect_load_model(self, episode):
        study_model_load(episode)

    def set_hidden_size(self, hidden_size):
        self.hidden_size = hidden_size

    def pygame_init(self):
        os.environ['SDL_VIDEO_WINDOW_POS'] = '%i,%i' % (100, 100)
        os.environ['SDL_VIDEO_CENTERED'] = '0'

        pygame.init()
        pygame.display.set_caption("Simulator")

        self.screen = pygame.display.set_mode([self.racing_map.SCREEN_WIDTH, self.racing_map.SCREEN_HEIGHT])
        self.B = pygame.font.SysFont("Arial", 55, bold=False, italic=False) 
        self.F = pygame.font.SysFont("Arial", 20, bold=False, italic=False)    
        self.D = pygame.font.SysFont("Arial", 15, bold=False, italic=False)
        self.clock = pygame.time.Clock()

    def screen_init(self):
        obs = obs_make(self.racing_map)
        o = obs.get_list()
        self.env = Game(True, self.sensor_num+1, fps=int(1.0/self.dt), map_data=self.racing_map, Font_1=self.F, Font_2=self.D, pygame_screen=self.screen)
        self.env.map_input(obs.get_all_list(), o[0], o[1])

    def view_stud_code(self):
        map_name = self.racing_map.MAP_NAME
        if map_name == "square":
            view_number = self.B.render(str(self.stu_number), True, (28,0,0))
            self.screen.blit(view_number,(self.racing_map.SCREEN_WIDTH/2 - 120, self.racing_map.SCREEN_HEIGHT/2 - 30))
        elif map_name == "snake":
            view_number = self.B.render(str(self.stu_number), True, (255,255,255))
            self.screen.blit(view_number,(self.racing_map.SCREEN_WIDTH/2 - 115, self.racing_map.SCREEN_HEIGHT/2 + 180))
            self.screen.blit(view_number,(self.racing_map.SCREEN_WIDTH/2 - 660, self.racing_map.SCREEN_HEIGHT/2 - 270))
            self.screen.blit(view_number,(self.racing_map.SCREEN_WIDTH/2 + 430, self.racing_map.SCREEN_HEIGHT/2 - 270))

    def display_flip(self):
        pygame.display.flip()
        self.clock.tick(self.framepersec)

    def calibration_time(self):
        self.dt = float(self.clock.get_time()) / 1000.0

    def model_save(self, episode):
        study_model_save(episode)     

    def making_video(self, episode):
        self.env.render(episode)

    def mainQ2targetQ(self):
        study_update()

    def set_init_location_pose_random(self, v):
        self.env.set_init_random(v)

    def get_episode_total_time(self):
        return time.time() - self.start_time

    def get_sensor_value(self):
        return self.distance_device

    def get_xycar_position(self):
        return self.xycar_position

    def get_xycar_yaw(self):
        return self.xycar_yaw

    def get_xycar_steering(self):
        return self.xycar_steering

    def get_step_number(self):
        return self.step_count

    def get_episode_done(self):
        return self.done

    def get_episode_success(self):
        if self.suc_code == 1:
            return "fail"
        elif self.suc_code == 2:
            return "success"
        else:
            return "not done"            

    def get_round_count(self):
        return self.round_count

    def get_memory_size(self):
        return self.memory.size()

    def get_space_bar_input(self):
        return self.space_bar

    def set_lidar_cnt(self, cnt):
        self.sensor_num = cnt
