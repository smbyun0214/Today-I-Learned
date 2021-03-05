#!/usr/bin/env python
import time, random, cv2, copy, os, time, collections, pygame, hashlib
import numpy as np

from PIL import Image
from collections import OrderedDict
from math import sin, cos, tan, asin, acos, radians, degrees, copysign, trunc

import torch
import torch.nn as nn
import torch.nn.functional as F

from env.resource import *
import base64


class ReplayBuffer:
    def __init__(self, buffer_limit):
        self.buffer = collections.deque(maxlen=buffer_limit)
    
    def put(self, transition):
        self.buffer.append(transition)
    
    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []
        
        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask_lst.append([done_mask])

        return torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
               torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
               torch.tensor(done_mask_lst)
    
    def size(self):
        return len(self.buffer)
    
    def free(self):
        self.buffer.clear()

class Car:
    location = [[-64,  0], [-63,-22], [-63, 22], [-53,-27], [-54, 27],
                [-45,-30], [-21,-30], [  2,-30], [ 24,-30], [ 46,-30],
                [-47, 29], [-24, 30], [  0, 30], [ 25, 30], [ 46, 30],
                [ 54,-23], [ 55, 22], [ 60,-14], [ 60, 14], [ 63,  0]]
    
    def __init__(self, pygame_go, state_num, map_data=None, screen=None):
        self.lidar_angle = 0
        self.lidar_posi = [62, -1]
        self.location_cnt = len(self.location)
        self.screen = screen
        self.lidar = distance_device()
        self.lidar_data_cnt = state_num - 1
        self.lidar_init(self.lidar_data_cnt)
        self.lidar_vector_aver = 0
        self.racing_map = map_data
        self.pygame_on = pygame_go
        self.init_pos_random = False
        self.x_stop = False
        self.y_stop = False
        self.wall_ac = False
        self.goalIn = False
        
        if self.pygame_on:
            self.car_x_ori = [-64,-64, 64, 64]
            self.car_y_ori = [-32, 32,-32, 32]
            
            self.colorImage  = self.car_image_decode()
            self.rotated = self.PIL2PygameIMG(0)
            self.car_image = pygame.sprite.Sprite()
            self.car_img_x = 0
            self.car_img_y = 0

    def car_image_decode(self):
        decode_car_img = base64.b64decode(car_image)
        return Image.frombytes('RGB', (128,64), decode_car_img, 'raw')

    def set_random(self, v):
        self.init_pos_random = v
        
    def set_drive(self, angle, velocity):
        self.steering = angle
        self.Linear_velocity = velocity

    def set_init_car(self):
        if self.init_pos_random:
            pos = random.randint(0, 1)
            ang = random.randint(0, 1)
            self.position = [ self.racing_map.INIT_POSITION_RT[pos][0], self.racing_map.INIT_POSITION_RT[pos][1] ]
            self.polor_coordinate = [ 0.0, self.racing_map.INIT_ANGLE_RT[ang] ]
        else:
            self.position = [ self.racing_map.INIT_POSITION_RF[0], self.racing_map.INIT_POSITION_RF[1] ]
            self.polor_coordinate = [ 0.0, self.racing_map.INIT_ANGLE_RF ]
        self.Linear_velocity = 1.0
        self.angular_velocity = 0.0
        self.length = 1.28
        self.steering = 0.0
        self.max_steering = 30.0

        self.end = False
        self.suc = None
        self.goal_count = 0
        self.x_stop = False
        self.y_stop = False
        self.wall_ac = False

        self.goal_count = 0
    
    def lidar_init(self, data_cnt):
        self.lidar_distance = [0.0 for _ in range(data_cnt)]

    def stop(self, accident):
        self.Linear_velocity = 0.0
        self.steering = 0.0
        self.end = True
        self.suc = not accident

    def next_game(self, accident):
        self.Linear_velocity = 0.0
        self.steering = 0.0
        self.end = True
        self.suc = not accident

    def PIL2PygameIMG(self, angle):
        IMG = self.colorImage.rotate(angle, expand=True, center=(64, 32), fillcolor=(255,255,255))
        return pygame.image.fromstring(IMG.tobytes("raw", 'RGB'), IMG.size, 'RGB')

    def update(self, dt):
        self.angular_velocity = 0.0
        
        if self.steering != 0.0:
            self.angular_velocity = (self.Linear_velocity / self.length) * sin(radians(self.steering))
        
        self.polor_coordinate[0] = dt * self.Linear_velocity
        
        self.position[0] += self.polor_coordinate[0] * cos(-radians(self.polor_coordinate[1]))
        self.position[1] += self.polor_coordinate[0] * sin(-radians(self.polor_coordinate[1]))

        self.polor_coordinate[1] += degrees(self.angular_velocity) * dt

        car_center = [
            self.position[0] * 100, 
            self.position[1] * 100
        ]
        
        if self.pygame_on:
            car_x = [0,0,0,0]
            car_y = [0,0,0,0]

            self.rotated = self.PIL2PygameIMG(self.polor_coordinate[1])

            for i in range(4):
                car_x[i] = self.car_x_ori[i] * cos(-radians(self.polor_coordinate[1])) - self.car_y_ori[i] * sin(-radians(self.polor_coordinate[1])) + car_center[0]
                car_y[i] = self.car_x_ori[i] * sin(-radians(self.polor_coordinate[1])) + self.car_y_ori[i] * cos(-radians(self.polor_coordinate[1])) + car_center[1]

            self.car_img_x = int(round(min(car_x)))
            self.car_img_y = int(round(min(car_y)))
        
        lidar_position = [0.0, 0.0]
        lidar_position[0] = self.lidar_posi[0] * cos(-radians(self.polor_coordinate[1])) - self.lidar_posi[1] * sin(-radians(self.polor_coordinate[1])) + car_center[0]
        lidar_position[1] = self.lidar_posi[0] * sin(-radians(self.polor_coordinate[1])) + self.lidar_posi[1] * cos(-radians(self.polor_coordinate[1])) + car_center[1]
        
        if self.pygame_on:
            pygame.draw.circle(self.screen, (0,0,255), [int(round(lidar_position[0])), int(round(lidar_position[1]))], 5)
        
        _, distance, cordinate = self.lidar.confirm_device(-self.polor_coordinate[1], lidar_position, self.racing_map.OBS, self.lidar_data_cnt, min_angle=-90, max_angle=90)

        if len(distance) == self.lidar_data_cnt:

            for i in range(self.lidar_data_cnt):
                self.lidar_distance[i] = distance[i]

            if self.pygame_on:
                self.screen.blit(self.rotated, [self.car_img_x, self.car_img_y])
                for i in range(self.lidar_data_cnt):
                    pygame.draw.line(self.screen, [255,0,0], lidar_position, (cordinate[i][0], cordinate[i][1]))

        for i in self.racing_map.WARP:
            if i[2] == 220:
                alpha = 62
                beta = 0
                if (45 < int(-self.polor_coordinate[1])%180 < 135):
                    alpha = -alpha
            elif i[3] == 220:
                alpha = 0
                beta = 62
                if (135 < int(-self.polor_coordinate[1])%180 < 180) or (-180 < int(-self.polor_coordinate[1])%180 < 135):
                    beta = -beta

            if (i[0] <= car_center[0]+beta <= i[0]+i[2]) and (i[1] <= car_center[1]+alpha <= i[1]+i[3]) and (self.goalIn==False):
                self.goalIn = True
                self.goal_count += 1
                if self.racing_map.MAP_NAME == "square":
                    self.racing_map.WARP_ANGLE += 180
            elif ((i[0] > car_center[0]+beta) or (car_center[0]+beta > i[0]+i[2])) and ((i[1]+alpha > car_center[1]) or (car_center[1]+alpha > i[1]+i[3])) and (self.goalIn==True):
                self.goalIn = False

        outline = [0.0, 0.0]

        for loc in range(self.location_cnt):
            outline[0] = self.location[loc][0] * cos(-radians(self.polor_coordinate[1])) - self.location[loc][1] * sin(-radians(self.polor_coordinate[1])) + car_center[0]
            outline[1] = self.location[loc][0] * sin(-radians(self.polor_coordinate[1])) + self.location[loc][1] * cos(-radians(self.polor_coordinate[1])) + car_center[1]

            for ww in self.racing_map.WARP:
                warp, bun = self.lidar.rect_in_point(outline, ww)

                if warp:
                    self.polor_coordinate = [0.0, self.racing_map.WARP_ANGLE]
                    break
            
            accidents = []
            for wall in self.racing_map.OBS:
                accident, bun = self.lidar.rect_in_point(outline, wall)
                if accident:
                    self.wall_ac = True
                    if bun == "L" or bun == "R": 
                        self.position[0] -= self.polor_coordinate[0] * cos(-radians(self.polor_coordinate[1]))
                    if bun == "U" or bun == "D":
                        self.position[1] -= self.polor_coordinate[0] * sin(-radians(self.polor_coordinate[1]))
                    accidents.append(accident)
                else:
                    self.wall_ac = False
  
            if True in accidents:
                #print("-------------wall-------------")
                self.stop(True)
                break

            goals = []
            for aword_point in self.racing_map.GOAL:
               goal, bun = self.lidar.rect_in_point(outline, aword_point)
               goals.append(goal)
               
            #if True in goals:
            #    print("-------------goal-------------")
            #    self.goal_count += 1
            #    if self.goal_count == len(GOAL):
            #    self.stop(False)
            #    break

class Game:
    def __init__(self, pygame_on, sensor_num, fps, map_data=None, Font_1=None, Font_2=None, pygame_screen=None):
        self.set_random = True
        self.racing_map = map_data
        self.car = Car(pygame_on, sensor_num, map_data=self.racing_map)
        self.pygame_on = pygame_on
        if self.pygame_on:
            self.FF = Font_1
            self.DF = Font_2
            self.screen = pygame_screen
            self.car = Car(self.pygame_on, sensor_num, map_data=self.racing_map, screen=self.screen)
            self.stop_space = 0
        
        self.steering_AOC = 100 * 2
        self.lidar = []
        self.start_time = time.time()
        self.switch = True

        self.max_score = 0.0
        self.fps = fps

        self.carImage = self.car.car_image_decode()
        self.fourcc = cv2.VideoWriter_fourcc(*'DIVX')

    def map_input(self, all_sprite_list, wall_list, goal_list):
        self.all_sprite_list = all_sprite_list
        self.wall_list = wall_list
        self.goal_list = goal_list
    
    def set_init_random(self, v):
        self.car.set_random(v)

    def reset(self):
        self.step_init()

        self.time_chk = time.time()

        self.steps_beyond_done = None
        self.success = None
        self.exit = False
 
        self.action = 0
        self.angle = 0
        self.start_time = round(time.time(),2)
        self.chk_time = time.time()

        self.reward = 0
        self.car.set_init_car()

    def key_return(self):
        pygame.event.get()
        pressed = pygame.key.get_pressed()
        if pressed[pygame.K_RIGHT]:
            D = 2
        elif pressed[pygame.K_LEFT]:
            D = 0
        else:
            D =1
        return D

    def space_next(self):
        pygame.event.get()
        pressed = pygame.key.get_pressed()
        if pressed[pygame.K_SPACE]:
            self.car.end = True

    def img_loc(self, Frame, IMG, angle, x, y):
        car_x_ori = [-64,-64, 64, 64]
        car_y_ori = [-32, 32,-32, 32]

        IMG = IMG.rotate(angle, expand=True, center=(64, 32), fillcolor = "white")
        rotated = cv2.cvtColor(np.asarray(IMG), cv2.COLOR_RGB2BGR)
        height = rotated.shape[0]
        width = rotated.shape[1]

        car_x = [0,0,0,0]
        car_y = [0,0,0,0]
        for i in range(4):
            car_x[i] = car_x_ori[i] * cos(-radians(angle)) - car_y_ori[i] * sin(-radians(angle)) + x
            car_y[i] = car_x_ori[i] * sin(-radians(angle)) + car_y_ori[i] * cos(-radians(angle)) + y

        pic_pos = [int(round(min(car_x))), int(round(min(car_y)))]

        F = copy.deepcopy(Frame)

        for i in range(width):
            for j in range(height):
                if (pic_pos[1]+j >= self.racing_map.SCREEN_HEIGHT) or (pic_pos[0]+i >= self.racing_map.SCREEN_WIDTH):
                    continue
                F[pic_pos[1]+j, pic_pos[0]+i] = rotated[j, i]

        for i in self.racing_map.OBS:
            lu = (int(i[0]), int(i[1]))
            rd = (int(i[0]+i[2]), int(i[1]+i[3]))
            F = cv2.rectangle(F, lu, rd, (0,0,0), -1)

        return F

    def step_init(self):
        self.save_epi = []

    def step_save(self, angle, x, y):
        self.save_epi.append([angle, x, y])

    def render(self, epi_num):
        width = self.racing_map.SCREEN_WIDTH
        height = self.racing_map.SCREEN_HEIGHT
        FileName = "./video/"+str(epi_num)+".avi"
        out = cv2.VideoWriter(FileName, self.fourcc, float(self.fps), (width, height))
        F = np.full((height, width, 3), 255, np.uint8)
 
        all_frame = len(self.save_epi)

        print(" #### start making video : " + str(epi_num) + ".avi #### | frame cnt : " + str(all_frame))
        for i in range(all_frame):
            angle = self.save_epi[i][0]
            x = self.save_epi[i][1]
            y = self.save_epi[i][2]
            
            D = self.img_loc(F, self.carImage, angle, x, y)

            cv2.putText(D, "episode : "+str(epi_num), (30, 100), cv2.FONT_HERSHEY_DUPLEX, 1, 2, 2)
            out.write(D)

        print(" ##### end making video : " + str(epi_num) + ".avi ##### ")
        out.release()
        
    def step_not_move(self, dt):
        self.car.steering = max(-self.car.max_steering, min(self.car.steering, self.car.max_steering))
            
        if self.pygame_on:
            self.screen.fill([255,255,255])
            self.all_sprite_list.update(dt)

        self.car.update(dt)

        self.time = time.time() - self.start_time
        self.viewtime = trunc(self.time)

        if self.pygame_on:
            self.runtime = self.FF.render("Running time : " + str(self.viewtime), True, (28,0,0))
            self.all_sprite_list.draw(self.screen)
            self.screen.blit(self.runtime,(15,15))
            
        suc_code = 0
        if self.car.end == False:
            suc_code = 0
        elif self.car.end and (self.car.suc == False):
            suc_code = 1
        elif self.car.end and self.car.suc:
            suc_code = 2

        self.step_save(self.car.polor_coordinate[1], self.car.position[0]*100, self.car.position[1]*100)
        info = {"car_sensor": self.car.lidar_distance, "car_position":[self.car.position[0]*100,self.car.position[1]*100], "car_yaw":self.car.polor_coordinate[1], "car_steer":self.car.steering}
        goal_count = self.car.goal_count
        end = self.car.end

        #print(info)
 
        return info, goal_count, suc_code

    def step(self, action, dt):

        if action == 2:
            self.car.steering -= self.steering_AOC * dt # Right
        elif action == 0:
            self.car.steering += self.steering_AOC * dt # Left
        elif action == 1:
            self.car.steering = 0

        self.car.steering = max(-self.car.max_steering, min(self.car.steering, self.car.max_steering))
            
        if self.pygame_on:
            self.screen.fill([255,255,255])
            self.all_sprite_list.update(dt)

        self.car.update(dt)

        self.time = time.time() - self.start_time
        self.viewtime = trunc(self.time)

        if self.pygame_on:
            self.runtime = self.FF.render("Running time : " + str(self.viewtime), True, (28,0,0))
            self.all_sprite_list.draw(self.screen)
            self.screen.blit(self.runtime,(15,15))
            
        suc_code = 0
        if self.car.end == False:
            suc_code = 0
        elif self.car.end and (self.car.suc == False):
            suc_code = 1
        elif self.car.end and self.car.suc:
            suc_code = 2

        self.step_save(self.car.polor_coordinate[1], self.car.position[0]*100, self.car.position[1]*100)
        info = {"car_sensor": self.car.lidar_distance, "car_position":[self.car.position[0]*100,self.car.position[1]*100], "car_yaw":self.car.polor_coordinate[1], "car_steer":self.car.steering}
        goal_count = self.car.goal_count
        end = self.car.end

        #print(info)
 
        return info, goal_count, suc_code

class Wall(pygame.sprite.Sprite):
    def __init__(self, x, y, width, height):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.Surface([width,height])
        self.image.fill([0,0,0])
        self.rect = self.image.get_rect()
        self.rect.y = y
        self.rect.x = x

class obs_make:
    all_sprite_list = pygame.sprite.Group()
    wall_list = pygame.sprite.Group()
    goal_list = pygame.sprite.Group()
    
    def __init__(self, map_data):
        self.racing_map = map_data
        for i in self.racing_map.OBS:
            wall = Wall(i[0],i[1],i[2],i[3])
            self.wall_list.add(wall)
            self.all_sprite_list.add(wall)

        for i in self.racing_map.GOAL:
            wall = Wall(i[0],i[1],i[2],i[3])
            wall.image.fill([255,255,255])
            self.goal_list.add(wall)
            self.all_sprite_list.add(wall)

    def get_list(self): 
        return [self.wall_list, self.goal_list]

    def get_all_list(self):
        return self.all_sprite_list


class distance_device:

  def vector_aver(self, theta, distance, beam_cnt, min_angle=-90, max_angle=90):
    s = 0.0
    c = 0.0

    cnt = 0

    S = []
    C = []

    period = int((abs(min_angle) + abs(max_angle))/(beam_cnt-1))

    for i in range(min_angle, max_angle+1, period): 
      S.append(distance[cnt] * sin(radians(i)))
      C.append(distance[cnt] * cos(radians(i)))
      cnt += 1

    s = sum(S) / float(beam_cnt)
    c = sum(C) / float(beam_cnt)

    cr = acos(c/(((s**2) + (c**2)) ** (0.5)))

    return cr

  def confirm_device(self, theta, point, OBS, beam_cnt, min_angle=-90, max_angle=90):
    dts, loc = [], []
    if (beam_cnt-1) == 0:
        period = 1
    else:
        period = int((abs(min_angle) + abs(max_angle))/(beam_cnt-1))

    for i in range(min_angle, max_angle+1, period): 
      T = float(i) - theta
      ok, dt, lo = self.line_to_sprite_group([point[0], point[1]], OBS, T)
      if not ok:
        continue
      dts.append(dt)
      loc.append(lo)
      if len(dts) >= beam_cnt:
        break

    if len(dts) == 0:
      return False, float('inf'), [float('inf'), float('inf')]
    return True, dts, loc
    
    
  def line_to_sprite_group(self, point, OBS, theta):
    distance = []
    location = []
    for wall_num in range(len(OBS)):
      OK, dist, loca = self.line_to_rect(point, theta, OBS[wall_num])
      if OK:
        distance.append(dist)
        location.append(loca)

    if len(distance) == 0:
      return False, float('inf'), [float('inf'), float('inf')]

    min_distance = min(distance)
    min_index = distance.index(min_distance)

    return True, min_distance, location[min_index]
    
    
  def line_to_rect(self, point, theta, wall):
    m = tan(radians(-theta))

    rectLine_x = [
        wall[0],
        wall[0] + wall[2]
    ]

    rectLine_y = [
        wall[1],
        wall[1] + wall[3]
    ]

    cross = []
    distance = []

    bx = float(point[1])-(m*float(point[0]))
  
    for rl in rectLine_x:
      sol = float(m*float(rl) + bx)

      if sol > rectLine_y[1]:
        continue
      if sol < rectLine_y[0]:
        continue

      croc = [rl, sol]
      dit = self.distance_calc(point, croc)


      if abs(float(sol) - (dit*sin(radians(-theta)) + point[1])) > 0.01:
        continue
      if abs(float(rl) - (dit*cos(radians(-theta)) + point[0])) > 0.01:
        continue

      cross.append(croc)
      distance.append(self.distance_calc(point, croc))

    by = -1 * float('inf')
    if m != 0:
      by = float(point[0])-(float(point[1])/m)

    for rl in rectLine_y:
      sol = float('inf')
      if m != 0:
        sol = float(float(rl)/m + by)

      if sol > rectLine_x[1]:
        continue
      if sol < rectLine_x[0]:
        continue

      croc = [sol, rl]
      dit = self.distance_calc(point, croc)

      if abs(float(sol) - (dit*cos(radians(-theta)) + point[0])) > 0.01:
        continue
      if abs(float(rl) - (dit*sin(radians(-theta)) + point[1])) > 0.01:
        continue

      cross.append(croc)
      distance.append(self.distance_calc(point, croc))

    if len(distance) == 0:
      return False, -1, [-1, -1]

    min_distance = min(distance)
    min_index = distance.index(min_distance)

    return True, min_distance, cross[min_index]
    
    
  def distance_calc(self, point_1, point_2):
    if float('inf') in point_1:
        return float('inf')
    if float('inf') in point_2:
        return float('inf')
    return (((point_1[0] - point_2[0])**2) + ((point_1[1] - point_2[1])**2)) ** (0.5)
    
    
  def rect_in_point(self, point, wall):
    a = {"L":abs(point[0] - wall[0]), "R":abs(point[0] - (wall[0]+wall[2])), "U":abs(point[1] - wall[1]), "D":abs(point[1] - (wall[1]+wall[3]))}
    b = min(a.values())
    rtn = ""
    for key, value in a.items():
      if value == b:
          rtn = key
          break
    if (wall[0] <= point[0] <= (wall[0]+wall[2])) and (wall[1] <= point[1] <= (wall[1]+wall[3])):
      return True, rtn
    return False, ""
    
