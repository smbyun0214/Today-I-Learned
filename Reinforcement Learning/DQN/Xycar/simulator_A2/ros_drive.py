#!/usr/bin/env python

import time#, rospy
from env.xycarRL import *
from env.rosModule import *

def action_control(state):
    action = 1
    return action

if __name__ == '__main__':
    rosmd = rosmodule("xycar_simulator")
    xycar = learning_xycar(False)
    xycar.set_map("square") # snake, square
    xycar.pygame_init()

    lidar_cnt = 180
    xycar.set_lidar_cnt(lidar_cnt)

    state_select = {
        "car sensor" : True,
        "car yaw" : False,
        "car position" : False,
        "car steer" : False
    }

    xycar.state_setup(state_select)
    xycar.screen_init()

    xycar.set_init_location_pose_random(False) 

    time.sleep(0.5)
    ang, spd = 0.0, 0.0

    while (not xycar.pygame_exit):
        state = xycar.episode_init()

        while (not xycar.get_episode_done()) and (not xycar.pygame_exit):
            xycar.pygame_exit_chk()
            xycar.calibration_time()

            rosmd.pub_fake_lidar_data(state)
            ang, spd = rosmd.get_motor_data()
            xycar.env.car.set_drive(ang, spd)
            next_state = xycar.step_not_move()

            if xycar.get_episode_done():
                break

            state = next_state
            xycar.display_flip()
        
