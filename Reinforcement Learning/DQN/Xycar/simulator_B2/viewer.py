#!/usr/bin/env python

import time
from xycarRL import *
from rosModule import rosmodule

def next_state_rtn(laser_msg):
    ratio = 213.33
    idx = [0, 44, 89, 134, 179]
    current_ipt = []

    for i in range(len(idx)):
        if idx[i] == 0:
            tmp = [laser_msg[idx[i]], laser_msg[idx[i]+1], laser_msg[idx[i]+2]]
        elif idx[i] == 179:
            tmp = [laser_msg[idx[i]-2], laser_msg[idx[i]-1], laser_msg[idx[i]]]
        else:
            tmp = [laser_msg[idx[i]-1], laser_msg[idx[i]], laser_msg[idx[i]+1]]
        
        current_ipt.append(min(tmp))
    
    rtn = np.array(current_ipt)

    for j in range(len(curent_ipt)):
        rtn[j] *= ratio
    return rtn

if __name__ == '__main__':
    xycar = learning_xycar(False)
    xycar.set_map("main_map") # snake, square
    xycar.pygame_init()

    # lidar_cnt = 5
    # xycar.set_lidar_cnt(lidar_cnt)
    
    hidden_layer = [256, 256]
    xycar.set_hidden_size(hidden_layer)

    state_select = {
        "car sensor" : True,
        "car yaw" : False,
        "car position" : False,
        "car steer" : True
    }

    xycar.state_setup(state_select)
    
    ros_module = rosmodule()

    #xycar.screen_init()
    xycar.ML_init("DDQN")

    xycar.set_init_location_pose_random(True) 

    view_epi = 506

    xycar.load_model(view_epi)

    time.sleep(0.5)

    angle = 0
    max_angle = np.radians(30.0)
    handle_weights = np.radians(6.6)
    rate = rospy.Rate(30)

    while (not xycar.pygame_exit):
        state = xycar.episode_init()

        while (xycar.get_episode_done()) or (not xycar.pygame_exit):
            xycar.pygame_exit_chk()
            xycar.calibration_time()
            action = xycar.get_action_viewer(state)

            action = xycar.get_action_viewer(state)
            if action == 2:
                angle += handle_weights
            elif action == 0:
                angle -= hangle_weights
            elif action == 1:
                angle = 0
            
            angle = max(-max_angle, min(angle, max_angle))
            ros_module.auto_drive(angle, 0.234)

            if xycar.get_episode_done():
                break
            
            next_state = xycar.step(action)

            state = next_state
            xycar.display_flip()
        
