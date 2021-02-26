#!/usr/bin/env python
# -*- coding: utf-8 -*-

import my_reward

from xycarRL import *
# from visual import *
    
if __name__ == '__main__':
    xycar = learning_xycar()
    xycar.set_map("main_map") # snake, square

    hyper_param = {
        "sensor_num" : 5,
        "learning_rate" : 0.0001,
        "discount_factor" : 0.99,
        "optimizer_steps" : 5,
        "batch_size" : 100,
        "min_history" : 100,
        "buffer_limit" : 100000,
        "max_episode" : 10000,
        "update_cycle" : 100,
        "hidden_size" : [256, 256]
    }

    xycar.set_hyperparam(hyper_param)

    state_select = {
        "car sensor" : True,
        "car yaw" : False,
        "car position" : False,
        "car steer" : True
    }

    xycar.state_setup(state_select)
    xycar.Experience_replay_init()
    xycar.ML_init("DDQN")

    xycar.set_init_location_pose_random(True) 

    # visual = visualize(port=8888)
    # visual.chart_init()

    ALL_STEPS, ALL_REWARD = 0, 0
    episode = 0

    while (0 <= episode <= int(hyper_param["max_episode"])):
        episode += 1
        epsilon = max(0.01, 0.08 - 0.01*(float(episode)/200.0))
        xycar.set_E_greedy_func(epsilon)

        ## 환경 초기화 ##
        state = xycar.episode_init()
        reward, score = 0, 0.0

        while not xycar.get_episode_done():
            ## state 값으로 action 값 받아오기 ##
            action = xycar.get_action(state)

            ## 다음 state 값 받아오기 ##
            next_state = xycar.step(action)

            ## episode 종료 여부 확인하기 ##
            if xycar.get_episode_done():
                
                ## 에피소드 종료시 지급할 reward 설계에 필요한 인자 넣기 ##
                for_reward = [
                    xycar.get_sensor_value(),
                    xycar.get_xycar_steering()
                ]
                
                reward += my_reward.reward_end_game(for_reward)
                
                ## Experience_replay 메모리에 state,action,next_state,reward 순으로 넣기
                xycar.Experience_replay_memory_input(state, action, next_state, reward)
                break

            ## 에피소드 중 지급할 reward 설계에 필요한 인자 넣기 ##
            for_reward = [
                xycar.get_sensor_value(),
                xycar.get_xycar_steering()
            ]

            reward = my_reward.reward_in_game(for_reward)
            
            ## Experience_replay 메모리에 state, action, next_state, reward 순으로 넣기 ##
            xycar.Experience_replay_memory_input(state, action, next_state, reward)
            
            state = next_state
            score += reward
            
            if (xycar.get_step_number() % 100 == 0) and (xycar.get_step_number() != 0):
                print("step: {}, epsilon: {:.1f}%, score: {}, max score: {}".format(xycar.get_step_number(), epsilon*100, score, xycar.get_max_score()))

        ALL_STEPS += xycar.get_step_number()
        ALL_REWARD += score

        if xycar.get_max_score() < score:
            xycar.max_score_update(score)
            xycar.model_save(episode)
            xycar.making_video(episode)

        if xycar.get_memory_size() > hyper_param["min_history"]:
            ## 훈련 개시 함수 ##
            loss = xycar.train(hyper_param["optimizer_steps"])
            # visual.loss_graph_update(episode, loss)

        # visual.dead_position_update(xycar.get_xycar_position())
        # visual.reward_update(episode, score)
        # visual.learning_curve_update(episode, ALL_REWARD)

        if (xycar.get_memory_size() > hyper_param["min_history"]) and ((episode % hyper_param["update_cycle"]) == 0) and (episode != 0):
            ## main Q를 target Q에 update 하기 ##
            xycar.mainQ2targetQ()

        if (episode % 10 == 0) and (episode != 0):
            print("episode: {}, memory size: {}, epsilon: {:.1f}%, score: {}, max score: {}".format(episode, xycar.get_memory_size(), epsilon*100, score, xycar.get_max_score()))


    xycar.max_score_update(score)
    xycar.model_save(episode)
    xycar.making_video(episode)