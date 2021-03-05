import pygame
import numpy as np

# from visdom import Visdom
from agent import Car
from environment import Environment
from dqn import *

# vis = Visdom()

# opts_score = opts=dict(title = "Score", showlegend = True)
# plt_score = vis.line(X=[0], Y=[[0]], opts=opts_score)

# opts_loss = opts=dict(title = "Loss", showlegend = True)
# plt_loss = vis.line(X=[0], Y=[[0]], opts=opts_loss)

# opts_all_score = dict(title = "All Score", showlegend = True)
# plt_all_score = vis.line(X=[0], Y=[[0]], opts=opts_all_score)

# opts_collision = dict(title = "Collision Pos", markersize = 3)
# plt_collision = vis.scatter([[0, 0]], opts=opts_collision)

def reward_in_game(data):
    reward = 1.0

    ultrasonic_left = data[0]
    ultrasonic_right = data[4]

    reward -= abs(ultrasonic_left - ultrasonic_right) * 0.001

    return float(reward)

def reward_end_game(data):
    return 0.0


if __name__ == "__main__":
    is_exit = False

    episode = 0
    max_episode = 1000000
    max_steps = 300000

    batch_size = 500
    min_history = 500

    all_score = 0
    max_score = 0

    epsilon = 0.3
    discount_factor = 0.99
    copy_target_cycle = 100

    learning_rate = 0.0001

    ratio = 0.6 / 128
    ratio *= 100

    env = Environment("rally_map.png", map_type="image", fps=30)
    car = Car()

    net = DDQN(5, 3, [256, 256], learning_rate=learning_rate, skip_frame=2, stack_frame=5)
    video_dir = "video_ddqn"

    train_again_episode = False
    train_again_epsilon = 0.3

    if train_again_episode:
        net.model_load(train_again_episode, eval=False)
        episode = train_again_episode
        epsilon = train_again_epsilon

    while not is_exit or episode < max_episode:
        
        while True:
            # 차량 위치 무작위 초기화
            random_pos, random_yaw = env.get_random_pos_and_yaw(car)
            car.position = random_pos
            car.yaw = random_yaw
            dist, _ = car.get_ultrasonic_distance(env)
            if np.min(dist) >= 50:
                break

        episode += 1
        epsilon = max(0.001, epsilon - 0.0001)

        step = 0
        score = 0

        obs, _ = car.get_ultrasonic_distance(env, ratio=ratio)

        for i in range(net.skip_frame*net.stack_frame):
            net.observation_set.append(obs)

        state = net.skip_stack_frame(obs)

        is_done = 0

        env.save_pos_and_image(car, obs, init=True)

        while not is_done and step < max_steps:
            # 행동 선택
            action = net.get_action(state, epsilon)

            # 조향각 조정
            if action == 0:
                steering_deg = -car.max_steering_deg
            elif action == 1:
                steering_deg = 0
            else:
                steering_deg = car.max_steering_deg

            # 차량 위치 업데이트
            car.update(car.DRIVE, steering_deg, env.dt)
            
            # 새로운 state 취득
            next_obs, _ = car.get_ultrasonic_distance(env, ratio=ratio)
            next_state = net.skip_stack_frame(next_obs)

            # 새로운 state에서의 보상 확인
            car_status = env.status(car)
            if car_status != env.WALL:
                reward = reward_in_game(state)
            else:
                reward = reward_end_game(state)
                is_done = 1

            # 메모리에 저장
            net.append_sample(state, action, reward, next_state, is_done)

            # # 만약 차량 상태가 Goal에 도착했을 경우
            # if car_status == env.GOAL:
            #     while True:
            #         # 차량 위치 무작위 초기화
            #         random_pos, random_yaw = env.get_random_pos_and_yaw(car)
            #         car.position = random_pos
            #         car.yaw = random_yaw
            #         dist, _ = car.get_ultrasonic_distance(env)
            #         if np.min(dist) >= 50:
            #             break
            
            step += 1
            score += reward
            state = next_state
            env.save_pos_and_image(car, next_obs)
        
        all_score += score

        if max_score < score or episode % 100 == 0:
            max_score = max(max_score, score)
            env.save_video(episode, dir=video_dir)
            net.model_save(episode)


        if min_history < len(net.experience_memory):
            loss = net.train_model(discount_factor, batch_size)
            print("Episode: {: >5} | Steps: {: >4} | Score: {: >6.2f} | Max Score: {: >6.2f} | Loss: {: >10.8f} | E: {:.3f}".format(episode, step, score, max_score, loss, epsilon))

            # x, y = car.position
            # vis.line(X=[episode], Y=[score], win=plt_score, update='append', opts=opts_score)
            # vis.line(X=[episode], Y=[loss], win=plt_loss, update='append', opts=opts_loss)
            # vis.line(X=[episode], Y=[all_score], win=plt_all_score, update='append', opts=opts_all_score)
            # vis.scatter([[x, y]], win=plt_collision, update='append', opts=opts_collision)

        if episode % copy_target_cycle == 0:
            net.update_target_model()


            """
            Env Human
            """
            """
            pressed = pygame.key.get_pressed()
            if pressed[pygame.K_LEFT]:
                angle += 5
            elif pressed[pygame.K_RIGHT]:
                angle -= 5
            if pressed[pygame.K_UP]:
                drive = car.DRIVE
            elif pressed[pygame.K_DOWN]:
                drive = car.REVERSE
            elif pressed[pygame.K_s]:
                drive = car.NEUTRAL
                car.velocity = 0
            
            angle = max(-car.max_steering_deg, min(angle, car.max_steering_deg))
            
            car.update(drive, angle, env.get_dt())

            car.update(car.NEUTRAL, i, env.get_dt())
            print("[DRIVE]: {} | DEG: {} | Velocity: {:.2f} | Accel: {:.2f}".format(
                drive, angle, car.velocity, car.acceleration))
            """
