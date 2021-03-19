# -*- coding: utf-8 -*-

from dqn import *
from simulator.simulator import Simulator

load_model_comment = None
load_model_episode = 370

sensor_count = 8
action_size = 6

skip_frame = 3
stack_frame = 5

learning_rate = 0.0001
epsilon_init = 0.3
memory_maxlen = 1000000

model = LinearModel(sensor_count, stack_frame, action_size)
agent = DQNAgent(model, learning_rate, epsilon_init, skip_frame, stack_frame, memory_maxlen)

env = Simulator(map="rally_map.png", fps=5)

episode = 0

while True:
    obs, _ = env.reset()

    agent.reset(obs)
    gear = env.BREAK

    state = agent.skip_stack_frame(obs)

    while not env.is_done:
        step = 0

        env.render()

        action = agent.get_action(state)

        # 조향각 조정
        if action % 3 == 0:
            steering_deg = -env.max_steering_deg
        elif action % 3 == 1:
            steering_deg = 0
        elif action % 3 == 2:
            steering_deg = env.max_steering_deg

        # 기어 조정
        if action // 3 == 0:
            if gear != env.DRIVE:
                env.car.reset()
            gear = env.DRIVE
        elif action // 3 == 1:
            if gear != env.REVERSE:
                env.car.reset()
            gear = env.REVERSE

        next_obs, _ = env.step(gear, steering_deg)
        next_state = agent.skip_stack_frame(next_obs)

        state = next_state

    episode += 1
