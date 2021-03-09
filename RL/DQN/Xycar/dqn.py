import os
import io
import dill
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter

from collections import deque


# 딥러닝 모델
# - 일반 네트워크와 타겟 네트워크 2가지 모델 이용
class NN(nn.Module):

    def __init__(self, input_size, stack_frame, action_size):   
        super(NN, self).__init__()

        self.input_size = input_size
        self.stack_frame = stack_frame
        self.action_size = action_size

        # 네트워크 모델: 3층의 은닉층으로 구성된 인공신경망
        self.fc1 = nn.Linear(self.input_size*self.stack_frame, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, self.action_size)
    

    def forward(self, x):
        # 입력: 상태
        # 출력: 각 행동에 대한 Q 값
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class DQNAgent():

    def __init__(self, model, target_model, optimizer, epsilon_init=1.0, memory_maxlen=100000, skip_frame=4, stack_frame=10):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.model = model.to(self.device)
        self.target_model = target_model.to(self.device)
        
        self.optimizer = optimizer
        self.loss = F.smooth_l1_loss

        self.experience_memory = deque(maxlen=memory_maxlen)
        self.observation_set = deque(maxlen=skip_frame*stack_frame)

        self.epsilon = epsilon_init
        self.epsilon_min = 0.01

        self.skip_frame = skip_frame
        self.stack_frame = stack_frame

        self.update_target()


    def update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())
    

    def get_action(self, state):
        if self.model.training and self.epsilon >= np.random.sample():
            return np.random.randint(0, self.model.action_size)
        
        with torch.no_grad():
            Q = self.model(torch.FloatTensor(state).unsqueeze(0).to(self.device))
            return np.argmax(Q.cpu().detach().numpy())


    def skip_stack_frame(self, observation):
        self.observation_set.append(observation)

        state = np.zeros([self.model.input_size*self.model.stack_frame])
        for i in range(self.stack_frame):
            state[self.model.input_size*i:self.model.input_size*(i+1)] = self.observation_set[-1-(self.skip_frame * i)]
        
        return state


    def append_sample(self, state, action, reward, next_state, done):
        self.experience_memory.append((state, action, reward, next_state, done))


    def train_model(self, discount_factor, batch_size):
        batch_indices = np.random.choice(len(self.experience_memory), min(batch_size, len(self.experience_memory)), replace=False)

        state_batch         = torch.FloatTensor(np.stack([self.experience_memory[idx][0] for idx in batch_indices], axis=0)).to(self.device)
        action_batch        = torch.FloatTensor(np.stack([self.experience_memory[idx][1] for idx in batch_indices], axis=0)).to(self.device)
        reward_batch        = torch.FloatTensor(np.stack([self.experience_memory[idx][2] for idx in batch_indices], axis=0)).to(self.device)
        next_state_batch    = torch.FloatTensor(np.stack([self.experience_memory[idx][3] for idx in batch_indices], axis=0)).to(self.device)
        done_batch          = torch.FloatTensor(np.stack([self.experience_memory[idx][4] for idx in batch_indices], axis=0)).to(self.device)

        eye = torch.eye(self.model.action_size).to(self.device)
        one_hot_action = eye[action_batch.view(-1).long()]
        q = (self.model(state_batch) * one_hot_action).sum(1)

        with torch.no_grad():
            max_q = torch.max(q).item()
            next_q = self.target_model(next_state_batch)
            target_q = reward_batch + next_q.max(1).values*(discount_factor*(1 - done_batch))

        loss = self.loss(q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.data, max_q


    def model_save(self, episode):
        script_dir = os.path.dirname(__file__) 
        dir_path = os.path.join(script_dir, "save_dqn")
        if not os.path.isdir(dir_path):
            os.mkdir(dir_path)
        save_path = os.path.join(script_dir, "save_dqn", "main_model_{:06}.pth".format(episode))

        buffer = io.BytesIO()
        torch.save(self.model.state_dict(), buffer, pickle_module=dill, _use_new_zipfile_serialization=False)
        with open(save_path, "wb") as f:
            f.write(buffer.getvalue())


    def model_load(self, episode, eval=True):
        script_dir = os.path.dirname(__file__) 
        load_path = os.path.join(script_dir, "save_dqn", "main_model_{:06}.pth".format(episode))
        
        # Python3
        self.model.load_state_dict(torch.load(load_path, map_location=self.device))
        self.target_model.load_state_dict(self.model.state_dict())

        if eval:
            self.model.eval()
            self.target_model.eval()

        print(self.model.training, self.target_model.training)



if __name__ == "__main__":
    from simulator.viewer import *
    from simulator.car import *

    car = Car((500, 500), np.radians(0))
    gear = car.BREAK
    steering_deg = 0

    skip_frame = 4
    stack_frame = 10

    run_step = 1000
    target_update_step = 1000
    save_step = 10

    model = NN(5, stack_frame, 3)
    target_model = NN(5, stack_frame, 3)

    learning_rate = 0.01
    discount_factor = 0.99
    batch_size = 32

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    agent = DQNAgent(model, target_model, optimizer)

    agent.model_load(880)
    
    if agent.model.training:
        writer = SummaryWriter()

    background_origin = cv.imread("simulator/map/rally_map4.png")

    height, width = background_origin.shape[:2]

    # 1초 = 1000ms
    # 30fps = 1000/30
    # delay = 1000//30
    delay = 10
    is_done = False

    episode = 0

    while not is_done:
        is_episode_done = 0
        episode_rewards = 0

        start_points, end_points, _ = get_ultrasonic_distance(background_origin, car)
        distances = np.sqrt(np.sum((start_points - end_points)**2, axis=1))
        distances_meter = rint(distances * car.meter_per_pixel * 100)

        for i in range(skip_frame * stack_frame):
            agent.observation_set.append(distances_meter)

        state = agent.skip_stack_frame(distances_meter)

        car.position = (np.random.randint(100, 200), 100)
        car.yaw = np.random.uniform(np.pi/5*2, np.pi/5*3)

        losses = []
        max_qs = []

        step = 0
        
        while not is_episode_done:
            if not agent.model.training:
                background = background_origin.copy()
                draw_car(background, car)
                cv.imshow("simulator", background)
                print(distances_meter)
            # print(state)

            action = agent.get_action(state)

            # 조향각 조정
            if action == 0:
                steering_deg = -car.max_steering_deg
            elif action == 1:
                steering_deg = 0
            else:
                steering_deg = car.max_steering_deg

            car.update(1/30, car.DRIVE, steering_deg)

            car_state = is_collision(background_origin, car, [255, 51, 4])
            if car_state == -1:
                is_episode_done = 1
                reward = 0
            elif car_state == 1:
                is_episode_done = 1
                reward = 2
                print("도착:", episode)
            else:
                reward = 1

            start_points, end_points, _ = get_ultrasonic_distance(background_origin, car)
            distances = np.sqrt(np.sum((start_points - end_points)**2, axis=1))
            distances_meter = rint(distances * car.meter_per_pixel * 100)
            next_state = agent.skip_stack_frame(distances_meter)

            if agent.model.training:
                agent.append_sample(state, action, reward, next_state, is_episode_done)

                loss, maxQ = agent.train_model(discount_factor, batch_size)

                losses.append(loss)
                max_qs.append(maxQ)

                # 타겟 네트워크 업데이트
                # 일정 스텝마다 타겟 네트워크 업데이트 수행
                if step != 0 and step % target_update_step == 0:
                    agent.update_target()


            episode_rewards += reward
            state = next_state
            step += 1

            if not agent.model.training:
                key = cv.waitKey(delay)
                if key == ord("q"):
                    is_done = True
                    break

        
        if agent.epsilon > agent.epsilon_min:
            agent.epsilon -= 1 / run_step
        
        # 모델 저장
        if agent.model.training and episode % save_step == 0 and episode != 0 :
            agent.model_save(episode)

        if agent.model.training:
            writer.add_scalar("DQN/loss", np.mean(losses), episode)
            writer.add_scalar("DQN/max_q", np.mean(max_qs), episode)
            writer.add_scalar("DQN/reward", episode_rewards, episode)
            writer.add_scalar("DQN/epsilon", agent.epsilon, episode)
            writer.flush()

        episode += 1
        