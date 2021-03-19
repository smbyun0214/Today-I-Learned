# -*- coding: utf-8 -*-

import os
import io

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from copy import deepcopy
from collections import deque


# 딥러닝 모델
# - 일반 네트워크와 타겟 네트워크 2가지 모델 이용
class LinearModel(nn.Module):

    def __init__(self, input_size, stack_frame, action_size):   
        super(LinearModel, self).__init__()

        self.input_size = input_size
        self.stack_frame = stack_frame
        self.action_size = action_size

        # 네트워크 모델: 3층의 은닉층으로 구성된 인공신경망
        self.fc1 = nn.Linear(self.input_size*self.stack_frame, 128)
        self.fc2 = nn.Linear(128, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, self.action_size)
    

    def forward(self, x):
        # 입력: 상태
        # 출력: 각 행동에 대한 Q 값
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class DQNAgent():

    def __init__(self, model, learning_rate=0.0001, epsilon_init=1.0, skip_frame=4, stack_frame=10, memory_maxlen=1000000):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = model
        self.target_model = deepcopy(self.model)
        self.update_target()

        if torch.cuda.is_available():
            self.model = model.cuda()
            self.target_model = self.target_model.cuda()
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.loss = F.mse_loss

        self.experience_memory = deque(maxlen=memory_maxlen)
        self.observation_set = deque(maxlen=skip_frame*stack_frame)

        self.epsilon = epsilon_init
        self.epsilon_min = 0.01

        self.skip_frame = skip_frame
        self.stack_frame = stack_frame


    def update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())
    

    def get_action(self, state):
        if self.model.training and self.epsilon >= np.random.sample():
            return np.random.randint(0, self.model.action_size)
        
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            Q = self.model(state)
            return np.argmax(Q.cpu().detach().numpy())


    def reset(self, observation):
        self.observation_set.clear()
        for i in range(self.skip_frame * self.stack_frame):
            self.observation_set.append(observation)


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

        state_batch         = torch.FloatTensor(np.stack([self.experience_memory[idx][0] for idx in batch_indices], axis=0))
        action_batch        = torch.FloatTensor(np.stack([self.experience_memory[idx][1] for idx in batch_indices], axis=0))
        reward_batch        = torch.FloatTensor(np.stack([self.experience_memory[idx][2] for idx in batch_indices], axis=0))
        next_state_batch    = torch.FloatTensor(np.stack([self.experience_memory[idx][3] for idx in batch_indices], axis=0))
        done_batch          = torch.FloatTensor(np.stack([self.experience_memory[idx][4] for idx in batch_indices], axis=0))

        eye = torch.eye(self.model.action_size)
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


    def model_save(self, episode, comment=None):
        script_dir = os.path.dirname(__file__)

        if comment: 
            dir_path = os.path.join(script_dir, "save_dqn_{}".format(comment))
        else:
            dir_path = os.path.join(script_dir, "save_dqn")

        if not os.path.isdir(dir_path):
            os.mkdir(dir_path)

        save_path = os.path.join(dir_path, "main_model_{:06}.pth".format(episode))

        # Python2
        buffer = io.BytesIO()
        torch.save(self.model.state_dict(), buffer, _use_new_zipfile_serialization=False)
        with open(save_path, "wb") as f:
            f.write(buffer.getvalue())


    def model_load(self, episode, comment= None, eval=True):
        script_dir = os.path.dirname(__file__) 

        if comment: 
            dir_path = os.path.join(script_dir, "save_dqn_{}".format(comment))
        else:
            dir_path = os.path.join(script_dir, "save_dqn")

        load_path = os.path.join(dir_path, "main_model_{:06}.pth".format(episode))
        
        # Python2
        with open(load_path, "rb") as f:
            buffer = io.BytesIO(f.read())
            self.model.load_state_dict(torch.load(buffer, map_location=self.device))
            self.model.to(self.device)

        if eval:
            self.model.eval()
        else:
            self.update_target()
