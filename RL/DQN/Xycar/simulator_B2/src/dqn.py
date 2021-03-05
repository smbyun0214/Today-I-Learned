#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import io
import dill
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict, deque
from copy import deepcopy

torch.manual_seed(42)

class DQN(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, learning_rate=0.01, memory_size=100000, stack_frame=10, skip_frame=5):
        super(DQN, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.skip_frame = skip_frame
        self.stack_frame = stack_frame

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.experience_memory = deque(maxlen=memory_size)
        self.observation_set = deque(maxlen=skip_frame*stack_frame)

        """
        모델 정보 설정
        """
        # input layer
        layer_dict = OrderedDict()
        input_unit_size = self.input_size * self.stack_frame
        layer_dict["input"] = nn.Linear(input_unit_size, hidden_size[0])

        for idx, (hidden0, hidden1) in enumerate(zip(hidden_size[:-1], hidden_size[1:]), start=1):
            # activation function
            name = "Relu_{}".format(idx)
            layer = nn.ReLU()
            layer_dict[name] = layer

            # next layer
            name = "hidden_{}".format(idx)
            layer = nn.Linear(hidden0, hidden1)
            layer_dict[name] = layer
        
        name = "Relu_{}".format(len(hidden_size))
        layer = nn.ReLU()
        layer_dict[name] = layer

        # output layer
        layer_dict["output"] = nn.Linear(hidden_size[-1], output_size)

        """
        네트워크 구성
        """
        self.model = nn.Sequential(layer_dict).to(self.device)
        self.target_model = nn.Sequential(layer_dict).to(self.device)
        self.update_target_model()
        print(self.model)

        """
        최적화기/Loss 설정
        """
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.loss = F.smooth_l1_loss

    
    def forward(self, X):
        return self.model(X).to(self.device)
    

    def get_action(self, state, epsilon=0.0):
        if np.random.sample() >= epsilon:
            state = torch.from_numpy(state).float().to(self.device)
            output = self.forward(state)
            return output.argmax().item()
        else:
            return np.random.randint(0, self.output_size)


    def skip_stack_frame(self, observation):
        self.observation_set.append(observation)

        state = np.zeros([self.input_size * self.stack_frame])

        for i in range(self.stack_frame):
            state[self.input_size*i:self.input_size*(i+1)] = self.observation_set[-1-(self.skip_frame * i)]
        
        return state


    def append_sample(self, state, action, reward, next_state, done):
        self.experience_memory.append((state, action, reward, next_state, done))


    def train_model(self, discount_factor, batch_size):
        batch_indices = np.random.choice(len(self.experience_memory), min(batch_size, len(self.experience_memory)), replace=False)
        state, action, reward, next_state, done = [], [], [], [], []

        for idx in batch_indices:
            _state, _action, _reward, _next_state, _done = self.experience_memory[idx]
            state.append(_state)
            action.append([_action])
            reward.append(_reward)
            next_state.append(_next_state)
            done.append(_done)

        state = torch.tensor(state, dtype=torch.float).to(self.device)
        action = torch.tensor(action).to(self.device)
        reward = torch.tensor(reward).to(self.device)
        next_state = torch.tensor(next_state, dtype=torch.float).to(self.device)
        done_mask = torch.tensor(done).to(self.device)

        eye = torch.eye(self.output_size).to(self.device)
        one_hot_action = eye[action.view(-1).long()]
        q = (self.model(state) * one_hot_action).sum(1)

        with torch.no_grad():
            max_q = torch.max(q).item()
            next_q = self.target_model(next_state)
            target_q = reward + next_q.max(1).values*(discount_factor*(1 - done_mask))

        loss = self.loss(q, target_q).to(self.device)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.data

        
    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())
    

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

        if self.eval:
            self.model.eval()
        else:
            self.model.train()

        # with open(load_path, "rb") as f:
        #     buffer = io.BytesIO(f.read())
        #     self.model.load_state_dict(torch.load(buffer, map_location=self.device))
        #     self.target_model.load_state_dict(self.model.state_dict())

    


class DDQN(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, learning_rate=0.01, memory_size=100000, stack_frame=10, skip_frame=5):
        super(DDQN, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.skip_frame = skip_frame
        self.stack_frame = stack_frame

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.experience_memory = deque(maxlen=memory_size)
        self.observation_set = deque(maxlen=skip_frame*stack_frame)

        """
        모델 정보 설정
        """
        # input layer
        layer_dict = OrderedDict()
        input_unit_size = self.input_size * self.stack_frame
        layer_dict["input"] = nn.Linear(input_unit_size, hidden_size[0])

        for idx, (hidden0, hidden1) in enumerate(zip(hidden_size[:-1], hidden_size[1:]), start=1):
            # activation function
            name = "Relu_{}".format(idx)
            layer = nn.ReLU()
            layer_dict[name] = layer

            # next layer
            name = "hidden_{}".format(idx)
            layer = nn.Linear(hidden0, hidden1)
            layer_dict[name] = layer
        
        name = "Relu_{}".format(len(hidden_size))
        layer = nn.ReLU()
        layer_dict[name] = layer

        # output layer
        layer_dict["output"] = nn.Linear(hidden_size[-1], output_size)

        """
        네트워크 구성
        """
        self.model = nn.Sequential(layer_dict).to(self.device)
        self.target_model = nn.Sequential(layer_dict).to(self.device)
        self.update_target_model()
        print(self.model)

        """
        최적화기/Loss 설정
        """
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.loss = F.smooth_l1_loss

    
    def forward(self, X):
        return self.model(X).to(self.device)
    

    def get_action(self, state, epsilon=0.0):
        if np.random.sample() >= epsilon:
            state = torch.from_numpy(state).float().to(self.device)
            output = self.forward(state)
            return output.argmax().item()
        else:
            return np.random.randint(0, self.output_size)


    def skip_stack_frame(self, observation):
        self.observation_set.append(observation)

        state = np.zeros([self.input_size * self.stack_frame])

        for i in range(self.stack_frame):
            state[self.input_size*i:self.input_size*(i+1)] = self.observation_set[-1-(self.skip_frame * i)]
        
        return state


    def append_sample(self, state, action, reward, next_state, done):
        self.experience_memory.append((state, action, reward, next_state, done))


    def train_model(self, discount_factor, batch_size):
        batch_indices = np.random.choice(len(self.experience_memory), min(batch_size, len(self.experience_memory)), replace=False)
        state, action, reward, next_state, done = [], [], [], [], []

        for idx in batch_indices:
            _state, _action, _reward, _next_state, _done = self.experience_memory[idx]
            state.append(_state)
            action.append([_action])
            reward.append(_reward)
            next_state.append(_next_state)
            done.append(_done)

        state = torch.tensor(state, dtype=torch.float).to(self.device)
        action = torch.tensor(action).to(self.device)
        reward = torch.tensor(reward).to(self.device)
        next_state = torch.tensor(next_state, dtype=torch.float).to(self.device)
        done_mask = torch.tensor(done).to(self.device)

        eye = torch.eye(self.output_size).to(self.device)
        one_hot_action = eye[action.view(-1).long()]
        q = (self.model(state) * one_hot_action).sum(1)

        with torch.no_grad():
            next_action_q = self.model(next_state).max(1)[1].unsqueeze(1)
            target_q = reward + self.target_model(next_state).gather(1, next_action_q).flatten()*(discount_factor*(1 - done_mask))

        loss = self.loss(q, target_q).to(self.device)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.data
    

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())
    

    def model_save(self, episode):
        script_dir = os.path.dirname(__file__) 
        dir_path = os.path.join(script_dir, "save_ddqn")
        if not os.path.isdir(dir_path):
            os.mkdir(dir_path)
        save_path = os.path.join(script_dir, "save_ddqn", "main_model_{:06}.pth".format(episode))

        buffer = io.BytesIO()
        torch.save(self.model.state_dict(), buffer, pickle_module=dill, _use_new_zipfile_serialization=False)
        with open(save_path, "wb") as f:
            f.write(buffer.getvalue())


    def model_load(self, episode, eval=True):
        script_dir = os.path.dirname(__file__) 
        load_path = os.path.join(script_dir, "save_ddqn", "main_model_{:06}.pth".format(episode))
        
        # Python3
        self.model.load_state_dict(torch.load(load_path, map_location=self.device))
        self.target_model.load_state_dict(self.model.state_dict())

        if self.eval:
            self.model.eval()
        else:
            self.model.train()
