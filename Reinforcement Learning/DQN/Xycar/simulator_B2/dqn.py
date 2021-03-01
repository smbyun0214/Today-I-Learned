import dill
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict, deque
from copy import deepcopy

torch.manual_seed(42)

class DQN(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, learning_rate=0.01, memory_size=100000):
        super(DQN, self).__init__()
        self.output_size = output_size

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.experience_memory = deque(maxlen=memory_size)

        """
        모델 정보 설정
        """
        # input layer
        layer_dict = OrderedDict()
        layer_dict["input"] = nn.Linear(input_size, hidden_size[0])

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
        self.target_model = nn.Sequential(deepcopy(layer_dict)).to(self.device)
        print(self.model)

        """
        최적화기/Loss 설정
        """
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.loss = F.smooth_l1_loss

    
    def forward(self, X):
        return self.model(X).to(self.device)
    

    def sample_action(self, observation, epsilon=0.0):
        if np.random.sample() >= epsilon:
            observation = torch.from_numpy(observation).float().to(self.device)
            output = self.forward(observation)
            return output.argmax().item()
        else:
            return np.random.randint(0, self.output_size)


    def learning(self, discount_factor, batch_size):
        batch_indices = np.random.choice(len(self.experience_memory), min(batch_size, len(self.experience_memory)), replace=False)
        state, action, reward, next_state, done = [], [], [], [], []

        for idx in batch_indices:
            _state, _action, _reward, _next_state, _done = self.experience_memory[idx]
            state.append(_state)
            action.append([_action])
            reward.append([_reward])
            next_state.append(_next_state)
            done.append([_done])

        state = torch.tensor(state, dtype=torch.float).to(self.device)
        action = torch.tensor(action).to(self.device)
        reward = torch.tensor(reward).to(self.device)
        next_state = torch.tensor(next_state, dtype=torch.float).to(self.device)
        done_mask = torch.tensor(done).to(self.device)

        out = self.model(state).gather(1, action)
        max_target_out = self.target_model(next_state).max(1)[0].unsqueeze(1)

        target = (reward + discount_factor*max_target_out*done_mask).to(self.device)

        loss = self.loss(out, target).to(self.device)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.data

        
    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())
    

    def model_save(self, episode):
        script_dir = os.path.dirname(__file__) 
        save_path = os.path.join(script_dir, "save", "main_model_{:06}.pth".format(episode))

        buffer = io.BytesIO()
        torch.save(self.model.state_dict(), buffer, pickle_module=dill)
        with open(save_path, "wb") as f:
            f.write(buffer.getvalue())


    def model_load(self, episode):
        script_dir = os.path.dirname(__file__) 
        load_path = os.path.join(script_dir, "save", "main_model_{:06}.pth".format(episode))

        with open(load_path, "rb") as f:
            buffer = io.BytesIO(f.read())
        self.model.load(torch.load(buffer, map_location=self.device))
        self.target_model.load_state_dict(self.model.state_dict)
    


class DDQN(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, learning_rate=0.01, memory_size=100000):
        super(DDQN, self).__init__()
        self.output_size = output_size

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.experience_memory = deque(maxlen=memory_size)

        """
        모델 정보 설정
        """
        # input layer
        layer_dict = OrderedDict()
        layer_dict["input"] = nn.Linear(input_size, hidden_size[0])

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
        self.target_model = nn.Sequential(deepcopy(layer_dict)).to(self.device)
        print(self.model)

        """
        최적화기/Loss 설정
        """
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.loss = F.smooth_l1_loss

    
    def forward(self, X):
        return self.model(X).to(self.device)
    

    def sample_action(self, observation, epsilon=0.0):
        if np.random.sample() >= epsilon:
            observation = torch.from_numpy(observation).float().to(self.device)
            output = self.forward(observation)
            return output.argmax().item()
        else:
            return np.random.randint(0, self.output_size)


    def learning(self, discount_factor, batch_size, learning_step=1):
        loss_list = []
        for _ in range(learning_step):
            batch_indices = np.random.choice(len(self.experience_memory), min(batch_size, len(self.experience_memory)), replace=False)
            state, action, reward, next_state, done = [], [], [], [], []

            for idx in batch_indices:
                _state, _action, _reward, _next_state, _done = self.experience_memory[idx]
                state.append(_state)
                action.append([_action])
                reward.append([_reward])
                next_state.append(_next_state)
                done.append([_done])

            state = torch.tensor(state, dtype=torch.float).to(self.device)
            action = torch.tensor(action).to(self.device)
            reward = torch.tensor(reward).to(self.device)
            next_state = torch.tensor(next_state, dtype=torch.float).to(self.device)
            done_mask = torch.tensor(done).to(self.device)

            out = self.model(state).gather(1, action)
            max_target_out = self.target_model(next_state).detach() \
                            .gather(1, self.model(next_state).detach().max(1)[1].unsqueeze(1))

            target = (reward + discount_factor*max_target_out*done_mask).to(self.device)

            loss = self.loss(out, target).to(self.device)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            loss_list.append(loss.data)

        return np.mean(loss_list)
    

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())
    

    def model_save(self, episode):
        script_dir = os.path.dirname(__file__) 
        save_path = os.path.join(script_dir, "save", "main_model_{:06}.pth".format(episode))

        buffer = io.BytesIO()
        torch.save(self.model.state_dict(), buffer, pickle_module=dill)
        with open(save_path, "wb") as f:
            f.write(buffer.getvalue())


    def model_load(self, episode):
        script_dir = os.path.dirname(__file__) 
        load_path = os.path.join(script_dir, "save", "main_model_{:06}.pth".format(episode))

        with open(load_path, "rb") as f:
            buffer = io.BytesIO(f.read())
        self.model.load(torch.load(buffer, map_location=self.device))
        self.target_model.load_state_dict(self.model.state_dict)
