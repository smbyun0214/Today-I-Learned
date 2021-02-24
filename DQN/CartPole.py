import numpy as np
import random
import datetime, os     # 학습된 모델을 저장하기 위해 사용
from collections import deque       # 리스트와 유사하게 데이터를 저장
                                    # 최대 길이를 넘어서 데이터가 저장되는 경우,
                                    # 가장 오래된 데이터부터 자동으로 삭제
import gym

import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn

# Environment
env = gym.make("CartPole-v0")

# Parameter Setting
algorithm = "DQN"

state_size = 4                      # 상태의 수 정의
action_size = env.action_space.n     # 행동의 수 정의

load_model = True       # 학습된 모델을 불러올지 결정
train_mode = False      # 학습을 수행할지 결정

batch_size = 32         # 한 스텝당 batch size의 수 만큼 데이터를 이용하여 학습
mem_maxlen = 10000      # Replay Memory의 최대 길이

skip_frame = 1          # 몇 개의 frame을 skip할지 결정
stack_frame = 1         # 몇 개의 frame을 stack할지 결정

start_train_step = 10000    # Replay memory에 일정 개수 이상 데이터를 채우고 학습 수행
run_step = 50000            # 학습을 진행할 스텝
test_step = 10000           # 학습 후 테스트를 진행할 스텝

target_update_step = 1000   # target network를 업데이트하는 스텝
print_episode = 10      # 해당 에피소드마다 한번씩 진행 상황 출력
save_step = 20000       # 해당 스텝마다 네트워크 모델 저장

epsilon_init = 1.0      # 초기 epsilon
epsilon_min = 0.1       # epsilon의 최소값

discount_factor = 0.99
learning_rate = 0.00025

date_time = datetime.datetime.now().strftime("%Y%m%d-%H-%M-%S")

save_path = "./saved_models/" + date_time
load_path = "./saved_models/20210224-23-14-51DQN"

# 딥러닝 연산을 위한 device 결정
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# 딥러닝 모델
class DQN(nn.Module):
    def __init__(self, network_name):   # 일반 네트워크와 타겟 네트워크 2가지 모델 이용
        super(DQN, self).__init__()
        input_Size = state_size * stack_frame

        # 네트워크 모델: 3층의 은닉층으로 구성된 인공신경망
        self.fc1 = nn.Linear(input_Size, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, action_size)
    

    def forward(self, x):
        # 입력: 상태
        # 출력: 각 행동에 대한 Q 값
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x



# DQNAgent 클래스: DQN 알고리즘을 위한 다양한 함수 정의
class DQNAgent():
    def __init__(self, model, target_model, optimizer):
        # 클래스의 함수들을 위한 값 설정
        self.model = model                  # 네트워크 모델 정의
        self.target_model = target_model    # 타겟 모델 정의
        self.optimizer = optimizer          # 최적화기 정의
        
        self.memory = deque(maxlen=mem_maxlen)              # Replay memory 정의
        self.obs_set = deque(maxlen=skip_frame*stack_frame) # 상태를 stack하기 위핸 obs_set

        self.epsilon = epsilon_init

        self.update_target()    # 타겟 네트워크 업데이트

        if load_model == True:
            self.model.load_state_dict(torch.load(load_path+"/model.pth"))
            print("Model is loaded from {}".format(load_path+"/model.pth"))
    

    # Epsilon greedy 기법에 따라 행동 결정
    def get_action(self, state):
        if train_mode:
            if self.epsilon > np.random.rand():
                # 랜덤하게 행동 결정
                return np.random.randint(0, action_size)
        
        with torch.no_grad():
            # 네트워크 연산에 따라 행동 결정
            Q = self.model(torch.FloatTensor(state).unsqueeze(0).to(device))
            return np.argmax(Q.cpu().detach().numpy())


    # 상태에 대해 frame skipping과 frame stacking을 수행
    def skip_stack_frame(self, obs):
        self.obs_set.append(obs)    # obs_set에 상태 추가

        state = np.zeros([state_size * stack_frame])    # 상태를 stack할 빈 array 생성

        # skip frame마다 한번씩 obs를 stacking
        # obs_set 내부에서 frame skipping을 수행하면 설정값 만큼 frame stacking 수행
        for i in range(stack_frame):
            state[state_size*i:state_size*(i+1)] = self.obs_set[-1-(skip_frame * i)]
        
        # 상태를 stack하여 최종 상태로 반환
        return state


    # Replay memory에 데이터 추가(상태, 행동 보상, 다음 상태, 게임 종료 여부)
    def append_sample(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))


    # 네트워크 모델 저장
    def save_model(self, load_model, train_mode):
        # 처음 모델을 학습하는 경우, 폴더 생성 및 save_path에 모델 저장
        if not load_model and train_mode:   # first training
            os.makedirs(save_path + algorithm, exist_ok=True)
            torch.save(self.model.state_dict(), save_path + algorithm + "/model.pth")
            print("Save Model: {}".format(save_path + algorithm))
        # 기존에 학습된 모델을 불러온 뒤 이어서 학습하는 경우, load_path에 모델 저장
        elif load_model and train_mode:     # additional training
            torch.save(self.model.state_dict(), load_path + "/model.pth")
            print("Save Model: {}".format(load_path))

    
    # 학습 수행을 진행하는 train_model 함수
    # 하나의 데이터가 아닌 batch 데이터를 통해 학습을 수행
    def train_model(self):
        # 학습을 위한 미니 배치 데이터 샘플링
        batch = random.sample(self.memory, min(len(self.memory), batch_size))  # memory에서 batch_size의 수 만큼 데이터 취득

        # batch에서 각각의 정보를 얻은 후 이를 array로 쌓아둠
        # Torch의 tensor로 변환하여 device에 올려줌
        state_batch         = torch.FloatTensor(np.stack([b[0] for b in batch], axis=0)).to(device)
        action_batch        = torch.FloatTensor(np.stack([b[1] for b in batch], axis=0)).to(device)
        reward_batch        = torch.FloatTensor(np.stack([b[2] for b in batch], axis=0)).to(device)
        next_state_batch    = torch.FloatTensor(np.stack([b[3] for b in batch], axis=0)).to(device)
        done_batch          = torch.FloatTensor(np.stack([b[4] for b in batch], axis=0)).to(device)

        # 실제 에이전트가 취한 행동에 대한 Q값 도출
        # Action = [1, 2, 0] ---> One hot action = [0, 1, 0], [0, 0, 1], [1, 0, 0]
        # Q 값                   One hot action
        # [[0.1  0.4  0.7],     [[0  1  0],         [[ 0  0.4  0],                 [[0.4],
        #  [0.3  0.2  0.4],  *   [0  0  1],     =    [ 0   0  0.4],  -- sum(1) -->  [0.4],
        #  [0.8  0.1  0.3]]      [1  0  0]]          [0.8  0   0]]                  [0.8]]
        eye = torch.eye(action_size).to(device)
        one_hot_action = eye[action_batch.view(-1).long()]
        q = (self.model(state_batch) * one_hot_action).sum(1)

        # 타겟의 식에 따라 타겟값 계산
        with torch.no_grad():
            max_Q = torch.max(q).item()
            next_q = self.target_model(next_state_batch)    # 다음 상태의 Q값은 타겟 네트워크를 통해 도출
            target_q = reward_batch + next_q.max(1).values*(discount_factor*(1 - done_batch))   # 모든 batch 데이터에 대한 타겟값 계산
        
        # 예측값(q)와 타겟값(target_q) 사이의 손실 함수값 계산(smooth L1 Loss)
        loss = F.smooth_l1_loss(q, target_q)

        # 인공신경망 학습
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # loss와 Q값 반환
        return loss.item(), max_Q


    # 타겟 네트워크 업데이트
    def update_target(self):
        # 일반 네트워크의 모든 변수들을 타겟 네트워크에 복제
        self.target_model.load_state_dict(self.model.state_dict())
    

# 메인 함수
if __name__ == "__main__":
    model = DQN("main").to(device)          # 네트워크 모델 정의 후 device 할당
    target_model = DQN("target").to(device)  # 타겟 네트워크 정의
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)    # 최적화기 정의(Adam optimizer)

    agent = DQNAgent(model, target_model, optimizer)  # DQNAgeng class를 agent로 설정

    model.train()   # 딥러닝 네트워크를 학습 모드로 설정

    # 스텝, 에피소드, reward_list, loss_list, max_Q_list 초기화
    step = 0
    episode = 0
    reward_list = []
    loss_list = []
    max_Q_list = []

    # 게임 진행을 위한 반복문
    while step < run_step + test_step:
        # 상태, episode, reward, done 정보 초기화
        obs = env.reset()   # 단일 상태를 obs로 저장
        episode_rewards = 0
        done = False

        # 에피소드를 처음 시작하는 경우 obs_set을 동일 obs로 skip_frame * stack_frame 수 만큼 채워줌
        for i in range(skip_frame * stack_frame):
            agent.obs_set.append(obs)
        
        state = agent.skip_stack_frame(obs) # frame_skipping & stacking 수행 후 최종 상태로 사용

        # 에피소드를 위한 반복문
        while not done:
            # 학습을 run step까지 진행하여 학습이 끝난 경우,
            # 학습 모드를 False로 설정하고 네트워크를 검증 모드로 설정
            if step == run_step:
                train_mode = False
                model.eval()
            
            # 행동 결정
            action = agent.get_action(state)

            # 다음 상태, 보상, 게임 종료 여부 정보 취득
            next_obs, reward, done, _ = env.step(action)

            # episode_rewards에 보상을 더해줌
            episode_rewards += reward

            # 다음 상태도 frame skipping & stacking 수행 후 next state로 사용
            next_state = agent.skip_stack_frame(next_obs)

            # 학습의 안정화를 위채 추가한 코드
            # - 카트가 최대한 중앙에서 벗어나지 않도록 학습
            # - 카트가 중앙에서 벗어날수록 패널티 부여
            # - 폴이 쓰러지지 않더라도, 카트가 중앙에서 너무 멀어지면서 학습이 제대로 수행되지 않는 경우 발생
            reward -= abs(next_obs[0])

            # 학습 모드인 경우, Replay memory에 경험 데이터 저장
            if train_mode:
                agent.append_sample(state, action, reward, next_state, done)
            # 검증 모드인 경우, epsilon을 0으로 설정
            else:
                agent.epsilon = 0.0
                # 해당 라인의 주석을 해제하면 학습이 끝나고 검증 수행시 게임 화면을 확인할 수 있음
                env.render()
                
            if train_mode:
                # Epsilon 감소
                if agent.epsilon > epsilon_min:
                    agent.epsilon -= 1 / run_step
                
                # 모델 학습
                loss, maxQ = agent.train_model()
                loss_list.append(loss)
                max_Q_list.append(maxQ)
            
                # 모델 저장
                if step % save_step == 0 and step != 0 and train_mode:
                    agent.save_model(load_model, train_mode)

                # 타겟 네트워크 업데이트
                # 일정 스텝마다 타겟 네트워크 업데이트 수행
                if step % target_update_step == 0:
                    agent.update_target()


            # 상태 및 스텝 정보 업데이트
            state = next_state
            step += 1
                
        reward_list.append(episode_rewards)
        episode += 1

        # 진행상황 출력
        if episode % print_episode == 0 and episode != 0:
            print("step: {} | episode: {} | reward: {:.2f} | loss: {:.4f} | maxQ: {:.2f} | epsilon: {:.4f}".format(
                step, episode, np.mean(reward_list), np.mean(loss_list), np.mean(max_Q_list), agent.epsilon))
            
            reward_list = []
            loss_list = []
            max_Q_list = []
        
    # 학습 및 검증 종료 이후 모델 저장
    agent.save_model(load_model, train_mode)
    env.close()
