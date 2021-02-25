# 라이브러리 불러오기
import numpy as np
import random
import gym

# 환경 정의
env = gym.make("Taxi-v3")   # Taxi-v2 환경을 env로 정의

# 파라미터 설정
action_size = env.action_space.n    # actino의 수: 6

discount_factor = 0.9   # 감가율
learning_rate = 0.1     # 학습률

run_step = 500000       # 학습 진행 스텝
test_step = 10000       # 학습 이후 테스트 진행 스텝

print_episode = 100     # 해당 스텝마다 한번씩 진행 상황 출력

epsilon_init = 1.0      # 초기 epsilon
epsilon_min = 0.1       # 최소 epsilon

train_mode = True       # 학습 모드



# Q-Agent Class: Q-Learning 관련 함수들을 정의
class Q_Agent():
    def __init__(self):
        self.Q_table = {}               # Q-table을 dictionary로 초기화
        self.epsilon = epsilon_init     # epsilon 값을 epsilon_init으로 초기화


    # 만약 Q-table 내에 상태 정보가 없으면 Q-table에서 해당 상태 초기화
    def init_Q_table(self, state):
        if state not in self.Q_table:
            self.Q_table[state] = np.zeros(action_size)
    

    # Epsilon greedy에 따라 행동 결정
    def get_action(self, state):
        if self.epsilon > np.random.rand():
            # 랜덤 행동 결정
            return np.random.randint(0, action_size)

        self.init_Q_table(state)

        # Q-table 기반으로 행동 결정
        return np.argmax(self.Q_table[state])
    

    # 학습 수행
    def train_model(self, state, action, reward, next_state, done):
        self.init_Q_table(state)
        self.init_Q_table(next_state)

        # 타겟값 계산 및 Q-table 업데이트
        target = reward + discount_factor*np.max(self.Q_table[next_state])
        Q_val = self.Q_table[state][action]

        if done:
            self.Q_table[state][action] = (1 - learning_rate)*Q_val + learning_rate*reward
        else:
            self.Q_table[state][action] = (1 - learning_rate)*Q_val + learning_rate*target

    
# Main 함수
if __name__ == "__main__":
    # Q_Agent 클래스 초기화
    agent = Q_Agent()

    # 스텝, 에피소드, 보상을 저장할 리스트 초기화
    step = 0
    episode = 0
    reward_list = []

    # 게임 진행 반복문
    while step < run_step + test_step:
        # 상태, 에피소드 동안의 보상, 게임 종류 여부 초기화
        state = str(env.reset())
        episode_rewards = 0
        done = False

        # 에피소드 진행을 위한 반복문
        while not done:
            if step >= run_step:
                train_mode = False
                # env.render()
            
            # 행동 결정
            action = agent.get_action(state)

            # 다음 상태, 보상, 게임 종료 정보 취득
            next_state, reward, done, _ = env.step(action)
            next_state = str(next_state)
            episode_rewards += reward

            # 학습 모드인 경우 Q-table 업데이트
            if train_mode:
                # epsilon 감소
                if agent.epsilon > epsilon_min:
                    agent.epsilon -= 1 / run_step
                
                # 학습 수행
                agent.train_model(state, action, reward, next_state, done)
            else:
                agent.epsilon = 0.0     # 학습된 대로 행동을 결정
            
            state = next_state
            step += 1
        
        reward_list.append(episode_rewards)
        episode += 1
    
        if episode != 0 and episode % print_episode == 0:
            print("Step: {} | Episode: {} | Epsilon: {:.3f} | Mean Rewards: {:.3f}".format(
                step, episode, agent.epsilon, np.mean(reward_list)))
            reward_list = []
    
    env.close()