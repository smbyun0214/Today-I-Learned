from dqn import *
from simulator.viewer import *
from simulator.car import *


def get_random_start_pos():
    return np.random.randint(100, 200), 100

def get_random_start_yaw():
    return np.random.uniform(np.pi/5*2, np.pi/5*3)


skip_frame = 4
stack_frame = 10

run_episode = 10000
save_episode = 10

update_target_episode = 100

epsilon_decay = 0.0001
learning_rate = 0.0001
discount_factor = 0.99
batch_size = 32

model = NN(5, stack_frame, 3)
target_model = NN(5, stack_frame, 3)

car = Car()
agent = DQNAgent(model, target_model, learning_rate)
writer = SummaryWriter()

map_origin = cv.imread("simulator/map/rally_map4.png")
goal_pixel = [255, 51, 4]

episode_cnt = 0

while episode_cnt < run_episode:
    step_losses = []
    step_max_qs = []

    is_episode_done = 0
    episode_rewards = 0

    car.position = get_random_start_pos()
    car.yaw = get_random_start_yaw()
    car.steering_deg = 0

    ultrasonic_start_points, ultrasonic_end_points, _ = get_ultrasonic_distance(map_origin, car)
    distances = np.sqrt(np.sum((ultrasonic_start_points - ultrasonic_end_points)**2, axis=1))
    distances_meter = rint(distances * car.meter_per_pixel * 100)

    agent.observation_set.clear()
    for i in range(skip_frame * stack_frame):
        agent.observation_set.append(distances_meter)

    state = agent.skip_stack_frame(distances_meter)

    while not is_episode_done:
        action = agent.get_action(state)

        # 조향각 조정
        if action == 0:
            steering_deg = -car.max_steering_deg
        elif action == 1:
            steering_deg = 0
        else:
            steering_deg = car.max_steering_deg

        car.update(1/30, car.DRIVE, steering_deg)

        car_status = is_collision(map_origin, car, goal_pixel)

        # 차량 충돌
        if car_status == -1:
            is_episode_done = 1
            reward = -100
        # 차량 도착
        elif car_status == 1:
            is_episode_done = 1
            reward = 100
            print("도착:", episode)
        else:
            reward = 1

        ultrasonic_start_points, ultrasonic_end_points, _ = get_ultrasonic_distance(map_origin, car)
        distances = np.sqrt(np.sum((ultrasonic_start_points - ultrasonic_end_points)**2, axis=1))
        distances_meter = rint(distances * car.meter_per_pixel * 100)
        
        next_state = agent.skip_stack_frame(distances_meter)

        agent.append_sample(state, action, reward, next_state, is_episode_done)

        loss, maxQ = agent.train_model(discount_factor, batch_size)


        if agent.epsilon > agent.epsilon_min:
            agent.epsilon -= epsilon_decay

        step_losses.append(loss)
        step_max_qs.append(maxQ)

        episode_rewards += reward
        state = next_state

    # 타겟 네트워크 업데이트
    # 일정 스텝마다 타겟 네트워크 업데이트 수행
    if episode_cnt % update_target_episode == 0:
        agent.update_target()
    
    # 모델 저장
    if episode_cnt % save_episode == 0:
        agent.model_save(episode_cnt)

    writer.add_scalar("DQN/loss", np.mean(step_losses), episode_cnt)
    writer.add_scalar("DQN/max_q", np.mean(step_max_qs), episode_cnt)
    writer.add_scalar("DQN/reward", episode_rewards, episode_cnt)
    writer.add_scalar("DQN/epsilon", agent.epsilon, episode_cnt)
    writer.flush()

    episode_cnt += 1
