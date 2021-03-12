from dqn import *
from simulator.viewer import *
from simulator.car import *


def get_random_start_pos():
    return np.random.randint(100, 200), np.random.randint(600, 750)

def get_random_start_yaw():
    return np.random.uniform(np.pi/5*2, np.pi/5*3)


skip_frame = 4
stack_frame = 10

run_episode = 10000
save_episode = 10

update_target_episode = 100

epsilon_decay = 0.00001
learning_rate = 0.0001
discount_factor = 0.99
batch_size = 32

max_step = 5000

model = NN(5, stack_frame, 3)
target_model = NN(5, stack_frame, 3)

car = Car()
agent = DQNAgent(model, target_model, learning_rate)
writer = SummaryWriter()

map_origin = cv.imread("simulator/map/rally_map2.png")

episode_cnt = 0
fps = 10

while episode_cnt < run_episode:
    step_losses = []
    step_max_qs = []

    is_episode_done = 0
    episode_rewards = 0
    step_cnt = 0

    car.reset()
    gear = car.BREAK

    car.position = get_random_start_pos()
    car.yaw = get_random_start_yaw()
    car.steering_deg = 0

    ultrasonic_start_points, ultrasonic_end_points, _ = get_ultrasonic_distance(map_origin, car)
    distances = np.sqrt(np.sum((ultrasonic_start_points[:5] - ultrasonic_end_points[:5])**2, axis=1))
    distances_meter = rint(distances * car.meter_per_pixel * 100)

    agent.observation_set.clear()
    for i in range(skip_frame * stack_frame):
        agent.observation_set.append(distances_meter)

    state = agent.skip_stack_frame(distances_meter)

    while not is_episode_done:
        background = map_origin.copy()

        draw_car(background, car)
        draw_ultrasonic(background, car, map_origin)
        cv.imshow("simulator", background)
        cv.waitKey(10)

        action = agent.get_action(state)

        # 조향각 조정
        if action % 3 == 0:
            steering_deg = -car.max_steering_deg
        elif action % 3 == 1:
            steering_deg = 0
        elif action % 3 == 2:
            steering_deg = car.max_steering_deg
        
        car.update(1/fps, car.DRIVE, steering_deg)

        ultrasonic_start_points, ultrasonic_end_points, _ = get_ultrasonic_distance(map_origin, car)
        distances = np.sqrt(np.sum((ultrasonic_start_points[:5] - ultrasonic_end_points[:5])**2, axis=1))
        distances_meter = rint(distances * car.meter_per_pixel * 100)
        
        next_state = agent.skip_stack_frame(distances_meter)

        # 차량 충돌
        reward = 1
        if is_collision(map_origin, car) or np.min(distances_meter) <= 10:
            is_episode_done = 1
            reward = -10
        elif step_cnt >= max_step:
            print("Max Step 도달:", episode_cnt)
            is_episode_done = 1
            reward = 0

        agent.append_sample(state, action, reward, next_state, is_episode_done)

        loss, maxQ = agent.train_model(discount_factor, batch_size)

        if agent.epsilon > agent.epsilon_min:
            agent.epsilon -= epsilon_decay

        step_losses.append(loss.item())
        step_max_qs.append(maxQ)

        episode_rewards += reward
        state = next_state
        step_cnt += 1

    print(episode_cnt)

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
