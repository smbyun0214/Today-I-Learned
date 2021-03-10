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

agent.model_load(830)

map_origin = cv.imread("simulator/map/rally_map4.png")
goal_pixel = [255, 51, 4]

episode_cnt = 0
# 30fps = 1000/30
# fps = int(1000/30)
fps = 10

while episode_cnt < run_episode:
    is_episode_done = 0

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
        background = map_origin.copy()
        draw_car(background, car)
        draw_ultrasonic(background, car, map_origin)
        cv.imshow("simulator", background)
        
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

        if car_status != 0:
            is_episode_done = 1

        ultrasonic_start_points, ultrasonic_end_points, _ = get_ultrasonic_distance(map_origin, car)
        distances = np.sqrt(np.sum((ultrasonic_start_points - ultrasonic_end_points)**2, axis=1))
        distances_meter = rint(distances * car.meter_per_pixel * 100)
        
        next_state = agent.skip_stack_frame(distances_meter)

        state = next_state

        key = cv.waitKey(fps)
        if key == ord("q"):
            episode_cnt = run_episode
            break
            
    episode_cnt += 1
