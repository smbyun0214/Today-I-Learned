#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
from std_msgs.msg import Int32MultiArray                # 초음파 센서 메시지
from xycar_motor.msg import xycar_motor as XycarMotor   # xycar 제어 메시지
from dqn import *


ultrasonic_data = None


def callback_ultrasonic(msg):
    global ultrasonic_data
    ultrasonic_data = np.array(msg.data)



if __name__ == '__main__':
    rospy.init_node("driver")
    sub = rospy.Subscriber("/xycar_ultrasonic", Int32MultiArray, callback_ultrasonic, queue_size=1)
    pub = rospy.Publisher("/xycar_motor", XycarMotor, queue_size=1)
    rate = rospy.Rate(10)

    episode = 1417
    max_steering_deg = 30.0

    net = DQN(5, 3, [256, 512, 256], skip_frame=2, stack_frame=5)
    net.model_load(episode)

    msg = XycarMotor()

    while ultrasonic_data is None:
        continue

    obs = ultrasonic_data[:5]

    for i in range(net.skip_frame*net.stack_frame):
        net.observation_set.append(obs)

    while not rospy.is_shutdown():
        # 상태 생성
        obs = ultrasonic_data[:5]
        net.observation_set.append(obs)
        state = net.skip_stack_frame(obs)

        # 행동 선택
        action = net.get_action(state, 0)


        # 조향각 조정
        if action == 0:
            steering_deg = -max_steering_deg
        elif action == 1:
            steering_deg = 0
        else:
            steering_deg = max_steering_deg

        msg.speed = 35
        msg.angle = steering_deg
        pub.publish(msg)
        