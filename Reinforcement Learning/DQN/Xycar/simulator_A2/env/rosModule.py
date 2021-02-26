#!/usr/bin/env python

import rospy
import numpy as np
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped

class rosmodule:

    laser_msg = None
    
    angle = 0.0
    speed = 0.0

    ack_msg = AckermannDriveStamped()
    ack_msg.header.frame_id = 'odom'

    fake_lidar_msg = LaserScan()
    fake_lidar_msg.header.frame_id = "laser"
    fake_lidar_msg.header.seq = 0
    fake_lidar_msg.angle_min = -3.12413907051
    fake_lidar_msg.angle_max = 3.14159274101
    fake_lidar_msg.angle_increment = 0.0174532923847
    fake_lidar_msg.scan_time = 0.075
    fake_lidar_msg.time_increment = fake_lidar_msg.scan_time/359
    fake_lidar_msg.range_min = 0.15000000596
    fake_lidar_msg.range_max = 12.0

    data_cnt = -1

    def __init__(self, node_name):
        rospy.init_node(node_name, anonymous = True)
        
        rospy.Subscriber('/scan', LaserScan, self.lidar_callback)
        rospy.Subscriber('/ackermann_cmd', AckermannDriveStamped, self.motor_callback)

        self.ackerm_publisher = rospy.Publisher('ackermann_cmd', AckermannDriveStamped, queue_size=1)
        self.fake_lidar_publisher = rospy.Publisher('/scan', LaserScan, queue_size=1)

    def auto_drive(self, steer_val, car_run_speed):
        self.ack_msg.header.stamp = rospy.Time.now()
        self.ack_msg.drive.steering_angle = steer_val
        self.ack_msg.drive.speed = car_run_speed
        self.ackerm_publisher.publish(self.ack_msg)

    def pub_fake_lidar_data(self, state):
        if len(state) != self.data_cnt:
            num = 360-len(state)
            self.rtn = [0.0 for i in range(num)]
            self.data_cnt = len(state)
        self.fake_lidar_msg.ranges = state + self.rtn
        self.fake_lidar_msg.header.seq += 1
        self.fake_lidar_msg.header.stamp = rospy.Time.now()
        self.fake_lidar_publisher.publish(self.fake_lidar_msg)
        
    def motor_callback(self, data):
        self.angle = max(-30.0, min(np.degrees(data.drive.steering_angle), 30.0))
        self.speed = min(max(data.drive.speed, -0.3), 0.3)

    def lidar_callback(self, data):
        self.laser_msg = data.ranges

    def get_laser_msg(self):
        return self.laser_msg
        
    def get_motor_data(self):
        return self.angle, self.speed

    def get_ros_shutdown_chk(self):
        return not rospy.is_shutdown()
    
    