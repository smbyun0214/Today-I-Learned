#!/usr/bin/env python

import rospy
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped

class rosmodule:
    
    laser_msg = None
    ack_msg = AckermannDriveStamped()
    ack_msg.header.frame_id = "odom"


    def __init__(self):
        rospy.init_node("dqn2xycar", anonymous=True)
        self.launch_data_read()

        rospy.Subscriber("/scan", LaserScan, self.lidar_callback)
        self.ackerm_publisher = rospy.Publisher("ackermann_cmd", AckermannDriveStamped, queue_size=1)


    def auto_drive(self, steer_val, car_run_speed):
        self.ack_msg.header.stamp = rospy.Time.now()
        self.ack_msg.drive.steering_angle = steer_val
        self.ack_msg.drive.speed = car_run_speed
        self.ackerm_publisher.publish(self.ack_msg)
    
    
    def lidar_callback(self, data):
        self.laser_msg = data.ranges
    
    
    def launch_data_read(self):
        self.hidden_size = []

        hidden_size_str = rospy.get_param("~hidden_size", "[]")
        self.view_epi = rospy.get_param("~view_epi", "0")
        self.output_size = rospy.get_param("~output_size", 0)
        self.LoadPath_main = rospy.get_param("~loadPath", "")

        hidden_size_str_list = hidden_size_str.replace("[", "").replace("]", "").split(",")
        for i in hidden_size_str_list:
            self.hidden_size.append(int(i))
    
    
    def get_laser_msg(self):
        return self.laser_msg
    

    def get_view_epi(self):
        return self.view_epi
    

    def get_output_size(self):
        return self.output_size
    

    def get_pth_path(self):
        return self.LoadPath_main
    

    def get_hidden_size(self):
        return self.hidden_size


    def get_ros_shutdown_chk(self):
        return not rospy.is_shutdown()