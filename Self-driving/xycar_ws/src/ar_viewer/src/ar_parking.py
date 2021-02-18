#!/usr/bin/env python
#-*- coding: utf-8 -*-
import rospy, math
import cv2, time
import numpy as np
from ar_track_alvar_msgs.msg import AlvarMarkers        # AR Tag 거리/자세 정보 토픽의 메시지
from tf.transformations import euler_from_quaternion    # 쿼터니언 값을 오일러 값으로 변환
from std_msgs.msg import Int32MultiArray                # 자동차 구동 제어 토픽의 메시지

xycar_msg = Int32MultiArray()

# AR Tag의 거리정보와 자세 정보를 저장할 공간 마련
arData = {
    "DX": 0.0, "DY": 0.0, "DZ": 0.0,
    "AX": 0.0, "AY": 0.0, "AZ": 0.0, "AW": 0.0,
    "isUpdate": False}
roll, pitch, yaw = 0, 0, 0

# /ar_pose_marker 토픽을 받을때 마다 호출되는 콜백함수 정의
def callback(msg):
    global arData

    for i in msg.markers:
        # AR Tag의 위치 정보(3차원 벡터값, x, y, z)
        arData["DX"] = i.pose.pose.position.x
        arData["DY"] = i.pose.pose.position.y
        arData["DZ"] = i.pose.pose.position.z
        # AR Tag의 자세 정보(쿼터니언 값, roll, pitch, yaw)
        arData["AX"] = i.pose.pose.orientation.x
        arData["AY"] = i.pose.pose.orientation.y
        arData["AZ"] = i.pose.pose.orientation.z
        arData["AW"] = i.pose.pose.orientation.w

    # callback 함수가 호출되어 업데이트 되었음을 표시
    arData["isUpdate"] = True


# Stanley Contorl 적용
def stanley_control(dx, dy, yaw, v):
    k = 0.17

    # AR Tag와 차량의 사이각을 yaw_term으로 사용
    theta = np.arctan2(dx, dy)
    yaw_term = theta + yaw

    # 차량과 AR Tag가 바라보는 축 사이의 거리를 cte로 사용
    dist = np.sqrt(dx**2 + dy**2)
    cte = dist * np.sin(yaw)
    cte_term = np.arctan2(k*cte, v)

    # cte_term은 AR Tag가 바라보는 축에 대한 차량에 따라 부호가 달라지는데,
    # 이 때, AR Tag를 바라보는 방향으로 조향각을 적용하기 위해 - 부호를 곱함
    steer = -cte_term + yaw_term
    return steer

# AR Tag 상태 표시 윈도우 생성
def show_ar_tag_position(dx, dy, yaw):
    # 높이x폭 = 100x500 크기의 이미지 준비
    img = np.ones((100, 500, 3))

    # 빨간색으로 선 긋기
    img = cv2.line(img, (25, 65), (475, 65), (0, 0, 255), 2)
    img = cv2.line(img, (25, 40), (25, 90), (0, 0, 255), 3)
    img = cv2.line(img, (250, 40), (250, 90), (0, 0, 255), 3)
    img = cv2.line(img, (475, 40), (475, 90), (0, 0, 255), 3)

    # 녹색 원을 그릴 위치 계산하기
    # DX 값이 "0"일 때, 화면의 중앙 위치에 오도록 세팅하기 위해
    # DX + 250 = point 값으로 설정
    # 제일 작은 값을 25, 제일 큰 값을 475로 제한
    point = max(25, min(dx + 250, 475))
    point = int(point)
    img = cv2.circle(img, (point, 65), 15, (0, 255, 0), -1)

    # x, y 축 방향의 좌표값을 가지고, AR Tag까지의 거리를 계산
    # DX, DY 값을 가지고 거리 계산(직각삼각형)
    dist = np.sqrt(dx**2 + dy**2)

    # dx, dy, yaw, distance 값을 문자열로 만들고, 좌측 상단에 표시
    str_dxy_yaw = "dx:{: >7.2f}  dy:{: >7.2f}  yaw:{: >7.2f}".format(dx, dy, yaw)
    str_dist = "{:>10.2f}".format(dist)

    cv2.putText(img, str_dxy_yaw, (20, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
    cv2.putText(img, str_dist, (400, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
    cv2.imshow("AR Tag Position", img)


rospy.init_node("ar_drive")
rospy.Subscriber("ar_pose_marker", AlvarMarkers, callback)  # 토픽의 구독 준비
motor_pub = rospy.Publisher("xycar_motor_msg", Int32MultiArray, queue_size=1)   # xycar_motor_msg 토픽의 발행 준비

while not rospy.is_shutdown():
    if not arData["isUpdate"]:
        continue

    # AR Tag의 자세 정보를 담은 쿼터니언 값을 오일러 값으로 변환
    (roll, pitch, yaw) = euler_from_quaternion(
        (arData["AX"], arData["AY"], arData["AZ"], arData["AW"]))

    dx, dy = arData["DX"], arData["DY"]

    # AR Tag까지 거리가 70이하일 경우 정지
    dist = np.sqrt(dx**2 + dy**2)
    if dist <= 72:
        speed = 0
    else:
        speed = 30

    # Stanley method를 사용해서 조향각 구하기
    angle = stanley_control(dx, dy, yaw, speed) * 100

    # 모터 제어 토픽을 발행: 시뮬레이터의 차량을 이동시킴
    xycar_msg.data = [angle, speed]
    motor_pub.publish(xycar_msg)
    arData["isUpdate"] = False

    # AR Tag 위치 표시 창 띄우기
    show_ar_tag_position(dx, dy, yaw)
    cv2.waitKey(10)

cv2.destroyAllWindows()
