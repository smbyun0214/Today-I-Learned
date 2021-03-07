import os
import numpy as np
import cv2 as cv
from car import Car

def get_rotation_matrix(yaw):
    return np.array([
        [np.cos(yaw), np.sin(yaw)],
        [-np.sin(yaw), np.cos(yaw)]])


def rint(point):
    return np.rint(point).astype(np.int32)


RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)

def draw_car(image, car):
    """차량의 테두리와 바퀴를 그리는 함수

    Args:
        image: 배경 이미지
        car: 차량 객체
    """
    # 차량의 모서리 좌표
    points = [[
        [-car.border_back, -car.border_left],       # 좌측 후방
        [car.border_front, -car.border_left],       # 좌측 전방
        [car.border_front, car.border_right],       # 우측 전방
        [-car.border_back, car.border_right]        # 우측 후방
    ]]

    # 현재 차량 방향의 회전변환행렬
    mtx1 = get_rotation_matrix(car.yaw)

    # 회전이 적용된 차량의 모서리 좌표
    rotated_points = np.dot(points, mtx1)
    
    # 차량의 위치 적용
    rotated_points += car.position

    # 차량 그리기
    rotated_points = rint(rotated_points)
    cv.polylines(image, rotated_points, True, BLUE)


    # 바퀴 1개의 모서리 좌표
    border_points = [
        [-car.wheel_length/2, -car.wheel_width/2],
        [car.wheel_length/2, -car.wheel_width/2],
        [car.wheel_length/2, car.wheel_width/2],
        [-car.wheel_length/2, car.wheel_width/2]
    ]

    # 차량 기준, 바퀴 위치 좌표
    front_points = [
        [car.wheel_front, -car.wheel_left],     # 전방 좌측
        [car.wheel_front, car.wheel_right]      # 전방 우측
    ]
    back_points = [
        [-car.wheel_back, -car.wheel_left],     # 후방 좌측
        [-car.wheel_back, car.wheel_right]      # 후방 우측
    ]

    # 현재 바퀴 방향의 회전변환행렬
    steering_rad = np.radians(car.steering_deg)
    mtx2 = get_rotation_matrix(car.yaw + steering_rad)
    
    # 회전이 적용된 바퀴의 모서리 좌표
    rotated_border_points1 = np.dot(border_points, mtx1)  # 차량의 방향만 적용
    rotated_border_points2 = np.dot(border_points, mtx2)  # 차량의 방향 + steering_rad가 적용

    # 회전이 적용된 바퀴 위치 좌표
    rotated_front_points = np.dot(front_points, mtx1)
    rotated_back_points = np.dot(back_points, mtx1)

    # 모든 회전이 적용된 바퀴의 모서리 좌표
    rotated_front_points = np.array([ rotated_border_points2 + pt for pt in rotated_front_points ])
    rotated_back_points = np.array([ rotated_border_points1 + pt for pt in rotated_back_points ])

    # 차량의 위치 적용
    rotated_front_points += car.position
    rotated_back_points += car.position

    # 바퀴 그리기
    rotated_front_points = rint(rotated_front_points)
    rotated_back_points = rint(rotated_back_points)
    cv.fillPoly(image, rotated_front_points, RED)
    cv.polylines(image, rotated_back_points, True, RED)


def draw_car_detail(image, car):
    """어떤 기준으로 차량이 움직이는지 나타내는 함수

    Args:
        image: 배경 이미지
        car: 차량 객체
    """
    # 바퀴 1개의 모서리 좌표
    border_points = [
        [-car.wheel_length/2, -car.wheel_width/2],
        [car.wheel_length/2, -car.wheel_width/2],
        [car.wheel_length/2, car.wheel_width/2],
        [-car.wheel_length/2, car.wheel_width/2]
    ]

    # 차량 기준, 바퀴 위치 좌표
    front_points = [
        [car.wheel_front, 0]        # 전방 중앙
    ]
    back_points = [
        [-car.wheel_back, 0]        # 후방 중앙
    ]

    # 현재 차량 방향의 회전변환행렬
    mtx1 = get_rotation_matrix(car.yaw)

    # 현재 바퀴 방향의 회전변환행렬
    steering_rad = np.radians(car.steering_deg)
    mtx2 = get_rotation_matrix(car.yaw + steering_rad)
    
    # 회전이 적용된 바퀴의 모서리 좌표
    rotated_border_points1 = np.dot(border_points, mtx1)  # 차량의 방향만 적용
    rotated_border_points2 = np.dot(border_points, mtx2)  # 차량의 방향 + steering_rad가 적용

    # 회전이 적용된 바퀴 위치 좌표
    rotated_front_points = np.dot(front_points, mtx1)
    rotated_back_points = np.dot(back_points, mtx1)

    # 모든 회전이 적용된 바퀴의 모서리 좌표
    rotated_points = np.array(
        [ rotated_border_points1 + pt for pt in rotated_back_points ]   \
      + [ rotated_border_points2 + pt for pt in rotated_front_points ])

    # 차량의 위치 적용
    rotated_points += car.position

    # 바퀴 그리기
    rotated_points = rint(rotated_points)
    cv.polylines(image, rotated_points, True, GREEN)


    # 직선 좌표
    front_point1 = [0, -150]
    front_point2 = [0, 150]
    back_point1 = [0, -150]
    back_point2 = [0, 150]

    # 차량의 회전이 적용된 직선 좌표
    rotated_back_point1 = np.dot(back_point1, mtx1)
    rotated_back_point2 = np.dot(back_point2, mtx1)
    rotated_front_point1 = np.dot(front_point1, mtx2)
    rotated_front_point2 = np.dot(front_point2, mtx2)

    # 바퀴의 위치 적용
    rotated_front_point1 += rotated_front_points[0]
    rotated_front_point2 += rotated_front_points[0]
    rotated_back_point1 += rotated_back_points[0]
    rotated_back_point2 += rotated_back_points[0]

    # 차량의 위치 적용
    rotated_front_point1 += car.position
    rotated_front_point2 += car.position
    rotated_back_point1 += car.position
    rotated_back_point2 += car.position

    # 직선 그리기
    rotated_front_point1 = tuple(rint(rotated_front_point1))
    rotated_front_point2 = tuple(rint(rotated_front_point2))
    rotated_back_point1 = tuple(rint(rotated_back_point1))
    rotated_back_point2 = tuple(rint(rotated_back_point2))
    cv.line(image, rotated_front_point1, rotated_front_point2, RED)
    cv.line(image, rotated_back_point1, rotated_back_point2, RED)



if __name__ == "__main__":
    car = Car((500, 500), np.radians(0))
    gear = car.BREAK
    steering_deg = 0

    # 1초 = 1000ms
    # 30fps = 1000/30
    delay = 1000//30
    while True:
        background = np.ones((1000, 1000, 3), dtype=np.uint8) * 255
        draw_car(background, car)
        # draw_car_detail(background, car)
        cv.imshow("simulator", background)
        
        key = cv.waitKey(delay)
        if key == ord("q"):
            break
        elif key == ord("h"):
            steering_deg = -30
        elif key == ord("l"):
            steering_deg = 30
        elif key == ord("k"):
            gear = car.DRIVE
        elif key == ord("j"):
            gear = car.REVERSE
        car.update(1/30, gear, steering_deg)