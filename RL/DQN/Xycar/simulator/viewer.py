import os
import numpy as np
import cv2 as cv
from simulator.car import Car


RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)


def get_rotation_matrix(yaw):
    return np.array([
        [np.cos(yaw), np.sin(yaw)],
        [-np.sin(yaw), np.cos(yaw)]])


def rint(point):
    return np.rint(point).astype(np.int32)


def in_range(map, x, y):
    height, width = map.shape[:2]
    return 0 <= x < width and 0 <= y < height


def draw_car(image, car):
    """차량의 테두리와 바퀴를 그리는 함수

    Args:
        image: 배경 이미지
        car: 차량 객체
    """
    # 차량 그리기
    border_points = car.get_border_points()
    cv.polylines(image, rint([border_points]), True, BLUE)

    # 바퀴 그리기
    front_border_points = car.get_front_wheel_border_points()
    back_border_points = car.get_back_wheel_border_points()
    cv.fillPoly(image, rint(front_border_points), RED)
    cv.polylines(image, rint(back_border_points), True, RED)


def draw_ultrasonic(image, car, map):
    """차량에 부착된 초음파센서를 그리는 함수

    Args:
        image: 배경 이미지
        car: 차량 객체
    """
    start_points, end_points, yaws = get_ultrasonic_distance(map, car)

    # 초음파센서의 위치 그리기
    for pt1, theta in zip(start_points, yaws):
        cv.circle(image, tuple(rint(pt1)), 1, GREEN, 3)

    # 초음파센서의 탐색 거리 그리기
    for pt1, pt2 in zip(start_points, end_points):
        cv.line(image, tuple(rint(pt1)), tuple(rint(pt2)), GREEN)


def get_ultrasonic_distance(map, car):
    """차량에 부착된 초음파센서가 거리를 측정할 때, 탐색 영역의 끝점을 계산하는 함수

    Args:
        map: 장애물이 있는 지도
        car: 차량 객체
    """
    start_points, yaws = car.get_ultrasonic_pos_and_yaw()
    end_points = []
    
    for (x1, y1), yaw in zip(start_points, yaws):
        # 초음파 탐색 기울기
        gradient = np.tan(yaw)

        # 초음파 탐색 좌표 계산
        if abs(gradient) >= car.max_ultrasonic_seek_pixel:
            xs = np.full(rint(car.max_ultrasonic_seek_pixel), x1)
            ys = np.linspace(y1, y1 + np.sign(gradient)*car.max_ultrasonic_seek_pixel, num=rint(car.max_ultrasonic_seek_pixel), endpoint=True)
        else:
            xs = np.linspace(
                x1, x1 + car.max_ultrasonic_seek_pixel*np.cos(yaw),
                num=rint(car.max_ultrasonic_seek_pixel), endpoint=True)
            ys = gradient*(xs - x1) + y1

        # 초음파가 장애물에 부딪힌 지점 계산
        prev_x2 = xs[0]
        prev_y2 = ys[0]
        for (x2, y2) in zip(xs, ys):
            x2 = rint(x2)
            y2 = rint(y2)
            # 맵에 있는 충돌 영역에 맞닿아있을 경우, 해당 좌표를 부딪힌 지점으로 설정
            if in_range(map, x2, y2):
                if not np.array_equal(map[y2, x2], [255, 255, 255]):
                    end_points.append((x2, y2))
                    break
            # 부딪히는 지점이 없을 경우, max_ultrasonic_seek_pixel만큼 떨어진 위치를 한계 탐색 지점으로 설정
            else:
                end_points.append((prev_x2, prev_y2))
                break
            prev_x2 = x2
            prev_y2 = y2
        else:
            end_points.append((xs[-1], ys[-1]))

    end_points = np.array(end_points)
    return start_points, end_points, yaws


def is_collision(map, car, road=[255, 255, 255]):
    """지도상에서 차량의 상태를 확인하는 함수

    Args:
        map: 장애물이 있는 지도
        car: 차량 객체
        road: 허용된 길의 색상값(BGR)
    
    Returns:
        bool: False: 길 위에 있음, True: 장애물에 충돌
    """
    border_points = car.get_border_points()

    # 차량 모서리 좌표를 한개 더 추가
    border_points = np.append(border_points, [border_points[0]], axis=0)
    
    # 각 면에서 충돌 검사에 사용할 점의 갯수
    nums = [
        car.border_length + 1,      # 좌측 길이
        car.border_width + 1,       # 전방 폭
        car.border_length + 1,      # 우측 길이
        car.border_width + 1,       # 후방 폭
    ]

    # 점과 맵의 충돌 영역 검사
    for pt1, pt2, num in zip(border_points[:-1], border_points[1:], nums):
        xs = rint(np.linspace(pt1[0], pt2[0], num=num, endpoint=True))
        ys = rint(np.linspace(pt1[1], pt2[1], num=num, endpoint=True))

        for x, y in zip(xs, ys):
            if in_range(map, x, y) and not np.array_equal(map[y, x], [255, 255, 255]):
                return True
    return False



if __name__ == "__main__":
    car = Car((500, 500), np.radians(0))
    gear = car.BREAK
    steering_deg = 0

    background_origin = cv.imread("map/rally_map3.png")
    # 1초 = 1000ms
    # 30fps = 1000/30
    delay = 1000//30
    while True:
        background = background_origin.copy()
        draw_car(background, car)
        # draw_car_detail(background, car)
        # draw_ultrasonic(background, car, background_origin)            
        cv.imshow("simulator", background)

        if is_collision(background_origin, car):
            car.position = (500, 500)
            car.yaw = np.radians(0)

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