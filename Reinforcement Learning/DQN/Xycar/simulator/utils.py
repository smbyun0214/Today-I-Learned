# -*- coding: utf-8 -*-

import os
import numpy as np
import cv2 as cv


RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)


def get_roate_pos(pos, yaw):
    x, y = pos
    return (x * np.cos(yaw), y * np.sin(yaw))


def normalize_radian(rad):
    while rad < -np.pi:
        rad += 2 * np.pi
    while np.pi < rad:
        rad -= 2 * np.pi
    return rad


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
    return image


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
    return image


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
        if abs(gradient) == np.pi:
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


def is_episode_done(map, car, reward_domain=None):
    """지도상에서 차량의 상태를 확인하는 함수

    Args:
        map: 장애물이 있는 지도
        car: 차량 객체
    
    Returns:
        bool:
            - True: 장애물에 충돌
            - False: 길 위에 위치 또는 보상 영역 통과
        int: reward
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
                return True, -1

    if reward_domain is not None:
        for pt1, pt2, num in zip(border_points[:-1], border_points[1:], nums):
            xs = rint(np.linspace(pt1[0], pt2[0], num=num, endpoint=True))
            ys = rint(np.linspace(pt1[1], pt2[1], num=num, endpoint=True))

            for x, y in zip(xs, ys):
                if np.array_equal(reward_domain[y, x], [255, 255, 255]):
                    return False, np.sign(car.velocity).astype(np.int16)
    
    return False, 0



def get_reward_domain(map, car, radius=500):
    mask = np.zeros(map.shape, dtype=np.uint8)
    mask = cv.circle(mask, tuple(rint(car.position)), radius, (255, 255, 255))
    return mask
    

def draw_reward_domain(image, reward_domain):
    mask = np.where(reward_domain == [255, 255, 255], [0, 255, 255], reward_domain)

    mask_16 = mask.astype(np.int16)
    image_16 = image.astype(np.int16)

    image = np.clip(image_16 - mask_16, [0, 0, 0], [255, 255, 255]).astype(np.uint8)
    return image