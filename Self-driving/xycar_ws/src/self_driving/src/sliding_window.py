#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import cv2 as cv
import numpy as np
from utils.trackbar import Trackbar
from collections import deque

# 파라미터 폴더 경로
src_path = os.path.dirname(os.path.join(__file__))
params_dir = os.path.join(src_path, "params", "image_processing")

# Trackbar 초기화
tb_sliding_window = Trackbar(os.path.join(params_dir, "sliding_window"), "sliding_window", debug=True)

# delta_xs = [-1, -1, -1,  0,  0,  1,  1,  1]
# delta_ys = [-1,  0,  1, -1,  1, -1,  0,  1]

delta_xs = [-1, -1,  0,  1,  1]
delta_ys = [ 0, -1, -1,  0, -1]

def in_range(x, y, frame):
    return 0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]

def get_nonzero(center_x, center_y, scan_width, scan_height, frame):
    if not in_range(center_x, center_y, frame):
        return -1

    scan_half_width = scan_width // 2
    scan_half_height = scan_height // 2

    up = max(0, min(center_y - scan_half_height, frame.shape[0]))
    down = max(0, min(center_y + scan_half_height, frame.shape[0]))
    left = max(0, min(center_x - scan_half_width, frame.shape[1]))
    right = max(0, min(center_x + scan_half_width, frame.shape[1]))

    return np.count_nonzero(frame[up:down, left:right])


def sliding_window(frame):
    """
    1. 슬라이딩 윈도우를 진행할 때, 왼쪽/오른쪽 영역을 나눠서 각각 진행한다.
        - 윈도우 너비: tb_sliding_window.getValue("window_width")
        - 윈도우 높이: frame의 높이 / 윈도우의 갯수
    2. 다음 슬라이딩 윈도우를 진행하기 위해서는 윈도우 내에 있는 0이 아닌 픽셀 갯수가
       윈도우 넓이 theta 일 경우에 진행한다.
    """
    # 현재 프레임의 가로, 세로 길이와 중점 x좌표를 구한다.
    frame_height, frame_width = frame.shape[:2]
    frame_half_width = frame_width // 2
    frame_half_height = frame_width // 2

    # 윈도우 가로, 세로 길이를 구한다.
    num_of_windows = tb_sliding_window.getValue("num_of_windows")
    win_width = tb_sliding_window.getValue("win_width")
    win_height = frame_height // num_of_windows
    win_half_height = win_height // 2

    # 다음 슬라이딩을 판단하는 theta를 구한다.
    win_theta = tb_sliding_window.getValue("win_theta") / 100.0
    scan_theta = tb_sliding_window.getValue("scan_theta") / 100.0
    count_threshold = win_width * win_height * win_theta

    """
    가장 처음의 윈도우의 중심을 구하기 위해, 왼쪽/오른쪽 영역에서 scan 윈도우의 높이만큼 히스토그램을 확인한다.
    히스토그램에서 여러개의 peak를 사용하기 위해, 가장 큰 값의 50% 이상인 연속되는 x좌표들의 중심을 peak로 삼는다.
    """
    scan_height = tb_sliding_window.getValue("scan_height")
    scan_width = tb_sliding_window.getValue("scan_width")
    scan_half_height = scan_height // 2
    scan_half_width = scan_width // 2

    # 왼쪽/오른쪽 히스토그램 계산
    left_histogram = np.sum(frame[-scan_height:, :frame_half_width], axis=0)
    right_histogram = np.sum(frame[-scan_height:, frame_half_width:], axis=0)

    # 왼쪽 히스토그램에서 가장 큰 값의 50%를 가지는 x좌표를 뽑아내고, 그중 연속되는 x좌표들을 수집한다.
    step_size = tb_sliding_window.getValue("step_size")
    left_xs = np.argwhere(left_histogram > np.max(left_histogram)*0.5).flatten()
    left_xs_group = np.split(left_xs, np.where(np.diff(left_xs) > step_size)[0]+1)
    # 연속된 x좌표들의 중점 좌표를 수집한다.
    left_center_xs = [ np.mean(xs, dtype=np.int32) for xs in left_xs_group if xs.size != 0 ]

    # 오른쪽 히스토그램에서 가장 큰 값의 50%를 가지는 x좌표를 뽑아내고, 그중 연속되는 x좌표들을 수집한다.
    right_xs = np.argwhere(right_histogram > np.max(right_histogram)*0.5).flatten()
    right_xs_group = np.split(right_xs, np.where(np.diff(right_xs) > step_size)[0]+1)
    # 연속된 x좌표들의 중점 좌표를 수집하는데, 이때 오른쪽 영역을 의미하므로 현제 프레임 가로 길이도 더한다.
    right_center_xs = [ frame_half_width + np.mean(xs, dtype=np.int32) for xs in right_xs_group if xs.size != 0 ]


    """
    왼쪽 영역에서 scan한 점들의 flood fill
    """
    left_visited = { (x, frame_height - scan_half_height) for x in left_center_xs }
    left_pos_q = deque(left_visited)

    while len(left_pos_q) > 0:
        center_x, center_y = left_pos_q.popleft()

        for dx, dy in zip(delta_xs, delta_ys):
            next_center_x = center_x + scan_half_width*dx
            next_center_y = center_y + scan_half_height*dy

            nonzero = get_nonzero(next_center_x, next_center_y, scan_width, scan_height, frame)

            if nonzero > scan_width * scan_height * scan_theta   \
            and (next_center_x, next_center_y) not in left_visited:
                left_visited.add((next_center_x, next_center_y))
                left_pos_q.append((next_center_x, next_center_y))

    """
    오른쪽 영역에서 scan한 점들의 flood fill
    """
    right_visited = { (x, frame_height - scan_half_height) for x in right_center_xs }
    right_pos_q = deque(right_visited)

    while len(right_pos_q) > 0:
        center_x, center_y = right_pos_q.popleft()

        for dx, dy in zip(delta_xs, delta_ys):
            next_center_x = center_x + scan_half_width*dx
            next_center_y = center_y + scan_half_height*dy

            nonzero = get_nonzero(next_center_x, next_center_y, scan_width, scan_height, frame)

            if nonzero > scan_width * scan_height * scan_theta   \
            and (next_center_x, next_center_y) not in right_visited:
                right_visited.add((next_center_x, next_center_y))
                right_pos_q.append((next_center_x, next_center_y))

    explain_image = cv.cvtColor(frame, cv.COLOR_GRAY2BGR)
    cv.rectangle(explain_image, (0, frame_height-scan_height), (frame_width, frame_height), (0, 255, 0))

    for x, y in left_visited:
        cv.rectangle(explain_image, (x-scan_half_width, y-scan_half_height), (x+scan_half_width, y+scan_half_height), (255, 0, 0), 1)
        cv.circle(explain_image, (x, y), 2, (0, 0, 255), -1)

    for x, y in right_visited:
        cv.rectangle(explain_image, (x-scan_half_width, y-scan_half_height), (x+scan_half_width, y+scan_half_height), (255, 255, 0), 1)
        cv.circle(explain_image, (x, y), 2, (0, 255, 255), -1)

    return explain_image

    """
    슬라이딩 윈도우가 여러 갈래로 나뉘어 출력되므로, 각 슬라이딩의 결과 좌표를 담는 리스트가 필요하다.
    """
    #  탐색된 슬라이딩 윈도우의 중점 좌표들을 수집하는 리스트
    left_x_group, left_y_group, right_x_group, right_y_group = [], [], [], []

    viewer = cv.cvtColor(frame, cv.COLOR_GRAY2BGR)
    for win_center_x in left_centers:
        win_center_y = frame_height - win_half_height
        left_x, left_y = recursive_sliding_window(frame, win_center_x, win_center_y, win_width, win_height, count_threshold, scan_width, scan_height, step_size, viewer)

        left_x_group.append(left_x)
        left_y_group.append(left_y)


    # for left_x, left_y in zip(left_x_group, left_y_group):
    #     for x, y in zip(left_x, left_y):
    #         cv.circle(viewer, (x, y), 3, (0, 0, 255), 5)
    # tb_sliding_window.show(viewer)

    for win_center_x in right_centers:
        win_center_y = frame_height - win_half_height
        right_x, right_y = recursive_sliding_window(frame, win_center_x, win_center_y, win_width, win_height, count_threshold, scan_width, scan_height, step_size, viewer)

        right_x_group.append(right_x)
        right_y_group.append(right_y)


    left_coeff = [ np.polyfit(ly, lx, 2) for ly, lx in zip(left_y_group, left_x_group) if len(ly) > 3 and len(lx) > 3 ]
    right_coeff = [ np.polyfit(ry, rx, 2) for ry, rx in zip(right_y_group, right_x_group) if len(ry) > 3 and len(rx) > 3 ]

    return left_coeff, right_coeff#, viewer
