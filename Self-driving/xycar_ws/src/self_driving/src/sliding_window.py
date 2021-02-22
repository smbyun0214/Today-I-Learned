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


def in_range(x, y, frame):
    return 0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]


def get_nonzero(center_x, center_y, scan_width, scan_height, frame):
    scan_half_width = scan_width // 2
    scan_half_height = scan_height // 2

    up = max(0, min(center_y - scan_half_height, frame.shape[0]))
    down = max(0, min(center_y + scan_half_height, frame.shape[0]))
    left = max(0, min(center_x - scan_half_width, frame.shape[1]))
    right = max(0, min(center_x + scan_half_width, frame.shape[1]))

    return np.count_nonzero(frame[up:down, left:right])


def get_peak_xs(center_x, center_y, scan_width, scan_height, hist_threshold, step_size, frame):
    if not in_range(center_x, center_y, frame):
        return []

    scan_half_width = scan_width // 2
    scan_half_height = scan_height // 2

    up = max(0, min(center_y - scan_half_height, frame.shape[0]))
    down = max(0, min(center_y + scan_half_height, frame.shape[0]))
    left = max(0, min(center_x - scan_half_width, frame.shape[1]))
    right = max(0, min(center_x + scan_half_width, frame.shape[1]))

    # 히스토그램 계산
    histogram = np.sum(frame[up:down, left:right], axis=0)

    # 히스토그램에서 가장 큰 값*hist_threshold 이상인  x좌표를 뽑아내고, 그중 연속되는 x좌표들을 수집한다.
    xs = np.argwhere(histogram > np.max(histogram)*hist_threshold).flatten()
    xs_group = np.split(xs, np.where(np.diff(xs) > step_size)[0]+1)

    # 연속된 x좌표들의 중점 좌표를 수집한다.
    xs_mean = [ np.mean(xs, dtype=np.int32) for xs in xs_group if xs.size != 0 ]
    peak_xs = [ left + x for x in xs_mean ]

    return peak_xs


def get_peak_ys(center_x, center_y, scan_width, scan_height, hist_threshold, step_size, frame):
    if not in_range(center_x, center_y, frame):
        return []

    scan_half_width = scan_width // 2
    scan_half_height = scan_height // 2

    up = max(0, min(center_y - scan_half_height, frame.shape[0]))
    down = max(0, min(center_y + scan_half_height, frame.shape[0]))
    left = max(0, min(center_x - scan_half_width, frame.shape[1]))
    right = max(0, min(center_x + scan_half_width, frame.shape[1]))

    # 히스토그램 계산
    histogram = np.sum(frame[up:down, left:right], axis=1)

    # 히스토그램에서 가장 큰 값*hist_threshold 이상인  x좌표를 뽑아내고, 그중 연속되는 x좌표들을 수집한다.
    ys = np.argwhere(histogram > np.max(histogram)*hist_threshold).flatten()
    ys_group = np.split(ys, np.where(np.diff(ys) > step_size)[0]+1)

    # 연속된 x좌표들의 중점 좌표를 수집한다.
    ys_mean = [ np.mean(ys, dtype=np.int32) for ys in ys_group if ys.size != 0 ]
    peak_ys = [ up + y for y in ys_mean ]

    return peak_ys


def get_floodfill(cache, center_x, center_y, scan_width, scan_height, scan_theta, hist_threshold, step_size, frame):
    """
    영역에서 scan한 점들의 flood fill
    """
    cache[(center_x, center_y)] = 0
    visited = { (center_x, center_y) }

    dist = 0
    queue = deque()
    queue.append((center_x, center_y, 0))

    while len(queue) > 0:
        center_x, center_y, cost = queue.popleft()

        if cache[(center_x, center_y)] > 30:
            continue

        cache[(center_x, center_y)] = cost
        dist = max(dist, cost)

        next_center_y = center_y - scan_height//2
        cand_peak_xs = get_peak_xs(center_x, next_center_y, scan_width*2, scan_height, hist_threshold, step_size, frame)

        for next_center_x in cand_peak_xs:
            if not in_range(next_center_x, next_center_y, frame):
                continue

            nonzero = get_nonzero(next_center_x, next_center_y, scan_width, scan_height, frame)

            if  nonzero > scan_width * scan_height * scan_theta \
            and (next_center_x, next_center_y) not in cache:
                visited.add((next_center_x, next_center_y))
                cache[(next_center_x, next_center_y)] = cost + 1
                queue.append((next_center_x, next_center_y, cost+1))

    return cache, visited, dist


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

    # 다음 슬라이딩을 판단하는 theta를 구한다.
    scan_theta = tb_sliding_window.getValue("scan_theta") / 100.0

    """
    가장 처음의 윈도우의 중심을 구하기 위해, 왼쪽/오른쪽 영역에서 scan 윈도우의 높이만큼 히스토그램을 확인한다.
    히스토그램에서 여러개의 peak를 사용하기 위해, 가장 큰 값의 50% 이상인 연속되는 x좌표들의 중심을 peak로 삼는다.
    """
    scan_height = tb_sliding_window.getValue("scan_height")
    scan_width = tb_sliding_window.getValue("scan_width")
    scan_half_height = scan_height // 2
    scan_half_width = scan_width // 2

    """
    ------------------------------
    |                            |
    |                            |
    |                            |
    |x|                        |x| x: range_edge_scan 영역
    |x|                        |x|
    |x|----------|  |----------|x| - range_scan 영역
    |x|----------|  |----------|x|

    1. 왼쪽 절반/오른쪽 절반 중점 검사
    2. 중점이 init_scan 영역에 있을 경우 시작점으로 선정
        2.1. 시작점이 없으면 왼족 Edge/오른쪽 Edge의 중점 검사
        2.2. Edge 중점 중 init_edge_scan 영역의 가장 하단 중점을 시작점으로 선정
    """
    step_size = tb_sliding_window.getValue("step_size")
    hist_threshold = tb_sliding_window.getValue("histogram_threshold", "default", 0, 70, 100) / 100.0

    range_scan_l = scan_half_width
    range_scan_r = frame_half_width - scan_half_width
    range_scan_u = frame_height - scan_height
    range_scan_d = frame_height

    range_scan_width = range_scan_r - range_scan_l
    range_scan_height = range_scan_d - range_scan_u
    range_scan_half_width = range_scan_width // 2
    range_scan_half_height = range_scan_height // 2

    range_edge_scan_l = 0
    range_edge_scan_r = scan_width
    range_edge_scan_u = frame_height//4*3
    range_edge_scan_d = frame_height

    range_edge_scan_width = range_edge_scan_r - range_edge_scan_l
    range_edge_scan_height = range_edge_scan_d - range_edge_scan_u
    range_edge_scan_half_width = range_edge_scan_width // 2
    range_edge_scan_half_height = range_edge_scan_height // 2

    """
    왼쪽 영역/오른쪽 영역에서 scan한 점들의 flood fill
    """
    left_peak_xs = get_peak_xs(frame_half_width//2, frame_height-range_scan_half_height,
                               range_scan_width, range_scan_height, hist_threshold, step_size, frame)
    right_peak_xs = get_peak_xs(frame_half_width//2*3, frame_height-range_scan_half_height,
                                range_scan_width, range_scan_height, hist_threshold, step_size, frame)

    left_cache, right_cache = {}, {}
    left_visited, right_visited = [], []
    left_dist, right_dist = 0, 0

    center_y = frame_height - range_scan_half_height

    for center_x in reversed(left_peak_xs):
        left_cache, visited, dist = get_floodfill(left_cache, center_x, center_y,       \
                                                  scan_width, scan_height, scan_theta,  \
                                                  hist_threshold, step_size, frame)
        left_visited.append(visited)
        left_dist = max(left_dist, dist)


    for center_x in right_peak_xs:
        right_cache, visited, dist = get_floodfill(right_cache, center_x, center_y,     \
                                                   scan_width, scan_height, scan_theta, \
                                                   hist_threshold, step_size, frame)
        right_visited.append(visited)
        right_dist = max(right_dist, dist)


    """
    왼쪽 영역/오른쪽 영역에서 검출한 차선의 최대 길이가 10이하인 경우 측면 차선 검출
    """
    if left_dist <= 15:
        left_peak_ys = get_peak_ys(range_edge_scan_half_width, frame_height-range_edge_scan_half_height,
                                   range_edge_scan_width, range_edge_scan_height, hist_threshold, step_size, frame)

        center_x = range_edge_scan_half_width
        for center_y in reversed(left_peak_ys):
            left_cache, visited, _ = get_floodfill(left_cache, center_x, center_y,  \
                                            scan_width, scan_height, scan_theta,    \
                                            hist_threshold, step_size, frame)
            left_visited.append(visited)

    if right_dist <= 15:
        right_peak_ys = get_peak_ys(frame_width-range_edge_scan_half_width, frame_height-range_edge_scan_half_height,
                                   range_edge_scan_width, range_edge_scan_height, hist_threshold, step_size, frame)

        center_x = frame_width - range_edge_scan_half_width
        for center_y in right_peak_ys:
            right_cache, visited, _ = get_floodfill(right_cache, center_x, center_y,\
                                             scan_width, scan_height, scan_theta,   \
                                             hist_threshold, step_size, frame)
            right_visited.append(visited)

    choosen_left = max(left_visited, key=lambda v: len(v)) if len(left_visited) else []
    choosen_right = max(right_visited, key=lambda v: len(v)) if len(right_visited) else []


    """
    Explain Image
    """
    explain_image = cv.cvtColor(frame, cv.COLOR_GRAY2BGR)

    cv.rectangle(explain_image,
                (range_scan_l, frame_height-range_scan_height),
                (range_scan_r, frame_height),
                (0, 255, 0))
    cv.rectangle(explain_image,
                (frame_half_width+range_scan_l, frame_height-range_scan_height),
                (frame_half_width+range_scan_r, frame_height),
                (0, 255, 0))

    cv.rectangle(explain_image,
                (0, range_edge_scan_u),
                (range_edge_scan_width, range_edge_scan_d),
                (0, 255, 0))
    cv.rectangle(explain_image,
                (frame_width-range_edge_scan_width, range_edge_scan_u),
                (frame_width, range_edge_scan_d),
                (0, 255, 0))

    for x, y in choosen_left:
        cv.rectangle(explain_image, (x-scan_half_width, y-scan_half_height), (x+scan_half_width, y+scan_half_height), (255, 0, 0), 1)
        cv.circle(explain_image, (x, y), 2, (0, 0, 255), -1)

    for x, y in choosen_right:
        cv.rectangle(explain_image, (x-scan_half_width, y-scan_half_height), (x+scan_half_width, y+scan_half_height), (255, 255, 0), 1)
        cv.circle(explain_image, (x, y), 2, (0, 255, 255), -1)

    cv.putText(explain_image, str(left_dist), (0, 100), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv.putText(explain_image, str(right_dist), (430, 100), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    return choosen_left, choosen_right, explain_image
    #
    # """
    # 슬라이딩 윈도우가 여러 갈래로 나뉘어 출력되므로, 각 슬라이딩의 결과 좌표를 담는 리스트가 필요하다.
    # """
    # #  탐색된 슬라이딩 윈도우의 중점 좌표들을 수집하는 리스트
    # left_x_group, left_y_group, right_x_group, right_y_group = [], [], [], []
    #
    # viewer = cv.cvtColor(frame, cv.COLOR_GRAY2BGR)
    # for win_center_x in left_centers:
    #     win_center_y = frame_height - win_half_height
    #     left_x, left_y = recursive_sliding_window(frame, win_center_x, win_center_y, win_width, win_height, count_threshold, scan_width, scan_height, step_size, viewer)
    #
    #     left_x_group.append(left_x)
    #     left_y_group.append(left_y)
    #
    #
    # # for left_x, left_y in zip(left_x_group, left_y_group):
    # #     for x, y in zip(left_x, left_y):
    # #         cv.circle(viewer, (x, y), 3, (0, 0, 255), 5)
    # # tb_sliding_window.show(viewer)
    #
    # for win_center_x in right_centers:
    #     win_center_y = frame_height - win_half_height
    #     right_x, right_y = recursive_sliding_window(frame, win_center_x, win_center_y, win_width, win_height, count_threshold, scan_width, scan_height, step_size, viewer)
    #
    #     right_x_group.append(right_x)
    #     right_y_group.append(right_y)
    #
    #
    # left_coeff = [ np.polyfit(ly, lx, 2) for ly, lx in zip(left_y_group, left_x_group) if len(ly) > 3 and len(lx) > 3 ]
    # right_coeff = [ np.polyfit(ry, rx, 2) for ry, rx in zip(right_y_group, right_x_group) if len(ry) > 3 and len(rx) > 3 ]
    #
    # return left_coeff, right_coeff#, viewer
