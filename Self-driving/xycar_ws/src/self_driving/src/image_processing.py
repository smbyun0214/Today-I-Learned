#!/usr/bin/env python
#-*- coding: utf-8 -*-

import os
import numpy as np
import cv2 as cv
from utils.trackbar import Trackbar

# 파라미터 폴더 경로
src_path = os.path.dirname(os.path.join(__file__))
params_dir = os.path.join(src_path, "params", "image_processing")

tb_adap = Trackbar(os.path.join(params_dir, "adaptiveThreshold"), "adaptiveThreshold", debug=False)
tb_persp = Trackbar(os.path.join(params_dir, "perspective_transform"), "perspective_transform", debug=False)
tb_Canny = Trackbar(os.path.join(params_dir, "Canny"), "Canny", debug=False)
tb_Hough = Trackbar(os.path.join(params_dir, "HoughLinesP"), "HoughLinesP", debug=False)
tb_adap2 = Trackbar(os.path.join(params_dir, "adaptiveThreshold"), "adaptiveThreshold2", debug=False)


def processing(frame):
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (3, 3), 0)

    # Adaptive Threshold를 사용하여 이진화
    C = tb_adap.getValue("C")
    blockSize = tb_adap.getValue("blockSize")
    binary = cv.adaptiveThreshold(blur, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, blockSize, C)

    # blur = cv.GaussianBlur(binary, (3, 3), 0)
    # threshold1 = tb_Canny.getValue("threshold1")
    # threshold2 = tb_Canny.getValue("threshold1")
    # binary = cv.Canny(blur, threshold1, threshold2, apertureSize=3)

    # Perspective Transform 적용
    frame_height, frame_width = frame.shape[:2]
    margin = tb_persp.getValue("margin")
    roi_x1, roi_x3 = tb_persp.getValue("x1"), tb_persp.getValue("x3")
    roi_x2, roi_x4 = tb_persp.getValue("x2"), tb_persp.getValue("x4")
    roi_y1, roi_y2 = tb_persp.getValue("y1"), tb_persp.getValue("y2")
    roi_x, roi_y = max(0, min(roi_x1, roi_x2)), max(0, min(roi_y1, roi_y2))
    roi_width, roi_height = min(max(roi_x3, roi_x4), frame_width) - roi_x, max(0, roi_y2-roi_y1)

    tf_dst_size = tb_persp.getValue("size")
    tf_src_pts = np.array([[roi_x1, roi_y1], [roi_x2, roi_y2], [roi_x3, roi_y1], [roi_x4, roi_y2]], dtype=np.float32)
    tf_dst_pts = np.array([[0, 0], [0, tf_dst_size], [tf_dst_size, 0], [tf_dst_size, tf_dst_size]], dtype=np.float32)

    tf_matrix = cv.getPerspectiveTransform(tf_src_pts, tf_dst_pts)
    # tf_matrix_inv = cv.getPerspectiveTransform(dst_pts, src_pts)

    tf_image = cv.warpPerspective(binary, tf_matrix, (tf_dst_size, tf_dst_size), flags=cv.INTER_LINEAR)

    roi = [(roi_x1, roi_y1), (roi_x2, roi_y2), (roi_x4, roi_y2), (roi_x3, roi_y1), (roi_x1, roi_y1)]
    explain_image = cv.cvtColor(binary, cv.COLOR_GRAY2BGR)
    for pt1, pt2 in zip(roi[:-1], roi[1:]):
        cv.line(explain_image, pt1, pt2, (0, 0, 255))

    return tf_image, explain_image
