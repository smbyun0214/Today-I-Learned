#!/usr/bin/env python
#-*- coding: utf-8 -*-

import yaml
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from image_processing import processing
from sliding_window import sliding_window
from scipy.optimize import least_squares

with open("../calibration/usb_cam.yaml") as f:
    data = yaml.load(f)

width, height = data["image_width"], data["image_height"]

cameraMatrix = data["camera_matrix"]["data"]
rows, cols = data["camera_matrix"]["rows"], data["camera_matrix"]["cols"]
cameraMatrix = np.array(cameraMatrix).reshape((rows, cols))

distCoeffs = data["distortion_coefficients"]["data"]
rows, cols = data["distortion_coefficients"]["rows"], data["distortion_coefficients"]["cols"]
distCoeffs = np.array(distCoeffs).reshape((rows, cols))

imageSize = (width, height)
newImgSize = (width, height)

newCameraMatrix, calibrated_ROI = cv.getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, imageSize, 1, newImgSize)

def calibrate_image(frame):
    tf_image = cv.undistort(frame, cameraMatrix, distCoeffs, None, newCameraMatrix)
    roi_x, roi_y, roi_width, roi_height = calibrated_ROI
    tf_image = tf_image[roi_y:roi_y+roi_height, roi_x:roi_x+roi_width].copy()
    return cv.resize(tf_image, (frame.shape[1], frame.shape[0]))


filename1 = "/media/psf/version1/bag/rally.avi"
filename2 = "/media/psf/version1/track5.avi"

# 파일에서 영상 불러오기
cap1 = cv.VideoCapture(filename1)
cap2 = cv.VideoCapture(filename2)

if not cap1.isOpened():
    raise FileExistsError(filename1)
if not cap2.isOpened():
    raise FileExistsError(filename2)

# 영상 FPS, WIDTH, HEIGHT 불러오기
FPS = cap1.get(cv.CAP_PROP_FPS)
COUNT1 = cap1.get(cv.CAP_PROP_FRAME_COUNT)
COUNT2 = cap2.get(cv.CAP_PROP_FRAME_COUNT)

while True:
    # 프레임 한개 불러오기
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    if not ret1 or frame1 is None:
        cap1.set(cv.CAP_PROP_POS_FRAMES, 0)
        continue
    if not ret2 or frame2 is None:
        cap2.set(cv.CAP_PROP_POS_FRAMES, 0)
        continue

    POS1 = cap1.get(cv.CAP_PROP_POS_FRAMES)
    POS2 = cap2.get(cv.CAP_PROP_POS_FRAMES)

    def routine(frame, POS):
        frame = calibrate_image(frame)

        processed_frame, explain1 = processing(frame)
        choosen_left, choosen_right, explain2 = sliding_window(processed_frame)

        def fun(y, t, x):
            return y0[0]**2*t + y0[1]*t + y0[2] - x
            # return y0[0]**3*t + y0[1]**2*t + y0[2]*t + y0[3] - x

        for choosen in [choosen_left, choosen_right]:
            xs, ys = [], []

            for x, y in choosen:
                xs.append(x)
                ys.append(y)

            xs = np.array(xs)
            ys = np.array(ys)

            if len(xs) > 15 and len(ys) > 15:
                y0 = np.polyfit(ys, xs, 2)
                res_log = least_squares(fun, y0, loss='cauchy', f_scale=0.1, args=(xs, ys))

                f_3 = np.poly1d(res_log.x)

                new_ys = np.linspace(0, processed_frame.shape[0], num=processed_frame.shape[0], endpoint=True)
                new_xs = f_3(new_ys)

                for x, y in zip(new_xs, new_ys):
                    if 0 < y < processed_frame.shape[0] and 0 < x < processed_frame.shape[1]:
                        x, y = int(x), int(y)
                        cv.circle(explain2, (x, y), 3, (0, 255, 255), -1)

        # Merge Explain
        vertical_line = np.zeros((explain1.shape[0], 5, 3), dtype=np.uint8)
        explain_merge = np.hstack((explain2, vertical_line, explain1))
        cv.putText(explain_merge, "POS: {}".format(int(POS)), (0, 40), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        return explain_merge

    def merge_all(explain1_merge, explain2_merge):
        explain_merge_all = np.vstack((explain1_merge, explain2_merge))
        width = int(explain_merge_all.shape[1] * 0.8)
        height = int(explain_merge_all.shape[0] * 0.8)
        return cv.resize(explain_merge_all, (width, height), interpolation=cv.INTER_LINEAR)

    explain1_merge = routine(frame1, POS1)
    explain2_merge = routine(frame2, POS2)
    explain_merge_all = merge_all(explain1_merge, explain2_merge)
    cv.imshow("merge all", explain_merge_all)

    # Rendering
    delay = int(1000//FPS)
    key = cv.waitKeyEx(delay)

    cap2.set(cv.CAP_PROP_POS_FRAMES, POS2+5);

    if key == 32:
        while key != cv.waitKey(delay):
            explain1_merge = routine(frame1, POS1)
            explain2_merge = routine(frame2, POS2)
            explain_merge_all = merge_all(explain1_merge, explain2_merge)
            cv.imshow("merge all", explain_merge_all)
    elif key == 65361:  # left arrow key
        cap1.set(cv.CAP_PROP_POS_FRAMES, max(0, min((POS1-30 + COUNT1) % COUNT1, COUNT1-1)));
        cap2.set(cv.CAP_PROP_POS_FRAMES, max(0, min((POS2-30 + COUNT2) % COUNT2, COUNT2-1)));
    elif key == 65363:  # right arrow key
        cap1.set(cv.CAP_PROP_POS_FRAMES, max(0, min((POS1+30) % COUNT1, COUNT1-1)));
        cap2.set(cv.CAP_PROP_POS_FRAMES, max(0, min((POS2+30) % COUNT2, COUNT2-1)));
    elif key == ord('q'):
        break

cv.distoryAllWindows()
cap.release()
