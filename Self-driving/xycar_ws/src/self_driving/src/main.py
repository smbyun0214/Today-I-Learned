#!/usr/bin/env python
#-*- coding: utf-8 -*-

import yaml
import numpy as np
import cv2 as cv
from image_processing import processing
from sliding_window import sliding_window


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


filename = "/media/psf/version1/bag/rally.avi"

# 파일에서 영상 불러오기
cap = cv.VideoCapture(filename)

if not cap.isOpened():
    raise FileExistsError(filename)

# 영상 FPS, WIDTH, HEIGHT 불러오기
FPS = cap.get(cv.CAP_PROP_FPS)
WIDTH = cap.get(cv.CAP_PROP_FRAME_WIDTH)
HEIGHT = cap.get(cv.CAP_PROP_FRAME_HEIGHT)

rec = cv.VideoWriter("rally_ver_0_1.avi", cv.VideoWriter_fourcc(*"XVID"), FPS, (1125, 480))

while True:
    # 프레임 한개 불러오기
    ret, frame = cap.read()
    if not ret:
        break

    frame = calibrate_image(frame)

    processed_frame, explain1 = processing(frame)
    explain2 = sliding_window(processed_frame)

    # WIDTH 5 +
    vertical_line = np.zeros((explain1.shape[0], 5, 3), dtype=np.uint8)
    merge = np.hstack((explain1, vertical_line, explain2))
    rec.write(merge)

    cv.imshow("merge", merge)

    # Rendering
    delay = int(1000//FPS)
    key = cv.waitKey(delay)
    if key == 32:
        while key != cv.waitKey(delay):
            processed_frame, explain_image = processing(frame)
            sliding_window(processed_frame)
    elif key == ord('q'):
        break

rec.release()
cap.release()
