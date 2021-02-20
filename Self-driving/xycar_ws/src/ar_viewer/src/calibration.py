import numpy as np
import cv2

cameraMatrix = np.array([
    [356.528087, 0.000000, 347.208505],
    [0.000000, 358.962305, 242.265865],
    [0.000000, 0.000000, 1.000000]])
distCoeffs = np.array([-0.311566, 0.074460, -0.003667, -0.001818, 0.000000])

width, height = 640, 480
imageSize = (width, height)
newImgSize = (width, height)

newCameraMatrix, calibrated_ROI = cv2.getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, imageSize, 1, newImgSize)

def calibrate_image(frame):
    tf_image = cv2.undistort(frame, cameraMatrix, distCoeffs, None, newCameraMatrix)
    roi_x, roi_y, roi_width, roi_height = calibrated_ROI
    
    output = np.zeros(tf_image.shape, dtype=np.uint8)
    output[roi_y:roi_y+roi_height, roi_x:roi_x+roi_width] = tf_image[roi_y:roi_y+roi_height, roi_x:roi_x+roi_width].copy()
    return output
