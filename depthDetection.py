import numpy as np
import cv2

imgL = cv2.imread('C:\\Users\\aidan\\Documents\\BrevilleInternship\\Output\\test1.jpeg')
imgR = cv2.imread('C:\\Users\\aidan\\Documents\\BrevilleInternship\\Output\\test2.jpeg')

# disparity range is tuned for 'aloe' image pair
window_size = 3
min_disp = 32
num_disp = 116 - min_disp
stereo = cv2.StereoSGBM_create(minDisparity=min_disp,
                               numDisparities=num_disp,
                               uniquenessRatio=10,
                               speckleWindowSize=100,
                               speckleRange=32,
                               disp12MaxDiff=1,
                               P1=8 * 3 * window_size ** 2,
                               P2=32 * 3 * window_size ** 2
                               )

disp = stereo.compute(imgL, imgR).astype(np.float32) / 16.0

cv2.imshow('left', imgL)
cv2.imshow('right', imgR)
cv2.imshow('disparity', (disp - min_disp) / num_disp)
cv2.waitKey()
cv2.destroyAllWindows()
