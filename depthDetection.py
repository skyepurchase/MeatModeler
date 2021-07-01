import numpy as np
import cv2

imgL = cv2.imread('C:\\Users\\aidan\\Documents\\BrevilleInternship\\Output\\Rectified\\left-right-L.jpg')
imgR = cv2.imread('C:\\Users\\aidan\\Documents\\BrevilleInternship\\Output\\Rectified\\left-right-R.jpg')

# disparity range is tuned for 'aloe' image pair
window_size = 3
left_matcher = cv2.StereoSGBM_create(minDisparity=0,
                                     numDisparities=160,
                                     blockSize=5,
                                     uniquenessRatio=10,
                                     speckleWindowSize=0,
                                     speckleRange=2,
                                     disp12MaxDiff=1,
                                     P1=8 * 3 * window_size ** 2,
                                     P2=32 * 3 * window_size ** 2,
                                     preFilterCap=63,
                                     mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
                                     )

disp = left_matcher.compute(imgL, imgR).astype(np.float32) / 16.0
disp = cv2.addWeighted(imgL, 0.5, imgR, 0.5, 0)
# imgL = cv2.resize(imgL, (int(imgL.shape[1] * 0.1), int(imgL.shape[0] * 0.1)))
# imgR = cv2.resize(imgR, (int(imgR.shape[1] * 0.1), int(imgR.shape[0] * 0.1)))
# disp = cv2.resize(disp, (int(disp.shape[1] * 0.1), int(disp.shape[0] * 0.1)))

cv2.imshow('left', imgL)
cv2.imshow('right', imgR)
cv2.imshow('disparity', disp)
cv2.waitKey()
cv2.destroyAllWindows()
