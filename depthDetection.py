import numpy as np
import cv2

imgL = cv2.imread('C:\\Users\\aidan\\Documents\\BrevilleInternship\\Output\\Rectified\\left-right-L.jpg')
imgR = cv2.imread('C:\\Users\\aidan\\Documents\\BrevilleInternship\\Output\\Rectified\\left-right-R.jpg')
imgLO = cv2.imread('C:\\Users\\aidan\\Documents\\BrevilleInternship\\Input\\left.jpg')
imgRO = cv2.imread('C:\\Users\\aidan\\Documents\\BrevilleInternship\\Input\\right.jpg')


def resize(img, scale):
    return cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)))


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

dispO = left_matcher.compute(imgLO, imgRO).astype(np.float32) / 16.0
disp = left_matcher.compute(imgL, imgR).astype(np.float32) / 16.0
# disp = cv2.addWeighted(imgL, 0.5, imgR, 0.5, 0)
scale = 0.15
imgL = resize(imgL, scale)
imgR = resize(imgR, scale)
imgLO = resize(imgLO, scale)
imgRO = resize(imgRO, scale)
disp = resize(disp, scale)
dispO = resize(disp, scale)

cv2.imshow('left', imgL)
cv2.imshow('right', imgR)
cv2.imshow('left original', imgLO)
cv2.imshow('right original', imgRO)
cv2.imshow('disparity original', dispO)
cv2.imshow('disparity', disp)
cv2.waitKey()
cv2.destroyAllWindows()
