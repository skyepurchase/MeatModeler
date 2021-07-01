import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt


def drawlines(img1, img2, lines, pts_in1, pts_in2):
    r, c = img1.shape
    img1 = cv.cvtColor(img1, cv.COLOR_GRAY2BGR)
    img2 = cv.cvtColor(img2, cv.COLOR_GRAY2BGR)

    for r, pt1, pt2 in zip(lines, pts_in1, pts_in2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2] / r[1]])
        x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1]])

        cv.line(img1, (x0, y0), (x1, y1), color, 1)
        cv.circle(img1, tuple(pt1), 5, color, -1)
        cv.circle(img2, tuple(pt2), 5, color, -1)

    return img1, img2


def getDescriptors(img):
    sift = cv.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(img, None)

    return keypoints, descriptors


class ImageRectifier:
    def __init__(self, path):
        self.path = path

    def calculateF(self, img1, img2):

        # find the keypoints and descriptors with SIFT
        kp1, des1 = getDescriptors(img1)
        kp2, des2 = getDescriptors(img2)

        # FLANN parameters
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)
        pts1 = []
        pts2 = []

        # ratio test as per Lowe's paper
        for i, (m, n) in enumerate(matches):
            if m.distance < 0.8 * n.distance:
                pts2.append(kp2[m.trainIdx].pt)
                pts1.append(kp1[m.queryIdx].pt)

        pts1 = np.int32(pts1)
        pts2 = np.int32(pts2)
        F, mask = cv.findFundamentalMat(pts1, pts2, cv.FM_LMEDS)

        if F is not None:
            # We select only inlier points
            pts1 = pts1[mask.ravel() == 1]
            pts2 = pts2[mask.ravel() == 1]

        return F, pts1, pts2

    def drawEpilines(self, img1, img2, data=None):
        if data is None or len(data) != 3:
            F, pts1, pts2 = self.calculateF(img1, img2)
            if F is None:
                return None
        else:
            F, pts1, pts2 = data

        lines1 = cv.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F)
        lines1 = lines1.reshape(-1, 3)
        img1_out, img2_dots = drawlines(img1, img2, lines1, pts1, pts2)

        lines2 = cv.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F)
        lines2 = lines2.reshape(-1, 3)
        img2_out, img1_dots = drawlines(img2, img1, lines2, pts2, pts1)

        return img1_out, img2_out

    def rectify(self, src1, src2, display_lines=False):
        img1 = cv.imread(src1)
        img2 = cv.imread(src2)
        F, pts1, pts2 = self.calculateF(img1, img2)

        if F is None:
            return None

        try:
            success, H1, H2 = cv.stereoRectifyUncalibrated(pts1, pts2, F, (img1.shape[1], img1.shape[0]))
            H1_inv = np.linalg.inv(H1)

            if display_lines:
                img1, img2 = self.drawEpilines(img1, img2, (F, pts1, pts2))

            img1_wrap = cv.warpPerspective(img1, H1, (img1.shape[1], img1.shape[0]))
            img2_wrap = cv.warpPerspective(img2, H2, (img2.shape[1], img2.shape[0]))
            img2_mosaic = cv.warpPerspective(img2_wrap, H1_inv, (img2_wrap.shape[1], img2_wrap.shape[0]))
            mosaic = cv.addWeighted(img1, 0.5, img2_mosaic, 0.5, 0)

            name1 = src1.split("\\")[-1].split(".")[0]
            name2 = src2.split("\\")[-1].split(".")[0]
            filename = self.path + "Mosaics\\" + name1 + "-" + name2 + "-mosaic.jpg"
            filenameL = self.path + name1 + "-" + name2 + "-L.jpg"
            filenameR = self.path + name1 + "-" + name2 + "-R.jpg"
            cv.imwrite(filename, mosaic)
            cv.imwrite(filenameL, img1_wrap)
            cv.imwrite(filenameR, img2_wrap)

            return img1_wrap, img2_wrap, mosaic
        except cv.error as e:
            # TODO: Deal with this mismatch error
            print(e)
            return None
