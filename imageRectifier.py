import numpy as np
import cv2


def getDescriptors(img):
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(img, None)
    print(descriptors)

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
        flann = cv2.FlannBasedMatcher(index_params, search_params)
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
        F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_LMEDS)

        if F is not None:
            # We select only inlier points
            pts1 = pts1[mask.ravel() == 1]
            pts2 = pts2[mask.ravel() == 1]

        return F, pts1, pts2

    def rectify(self, src1, src2):
        img1 = cv2.imread(src1)
        img2 = cv2.imread(src2)
        F, pts1, pts2 = self.calculateF(img1, img2)

        if F is None:
            return None

        try:
            success, H1, H2 = cv2.stereoRectifyUncalibrated(pts1, pts2, F, (img1.shape[1], img1.shape[0]))
            H1_inv = np.linalg.inv(H1)

            img1_wrap = cv2.warpPerspective(img1, H1, (img1.shape[1], img1.shape[0]))
            img2_wrap = cv2.warpPerspective(img2, H2, (img2.shape[1], img2.shape[0]))
            img2_mosaic = cv2.warpPerspective(img2_wrap, H1_inv, (img1.shape[1], img1.shape[0]))
            mosaic = cv2.addWeighted(img1, 0.5, img2_mosaic, 0.5, 0)

            name1 = src1.split("\\")[-1].split(".")[0]
            name2 = src2.split("\\")[-1].split(".")[0]
            filename = self.path + "Mosaics\\" + name1 + "-" + name2 + ".jpg"
            filenameL = self.path + name1 + "-" + name2 + "-L.jpg"
            filenameR = self.path + name1 + "-" + name2 + "-R.jpg"
            cv2.imwrite(filename, mosaic)
            cv2.imwrite(filenameL, img1_wrap)
            cv2.imwrite(filenameR, img2_wrap)

            return img1_wrap, img2_wrap
        except cv2.error as e:
            # TODO: Deal with this mismatch error
            print(e)
            return None
