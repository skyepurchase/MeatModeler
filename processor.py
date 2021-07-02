import cv2
import numpy as np


class Processor:
    def __init__(self):
        self.feature_params = dict(maxCorners=100,
                                   qualityLevel=0.3,
                                   minDistance=7,
                                   blockSize=7)
        self.lk_params = dict(winSize=(15,15),
                              maxLevel=3,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        self.color = np.random.randint(0, 255, (100, 3))

    def process(self, video):
        """
        Takes a video of a food item and returns the 3D mesh of the food item
        :param video: The video to be converted to a 3D mesh
        :return: A 3D mesh
        """
        cap = cv2.VideoCapture(video)

        success, old_frame = cap.read()
        old_grey = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
        p0 = cv2.goodFeaturesToTrack(old_grey, mask=None, **self.feature_params)

        mask = np.zeros_like(old_frame)

        success, frame = cap.read()
        frame_grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        while success:
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_grey, frame_grey, p0, None, **self.lk_params)

            if p1 is not None:
                good_new = p1[st==1]
                good_old = p0[st==1]

            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()
                mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), self.color[i].tolist(), 2)
                frame = cv2.circle(frame, (int(a), int(b)), 5, self.color[i].tolist(), -1)

            img = cv2.add(frame, mask)

            cv2.imshow("Tracking", img)
            key = cv2.waitKey() & 0xff
            if key == 27:
                break

            old_grey = frame_grey.copy()
            p0 = good_new.reshape(-1, 1, 2)

            success, frame = cap.read()
            frame_grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
