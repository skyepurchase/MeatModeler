import cv2
import numpy as np


def kp2pt(kp):
    print(kp.pt)
    return kp.pt


kp2pt_v = np.vectorize(kp2pt)


class Processor:
    def __init__(self):
        self.feature_params = dict(maxCorners=100,
                                   qualityLevel=0.3,
                                   minDistance=7,
                                   blockSize=7)
        self.lk_params = dict(winSize=(15, 15),
                              maxLevel=3,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        # self.color = np.random.randint(0, 255, (100, 3))
        # self.fast = cv2.FastFeatureDetector_create()

    def process(self, video):
        """
        Takes a video of a food item and returns the 3D mesh of the food item

        :param video: The video to be converted to a 3D mesh
        :return: A 3D mesh
        """
        # TODO: extract tracking code into separate function
        # TODO: utilise FAST rather than goodFeaturesToTrack
        # TODO: implement Point Vector storage and comparison
        cap = cv2.VideoCapture(video)

        # Extract features from the start frame
        success, start_frame = cap.read()
        keyframe_grey = cv2.cvtColor(start_frame, cv2.COLOR_BGR2GRAY)
        keyframe_p = cv2.goodFeaturesToTrack(keyframe_grey, mask=None, **self.feature_params)
        count = 0

        # mask = np.zeros_like(start_frame)

        success, frame = cap.read()

        while success:
            # Compare the last key frame to current key frame
            frame_grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            p1, st, err = cv2.calcOpticalFlowPyrLK(keyframe_grey, frame_grey, keyframe_p, None, **self.lk_params)

            # Keep only matching points
            if p1 is not None:
                # good_new = p1[st == 1]
                good_old = keyframe_p[st == 1]

            # for i, (new, old) in enumerate(zip(good_new, good_old)):
            #     a, b = new.ravel()
            #     c, d = old.ravel()
            #     frame = cv2.line(frame, (int(a), int(b)), (int(c), int(d)), self.color[i].tolist(), 2)
            #     frame = cv2.circle(frame, (int(a), int(b)), 5, self.color[i].tolist(), -1)
            #
            # img = cv2.add(frame, mask)
            #
            # cv2.imshow("Tracking", img)
            # key = cv2.waitKey() & 0xff
            # if key == 27:
            #     break

            if err is not None and np.average(err) > 30:
                # Current frame has deviated enough to be considered a key frame

                # Not going to be in final product
                filename = "C:\\Users\\aidan\\Documents\\BrevilleInternship\\Output\\Raw\\Image" + str(count) + ".jpg"
                cv2.imwrite(filename, keyframe_grey)
                count += 1

                # ReCalculate features and change to new keyframe
                keyframe_grey = frame_grey
                keyframe_p = cv2.goodFeaturesToTrack(keyframe_grey, mask=None, **self.feature_params)
            else:
                # Only want visible points
                keyframe_p = good_old.reshape(-1, 1, 2)

            success, frame = cap.read()
