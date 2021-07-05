import cv2
import numpy as np


def kp2pt(kp):
    print(kp.pt)
    return kp.pt


kp2pt_v = np.vectorize(kp2pt)


def increaseContrast(frame):
    """
    Increases the contrast of the grey scale images by applying CLAHE to the luminance

    :param frame: The frame to be editted
    :return: The increased contrast image
    """
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8))
    l_out = clahe.apply(l)
    lab_out = cv2.merge((l_out, a, b))

    return cv2.cvtColor(lab_out, cv2.COLOR_Lab2BGR)


class Processor:
    def __init__(self):
        self.feature_params = dict(maxCorners=100,
                                   qualityLevel=0.3,
                                   minDistance=7,
                                   blockSize=7)
        self.lk_params = dict(winSize=(15, 15),
                              maxLevel=3,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        self.flann_params = dict(algorithm=6,
                                 table_number=6,
                                 key_size=12,
                                 multi_probe_level=2)

        self.current_keyframe_grey = None
        self.current_keyframe_points = None

        self.current_orb_points = None
        self.current_orb_descriptors = None
        self.point_vector = []
        self.orb = cv2.ORB_create(nfeatures=2000)

        self.color = np.random.randint(0, 255, (100, 3))
        self.display = False
        self.count = 0

    def process(self, video, display=False):
        """
        Takes a video of a food item and returns the 3D mesh of the food item
        :param video: The video to be converted to a 3D mesh
        :param display: Whether the process should be displayed
        :return: A 3D mesh
        """
        # TODO: utilise FAST rather than goodFeaturesToTrack
        # TODO: implement Point Vector storage and comparison
        self.display = display

        cap = cv2.VideoCapture(video)

        # Extract features from the start frame
        _, start_frame = cap.read()
        self.current_orb_points, self.current_orb_descriptors = self.orb.detectAndCompute(start_frame, None)
        self.current_keyframe_grey = cv2.cvtColor(increaseContrast(start_frame), cv2.COLOR_BGR2GRAY)
        self.current_keyframe_points = cv2.goodFeaturesToTrack(self.current_keyframe_grey,
                                                               mask=None,
                                                               **self.feature_params)

        # Processing loop
        success, frame = cap.read()
        while success:
            if self.isKeyframe(frame):
                self.point_vector.append(self.calculateMatchedPoints(frame))
                self.current_keyframe_grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                self.current_keyframe_points = cv2.goodFeaturesToTrack(self.current_keyframe_grey,
                                                                       mask=None,
                                                                       **self.feature_params)

            success, frame = cap.read()

        print(self.point_vector)

    def isKeyframe(self, frame):
        """
        Determines whether a given frame is a keyframe for further analysis
        :param frame: The frame to be analysed
        :return: A boolean value on whether the frame was a keyframe
        """
        # Compare the last key frame to current key frame
        frame_grey = cv2.cvtColor(increaseContrast(frame), cv2.COLOR_BGR2GRAY)
        p1, st, err = cv2.calcOpticalFlowPyrLK(self.current_keyframe_grey,
                                               frame_grey,
                                               self.current_keyframe_points,
                                               None,
                                               **self.lk_params)

        # Keep only matching points
        if p1 is not None:
            if self.display:
                good_new = p1[st == 1]

            good_old = self.current_keyframe_points[st == 1]

        if self.display:
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()
                frame = cv2.line(frame, (int(a), int(b)), (int(c), int(d)), self.color[i].tolist(), 2)
                frame = cv2.circle(frame, (int(a), int(b)), 5, self.color[i].tolist(), -1)

            cv2.imshow("Tracking", frame)
            key = cv2.waitKey() & 0xff
            if key == 27:
                self.display = False

        # Current frame has deviated enough to be considered a key frame
        if err is not None and np.average(err) > 30:
            return True
        else:
            # Keep only visible points
            self.current_keyframe_points = good_old.reshape(-1, 1, 2)

            return False

    def calculateMatchedPoints(self, keyframe):
        """
        Finds which features in two keyframes match
        :param keyframe: The keyframe to compare to the previous keyframe
        :return: List of points
        """
        # TODO: convert to find new points efficiently
        # Get new points and descriptors
        new_points, new_descriptors = self.orb.detectAndCompute(keyframe, None)

        # FLANN based approach to find matches
        flann = cv2.FlannBasedMatcher(self.flann_params, {})
        matches = flann.knnMatch(self.current_orb_descriptors, new_descriptors, k=2)

        # Find which points can be considered matches
        good_matches = [match[0] for match in matches if
                        len(match) == 2 and match[0].distance < 0.7 * match[1].distance]

        # Update the keyframe points
        self.current_orb_points, self.current_orb_descriptors = new_points, new_descriptors

        return good_matches
