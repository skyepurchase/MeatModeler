import cv2
import numpy as np


class Processor:
    def __init__(self):
        pass

    def process(self, video):
        """
        Takes a video of a food item and returns the 3D mesh of the food item
        :param video: The video to be converted to a 3D mesh
        :return: A 3D mesh
        """
        cap = cv2.VideoCapture(video)
        tracker = cv2.TrackerMIL_create()

        success, frame = cap.read()

        if success:
            bbox = cv2.selectROI(frame, False)
            tracker.init(frame, bbox)

        success, frame = cap.read()

        while success:
            ok, bbox = tracker.update(frame)

            if ok:
                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)

            cv2.imshow("Tracking", frame)
            cv2.waitKey()

            success, frame = cap.read()

