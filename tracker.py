import cv2
import numpy as np


class Tracker:
    def __init__(self):
        self.track_vector = None
        self.point_vector1 = None
        self.point_vector2 = None

    def isKeyFrame(self, frame):
        pass