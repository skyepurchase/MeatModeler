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

        success, frame = cap.read()
        while success:
            success, frame = cap.read()
