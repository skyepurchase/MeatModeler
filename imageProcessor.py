import cv2
import os


class ImageProcessor:
    def __init__(self, path):
        self.path = path

    def split(self, video_path):
        """
        Splits a video into individual frames as .jpg images.

        :param video_path: Fully qualified path to the video to be split.
        :return: Fully qualified path to the file containing the split frames.
        """
        for file in os.listdir(self.path):
            os.remove(os.path.join(self.path, file))  # Remove the current images

        cam = cv2.VideoCapture(video_path)
        current_frame = 0

        ret, frame = cam.read()
        while ret:
            filename = self.path + "Frame" + str(current_frame) + ".jpg"
            print(filename)
            cv2.imwrite(filename, frame)
            current_frame += 1
            ret, frame = cam.read()

        return self.path

    def getPath(self):
        return self.path
