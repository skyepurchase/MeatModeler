import cv2
import imageRectifier


class ImageProcessor:
    def __init__(self, path, count, suitability_threshold=250, contrast_threshold=1):
        self.path = path
        self.image_count = count + 1  # Because of indexing errors
        self.suitability_threshold = suitability_threshold
        self.contrast_threshold = contrast_threshold

    def split(self, video_path):
        """
        Splits a video into individual frames as .jpg images. All 'blurry' frames are removed during this process

        :param video_path: Fully qualified path to the video to be split.
        :return: Fully qualified path to the file containing the split frames.
        """
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        current_frame = 0
        image_number = 0

        success, frame = cap.read()
        while success and image_number != self.image_count:
            frame = cv2.cvtColor(self.increaseContrast(frame), cv2.COLOR_BGR2GRAY)

            if self.canSample(current_frame, image_number, total_frames) and self.isSuitable(frame):
                filename = self.path + "Frame" + str(image_number) + ".jpg"
                cv2.imwrite(filename, frame)
                image_number += 1

            current_frame += 1
            success, frame = cap.read()

        return self.path

    def canSample(self, current_frame, image_number, total_frames):
        """
        Determines whether the current frame can be sampled

        :param current_frame: The index of the frame being assessed
        :param image_number: The number of images chosen
        :param total_frames: Total number of possible frames
        :return: A boolean representing whether the frame can be sampled
        """
        # This greedy approach concentrates samples near the end
        # TODO: Find potentially better heuristic

        if image_number == self.image_count:
            return False

        frame_gap = (total_frames - current_frame) / (self.image_count - image_number)
        if current_frame < image_number * frame_gap:
            return False  # Current frame is before the sampling gap

        return True

    def isSuitable(self, frame):
        """
        Determines whether a given frame is too blurry for further processing

        :param frame: The video frame to be assessed
        :return: A boolean value representing whether the frame is blurred or not
        """
        keypoints, _ = imageRectifier.getDescriptors(frame)

        if len(keypoints) < self.suitability_threshold:  # Suggests the edges are not defined
            return False

        return True

    def increaseContrast(self, frame):
        """
        Increases the contrast of the grey scale images by applying CLAHE to the luminance

        :param frame: The frame to be editted
        :return: The increased contrast image
        """
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        clahe = cv2.createCLAHE(clipLimit=self.contrast_threshold, tileGridSize=(8, 8))
        l_out = clahe.apply(l)
        lab_out = cv2.merge((l_out, a, b))

        return cv2.cvtColor(lab_out, cv2.COLOR_Lab2BGR)

    def getPath(self):
        return self.path
