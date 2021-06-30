import cv2


class ImageProcessor:
    def __init__(self, path, count, blur_threshold=5, clahe_threshold=3.5):
        self.path = path
        self.image_count = count + 1  # Because of indexing errors
        self.blur_threshold = blur_threshold
        self.clahe_threshold = clahe_threshold

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

            if self.canSample(current_frame, image_number, total_frames) and self.isBlurry(frame):
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

    def isBlurry(self, frame):
        """
        Determines whether a given frame is too blurry for further processing

        :param frame: The video frame to be assessed
        :return: A boolean value representing whether the frame is blurred or not
        """
        src = cv2.GaussianBlur(frame, (3, 3), 0)  # Remove noise by blurring image slightly
        value = cv2.Laplacian(src, cv2.CV_64F).var()  # Apply Laplacian filter to get edges

        if value > self.blur_threshold:  # Suggests the edges are defined
            return True

        return False

    def increaseContrast(self, frame):
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        clahe = cv2.createCLAHE(clipLimit=self.clahe_threshold, tileGridSize=(8, 8))
        l_out = clahe.apply(l)
        lab_out = cv2.merge((l_out, a, b))

        return cv2.cvtColor(lab_out, cv2.COLOR_Lab2BGR)

    def getPath(self):
        return self.path
