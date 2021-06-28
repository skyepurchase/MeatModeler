import cv2
import os


class ImageProcessor:
    def __init__(self, path, count, threshold=5):
        self.path = path
        self.image_count = count
        self.threshold = threshold

    def split(self, video_path):
        """
        Splits a video into individual frames as .jpg images. All 'blurry' frames are removed during this process

        :param video_path: Fully qualified path to the video to be split.
        :return: Fully qualified path to the file containing the split frames.
        """
        # for file in os.listdir(self.path):
        #     os.remove(os.path.join(self.path, file))  # Remove the current images

        cam = cv2.VideoCapture(video_path)
        total_frames = int(cam.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_gap = int(total_frames / self.image_count)

        current_frame = 0
        current_frame_offset = 0

        frame_number = 0
        frame_number_offset = 0

        ret, frame = cam.read()
        while ret:
            adjusted_frame_number = frame_number - frame_number_offset
            adjusted_current_frame = current_frame - current_frame_offset

            if adjusted_current_frame > adjusted_frame_number * frame_gap:  # If the sample is coming from the right gap

                if adjusted_current_frame >= (adjusted_frame_number + 1) * frame_gap:
                    # An entire gap was too blurry -> readjust future choices to get enough samples
                    # Heuristic favours later images, can mean not enough samples chosen or object not covered
                    # TODO: Find a better heuristic that doesn't require processing every image
                    frame_gap = int((total_frames - current_frame) / (self.image_count - frame_number))
                    frame_number_offset = frame_number
                    current_frame_offset = current_frame

                src = cv2.GaussianBlur(frame, (3, 3), 0)  # Remove noise by blurring image slightly
                src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
                value = cv2.Laplacian(src_gray, cv2.CV_64F).var()  # Apply Laplacian filter to get edges

                if value > self.threshold:  # Suggests the edges are very weak and thus blurry
                    filename = self.path + "Frame" + str(frame_number) + ".jpg"
                    cv2.imwrite(filename, frame)
                    frame_number += 1

            current_frame += 1
            ret, frame = cam.read()

        return self.path

    def getPath(self):
        return self.path
