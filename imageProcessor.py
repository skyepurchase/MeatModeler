import cv2
import os


class ImageProcessor:
    def __init__(self, path):
        self.path = path

    def split(self, video_path):
        """
        Splits a video into individual frames as .jpg images. All 'blurry' frames are removed during this process

        :param video_path: Fully qualified path to the video to be split.
        :return: Fully qualified path to the file containing the split frames.
        """
        # for file in os.listdir(self.path):
        #     os.remove(os.path.join(self.path, file))  # Remove the current images

        cam = cv2.VideoCapture(video_path)
        current_frame = 0

        ret, frame = cam.read()
        while ret:
            src = cv2.GaussianBlur(frame, (3, 3), 0)  # Remove noise by blurring image slightly
            src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
            value = cv2.Laplacian(src_gray, cv2.CV_64F).var()  # Apply Laplacian filter to get edges

            if value > 5:  # Suggests the edges are very weak and thus blurry
                filename = self.path + "Frame" + str(current_frame) + ".jpg"
                cv2.imwrite(filename, frame)
                current_frame += 1

            ret, frame = cam.read()

        return self.path

    def recreateVideo(self):
        """
        Reconstructs the video from the non-blurred images

        :return: The input folder contains the non-blurred video
        """
        filename = "C:\\Users\\aidan\\Documents\\BrevilleInternship\\Input\\UnblurredVideo.avi"

        img_array = []
        size = (500, 500)

        for img in os.listdir(self.path):
            src = cv2.imread(os.path.join(self.path, img))

            if src is None:
                print("Error loading image!")
                print(img)
                break

            height, width, layers = src.shape
            size = (width, height)
            img_array.append(src)

        out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'DIVX'), 30, size)

        img_array.reverse()
        for img in img_array:
            out.write(img)

        out.release()

    def getPath(self):
        return self.path
