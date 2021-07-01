import imageProcessor
import depthDetection


def main():
    video_filename = "C:\\Users\\aidan\\Documents\\BrevilleInternship\\Input\\WhatsApp Video 2021-06-28 at 16.24.37.mp4"
    processor = imageProcessor.ImageProcessor("C:\\Users\\aidan\\Documents\\BrevilleInternship\\Output\\",
                                              60,
                                              blur_threshold=5,
                                              clahe_threshold=3.5)
    processor.split(video_filename)
    # depthDectector = depthDetection.DepthDetector("C:\\Users\\aidan\\Documents\\BrevilleInternship\\Output\\", 400)
    # depthDectector.run()


if __name__ == '__main__':
    main()
