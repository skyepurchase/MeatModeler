import imageProcessor


def main():
    video_filename = "C:\\Users\\aidan\\Documents\\BrevilleInternship\\Input\\WhatsApp Video 2021-06-28 at 15.40.21.mp4"
    processor = imageProcessor.ImageProcessor("C:\\Users\\aidan\\Documents\\BrevilleInternship\\Output\\")
    processor.split(video_filename)
    processor.recreateVideo()


if __name__ == '__main__':
    main()
