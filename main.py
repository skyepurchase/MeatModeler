import imageProcessor


def main():
    video_filename = "C:\\Users\\aidan\\Documents\\BrevilleInternship\\Input\\WIN_20210628_14_49_54_Pro.mp4"
    processor = imageProcessor.ImageProcessor("C:\\Users\\aidan\\Documents\\BrevilleInternship\\Output\\")
    processor.split(video_filename)
    processor.recreateVideo()


if __name__ == '__main__':
    main()
