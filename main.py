import imageProcessor


def main():
    video_filename = "C:\\Users\\aidan\\Documents\\BrevilleInternship\\Input\\WIN_20210628_13_21_13_Pro.mp4"
    processor = imageProcessor.ImageProcessor("C:\\Users\\aidan\\Documents\\BrevilleInternship\\Output\\")
    processor.split(video_filename)


if __name__ == '__main__':
    main()
