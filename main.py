import imageProcessor


def main():
    video_filename = "C:\\Users\\aidan\\Documents\\BrevilleInternship\\Input\\WhatsApp Video 2021-06-28 at 16.24.37.mp4"
    processor = imageProcessor.ImageProcessor("C:\\Users\\aidan\\Documents\\BrevilleInternship\\Output\\",
                                              60,
                                              threshold=15)
    processor.split(video_filename)


if __name__ == '__main__':
    main()
