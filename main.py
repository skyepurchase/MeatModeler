import imageProcessor
import imageRectifier
import os

samples = 60
source = "C:\\Users\\aidan\\Documents\\BrevilleInternship\\Input\\"
raw_output = "C:\\Users\\aidan\\Documents\\BrevilleInternship\\Output\\Raw\\"
rectified_output = "C:\\Users\\aidan\\Documents\\BrevilleInternship\\Output\\Rectified\\"

def main():
    video_filename = source + "WhatsApp Video 2021-06-28 at 16.24.37.mp4"
    processor = imageProcessor.ImageProcessor(raw_output, 60)
    processor.split(video_filename)
    rectifier = imageRectifier.ImageRectifier(rectified_output)

    prev_src = ""
    for i in range(samples):
        if i > 0:
            rectifier.rectify(prev_src, raw_output + "Frame" + str(i) + ".jpg")
            print("Rectified", i)

        prev_src = raw_output + "Frame" + str(i) + ".jpg"


if __name__ == '__main__':
    main()
