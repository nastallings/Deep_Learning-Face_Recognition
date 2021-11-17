import cv2
import os


def generate_data():
    index = 0
    if len(os.listdir("Data/nstallings")) == 0:
        for video in os.listdir("Data/Videos"):
            vidcap = cv2.VideoCapture("Data/Videos/" + str(video))
            hasFrames = True
            while hasFrames:
                hasFrames, image = vidcap.read()
                if not hasFrames:
                    break
                cv2.imwrite(os.path.join("Data/nstallings", "nstallings_image_%d.png" % index), image)
                index += 1

